'use strict';
/*
 * video_local.js: local video models (MiniMax H3...) through stable-diffusion.cpp's sd-server, native async API /sdcpp/v1/vid_gen:
 * the image engine's twin (image_local.js), line for line, so play loads it and stop unloads it the same way.
 * sd-server lives only while a model is loaded: spawned on 127.0.0.1 on first use or on request, killed on
 * unload, after idle_min without use, or when ia_server exits. One model at a time, one job at a time.
 * Clips (video with its audio) are delivered as the industry does, MP4 with H.264 and AAC (~10x smaller than sd-server's
 * webm, and it plays everywhere, iPhone included): vid/<32 hex>.mp4, kept 24 h and served by the gateway at /vid/<32 hex>.mp4.
 */
const { spawn, execFile } = require('child_process');
const http = require('http'), net = require('net');
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const { fetchImage, loraDir, genOf } = require('./image_local');

const DIR = path.join(__dirname, 'vid'), LOG = path.join(__dirname, 'sd-server-video.log'), DAY = 864e5;
// limits in ms (load, job) and pending jobs (running + waiting); a job never outlives load + job + 30 s
const T = { load: 900000, job: 5400000, queue: 2 };
let C = { bin: '', idle_min: 10, models: {} }, cur = null, idle = null, fail = null, line = Promise.resolve(), pend = 0, sd = null;
// peak GB seen per model (catalog measured.gb), weights on disk
const used = {}; let size = {};
// the last clip of each model, linked on its row
const last = {};
// clips being made or failed, by their 32 hex: {t, err ('' while making)}; a made one leaves, its file answers
const MAKING = new Map();
const making = (hex) => { const m = MAKING.get(hex); return m ? m.err : undefined; };
// set by the gateway: saves a new peak
let grew = () => {};
// set by the gateway: true while a local chat turn is generating on the same GPU
let chatBusy = () => false;
// set by the gateway too: may this model take gb (a quiet engine lets go if not), and who to tell once no picture
// model is held or owed, so the pinned brain comes back
let room = (gb, who, live, cb) => cb(null), freed = () => {};
// a picture model is loaded or loading, or its last sd-server has not exited yet (it still holds its memory)
const held = () => !!(cur || sd), vacant = () => { if (!held() && !pend) freed(); };

function options(raw) {
  C = Object.assign({ bin: '', idle_min: 10 }, raw);
  C.models = C.models && typeof C.models === 'object' ? C.models : {};
  size = {};
  Object.keys(C.models).forEach((id) => { used[id] = Math.max(used[id] || 0, +((C.models[id] || {}).measured || {}).gb || 0); });
  if (cur && !has(cur.id)) unload();
}
// own keys only: "__proto__" or "constructor" are not models
const has = (id) => typeof id === 'string' && Object.prototype.hasOwnProperty.call(C.models, id);
// create, edit or both
const task = (id) => has(id) ? (C.models[id].task === 'ref2va' ? 'ref2va' : 'fl2va') : '';
const can = (id, t) => task(id) === t;
// weights on disk, GB (.index.json = its shards)
function disk(id) {
  if (size[id] == null) size[id] = +((C.models[id].args || []).reduce((s, f) => { try {
    return s + (/\.index\.json$/.test(f) ? [...new Set(Object.values(JSON.parse(fs.readFileSync(f, 'utf8')).weight_map))].reduce((t, x) => t + fs.statSync(path.join(path.dirname(f), x)).size, 0)
      : fs.statSync(f).isFile() ? fs.statSync(f).size : 0);
  } catch (_) { return s; } }, 0) / 1e9).toFixed(1);
  return size[id];
}
// GB needed: measured peak, else weights on disk
const need = (id) => used[id] || disk(id);
const peak = () => Math.max(0, ...Object.keys(C.models).map(need));
function measure(me) {
  const g = me.proc ? gpuGb(me.proc.pid) : 0, m = C.models[me.id];
  if (m && g > (used[me.id] || 0) + 0.1) { used[me.id] = g; m.measured = Object.assign({}, m.measured, { gb: g }); grew(); }
}

// GPU memory a process holds, from DRM fdinfo (amdgpu: vram + gtt per client), in GB; 0 when unknown
function gpuGb(pid) {
  const seen = {}; let kb = 0;
  try {
    fs.readdirSync('/proc/' + pid + '/fdinfo').forEach((f) => {
      let t = ''; try { t = fs.readFileSync('/proc/' + pid + '/fdinfo/' + f, 'utf8'); } catch (_) { return; }
      const id = /drm-client-id:\s*(\d+)/.exec(t);
      if (!id || seen[id[1]]) return; seen[id[1]] = 1;
      t.replace(/drm-memory-(?:vram|gtt):\s*(\d+)/g, (m, n) => { kb += +n; });
    });
  } catch (_) {}
  return +(kb * 1024 / 1e9).toFixed(1);
}
// jobs = pending pictures (running + waiting): while any, the owner's load/unload would kill one
const state = () => cur && { id: cur.id, state: cur.ready ? 'loaded' : 'loading', busy: cur.busy, jobs: pend, pid: cur.proc ? cur.proc.pid : 0, since: cur.since, last_use: cur.last };
// panel rows: every configured model with its live state and the last load or crash error
const list = () => Object.keys(C.models).map((id) => {
  const s = cur && cur.id === id ? state() : null;
  return Object.assign({ id: id, task: task(id), gb: need(id), idle_min: +C.idle_min || 0, state: '', busy: false, jobs: pend, pid: 0, since: 0, last_use: 0 }, s,
    { mem_gb: s && cur.ready && s.pid ? gpuGb(s.pid) : 0, err: fail && fail.id === id ? fail.msg : '', last: last[id] || '' });
});

// A response cut mid-body emits neither 'end' nor a request error: 'aborted'/'close' end it, cb runs exactly once.
function call(port, method, p, body, cb) {
  let once = 0; const done = (e, s, j) => { if (!once++) cb(e, s, j); };
  const data = body && Buffer.from(JSON.stringify(body));
  const r = http.request({ host: '127.0.0.1', port: port, path: p, method: method, timeout: 60000,
    headers: data ? { 'content-type': 'application/json', 'content-length': data.length } : {} }, (res) => {
    const b = []; res.on('data', (d) => b.push(d));
    res.on('end', () => { let j = null; try { j = JSON.parse(Buffer.concat(b)); } catch (_) {} done(null, res.statusCode, j); });
    res.on('aborted', () => done(new Error('sd-server closed the connection')));
    res.on('close', () => { if (!res.complete) done(new Error('sd-server closed the connection')); });
  });
  r.on('error', done); r.on('timeout', () => r.destroy(new Error('timeout')));
  if (data) r.write(data); r.end();
}
function freePort(cb) { const s = net.createServer(); s.on('error', () => cb(0)); s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => cb(p)); }); }
// the reason a start failed, from the end of sd-server.log
// the reason a start failed, from the end of sd-server.log; the out-of-memory line wins when there is one,
// because new_sd_ctx_t also fails for a bad path or a corrupt file and that is not a memory problem
const tail = () => { try { const l = fs.readFileSync(LOG, 'utf8').slice(-6000).split('\n').filter(Boolean); return (l.filter((x) => /OutOfDeviceMemory|out of memory/i.test(x)).pop() || l.filter((x) => /error|fail/i.test(x)).pop() || l.pop() || '').slice(0, 300); } catch (_) { return ''; } };
function touch() { clearTimeout(idle); if (cur && +C.idle_min > 0) idle = setTimeout(unload, C.idle_min * 60000); }

// Starts the model's sd-server (another model is killed first); cb(err) once it answers. gone: its caller left.
function load(id, cb, gone) {
  cb = cb || (() => {});
  if (!has(id)) return cb(new Error('Unknown local video model: ' + id));
  if (cur && cur.id === id) return cur.ready ? cb(null) : cur.wait.push(cb);
  unload(1);
  fail = null;
  const m = C.models[id], me = cur = { id: id, since: Date.now(), last: 0, ready: false, busy: false, wait: [cb], port: 0, proc: null };
  const done = (e) => {
    const w = me.wait; me.wait = [];
    if (e) { fail = { id: id, msg: e.message }; if (cur === me) unload(); } else if (cur === me) { me.ready = true; measure(me); touch(); }
    w.forEach((f) => f(e || (cur === me ? null : new Error('Video model unloaded.'))));
  };
  const boot = (port) => {
    if (cur !== me) return;
    if (!port) return done(new Error('no free port'));
    // the previous run's log survives one restart as .1: a crash reason is not wiped by the next load
    try { fs.renameSync(LOG, LOG + '.1'); } catch (_) {}
    let out = 'ignore'; try { out = fs.openSync(LOG, 'w'); } catch (_) {}
    let p;
    try { p = me.proc = spawn(C.bin, (m.args || []).concat(loraDir(m.gen), ['--eager-load', '--listen-ip', '127.0.0.1', '--listen-port', String(me.port = port)]), { stdio: ['ignore', out, out] }); }
    catch (e) { return done(e); } finally { if (typeof out === 'number') fs.closeSync(out); }
    p.on('error', done);
    if (p.pid) sd = p;
    // an exit after unload() is the kill itself; otherwise sd-server died (bad args, out of memory, device lost)
    p.on('exit', (code, sig) => {
      if (cur !== me) return; cur = null; clearTimeout(idle);
      const e = new Error('sd-server exited (' + (sig || code) + ') ' + tail());
      if (!me.ready) return done(e);
      me.dead = e.message; fail = { id: id, msg: e.message };
    });
    p.on('exit', () => { if (sd === p) sd = null; vacant(); });
    const t0 = Date.now(), poll = () => {
      if (cur !== me || me.ready) return;
      if (Date.now() - t0 > T.load) return done(new Error('sd-server did not answer in ' + T.load / 60000 + ' min'));
      call(port, 'GET', '/sdcpp/v1/capabilities', null, (e, s) => { if (cur === me && !me.ready) { if (!e && s === 200) done(null); else setTimeout(poll, 1000); } });
    };
    setTimeout(poll, 1000);
  };
  // The gateway decides if there is memory for it: a model that does not fit is never started at all. Wanted while
  // current; a picture whose caller left drops it, so nobody is evicted for it.
  room(need(id), id, () => cur === me && !(gone && gone() && unload()), (e) => e ? done(e) : freePort(boot));
}
// Kills the running sd-server; the kernel frees its memory when the process exits. next: another model takes its place.
function unload(next) {
  const c = cur; if (!c) return false;
  cur = null; clearTimeout(idle);
  if (c.proc) { try { c.proc.kill('SIGTERM'); } catch (_) {} setTimeout(() => { try { c.proc.kill('SIGKILL'); } catch (_) {} }, 5000).unref(); }
  const w = c.wait; c.wait = []; w.forEach((f) => f(new Error('Video model unloaded.')));
  if (!next) vacant();
  return true;
}
process.on('exit', () => { if (cur && cur.proc) try { cur.proc.kill('SIGKILL'); } catch (_) {} });

// One job at a time (the GPU runs at 99% during a picture): each waits for the one before, at most T.queue pending.
// job = {prompt, image?: data URL to edit, gone?: () => true once the caller left}; cb(err, png Buffer)
function run(id, job, cb) {
  if (pend >= T.queue) return cb(new Error('Video queue is full, try again later.'));
  pend++;
  const left = () => !!(job.gone && job.gone());
  line = line.then(() => new Promise((next) => {
    let over = 0, wd = 0; const end = (e, png) => { if (!over++) { clearTimeout(wd); pend--; vacant(); next(); cb(e, png); } };
    // a caller that left while waiting costs no load and no GPU
    if (left()) return end(new Error('client left'));
    // a lost callback (hung socket) must never block the line: the job ends by force
    wd = setTimeout(() => { unload(); end(new Error('timeout')); }, T.load + T.job + 30000);
    try {
      load(id, (e) => {
        if (left()) return end(new Error('client left'));
        if (e) return end(e);
        const me = cur, t0 = Date.now();
        me.busy = true; clearTimeout(idle);
        const fin = (e2, png) => { if (cur === me) { me.busy = false; me.last = Date.now(); touch(); } end(e2, png); };
        // the card's gen goes as sd-server's own fields, as the image engine sends it
        const g = Object.assign({ width: 1344, height: 768, fps: 24, video_frames: 121, sample_params: { sample_steps: 20, guidance: { txt_cfg: 1 } } }, genOf(C.models[id].gen), job.gen);
        // seconds -> frames: 4..15 s at its fps, 4n+1 as the video VAE packs them
        const fr = +job.seconds ? Math.round(Math.min(15, Math.max(4, +job.seconds)) * g.fps / 4) * 4 + 1 : g.video_frames;
        const body = Object.assign({ seed: -1 }, g, { prompt: job.prompt, video_frames: fr, output_format: 'webm' });
        if (job.first) body.init_image = job.first;
        if (job.last) body.end_image = job.last;
        if (job.refs && job.refs.length) body.ref_images = job.refs;
        // A chat generating beside a picture runs 2-2.5x slower (one GPU): the picture starts when the chat ends, 60 s at most.
        // sd-server cannot stop a running job, so a chat that arrives mid-picture still shares the GPU.
        const go = () => { if (chatBusy() && Date.now() - t0 < 60000) return setTimeout(() => (left() ? fin(new Error('client left')) : go()), 1000); job.waited = Date.now() - t0; send(); };
        const send = () => call(me.port, 'POST', '/sdcpp/v1/vid_gen', body, (e1, s, j) => {
          if (e1 || s !== 202 || !j || !j.id) return fin(new Error(e1 ? e1.message : String((j && j.error) || 'HTTP ' + s)));
          let asked = 0;
          const poll = () => {
            if (cur !== me) return fin(new Error(me.dead || 'Image model unloaded.'));
            measure(me);
            if (Date.now() - t0 > T.job) { unload(); return fin(new Error('timeout')); }
            // The caller left: cancel is asked once, but sd-server 866 cannot stop a generating job (capabilities:
            // cancel_generating false), so the line waits for its end and drops the picture: one job on the GPU, always.
            if (left() && !asked++) call(me.port, 'POST', '/sdcpp/v1/jobs/' + j.id + '/cancel', null, () => {});
            call(me.port, 'GET', '/sdcpp/v1/jobs/' + j.id, null, (e2, s2, k) => {
              const st = k && k.status, img = st === 'completed' && (k.result || {}).b64_json;
              if (img) return asked ? fin(new Error('client left')) : fin(null, Buffer.from(img, 'base64'));
              // a dropped connection is often sd-server dying: its exit reason, once known, is the better error
              if (e2) return setTimeout(() => fin(new Error(me.dead || e2.message)), 1000);
              if (s2 !== 200 || /completed|failed|cancelled/.test(st)) return fin(new Error(asked ? 'client left' : (k && k.error && k.error.message) || (st ? 'job ' + st + ' without video' : 'HTTP ' + s2)));
              setTimeout(poll, 1000);
            });
          };
          setTimeout(poll, 1000);
        });
        go();
      }, left);
    } catch (e) { end(e); }
  }));
}

// Clips older than 24 h are deleted: at start and on every write (the /vid route refuses them meanwhile).
function sweep() {
  const old = Date.now() - DAY;
  try { fs.readdirSync(DIR).forEach((f) => { try { if (fs.statSync(path.join(DIR, f)).mtimeMs < old) fs.unlinkSync(path.join(DIR, f)); } catch (_) {} }); } catch (_) {}
  MAKING.forEach((m, hex) => { if (m.err && m.t < old) MAKING.delete(hex); });
}
sweep();
// sd-server's webm -> vid/<hex>.mp4 (H.264 CRF 23 + AAC 128k, faststart: it plays while it downloads); cb(err)
function save(buf, hex, cb) {
  try { fs.mkdirSync(DIR, { recursive: true }); sweep(); fs.writeFileSync(path.join(DIR, hex + '.src.webm'), buf); } catch (e) { return cb(e); }
  const src = path.join(DIR, hex + '.src.webm');
  execFile('ffmpeg', ['-v', 'error', '-y', '-i', src, '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
    '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', path.join(DIR, hex + '.mp4')], { timeout: 300000 }, (e, o, er) => {
    fs.unlink(src, () => {});
    cb(e && new Error('ffmpeg: ' + (String(er || '').trim().split('\n').pop() || e.message)));
  });
}
// The tool runner, as the drawer's: {md,url,tin,tout,vids,usd}, err on failure. A clip is a background job (Google's background
// mode, OpenRouter's jobs): its link is returned at once, and the clip is made and kept even if the caller leaves; the link
// answers 202 while it is made, 410 if it failed, the clip once it is there.
// a = {prompt, seconds, first (the picture to animate: URL or data URL), last, refs (data URLs), base: absolute URL of this server}
function make(id, a, cb) {
  const oom = (s) => /OutOfDeviceMemory|out of memory/i.test(s) ? 'not enough memory for this video model right now' : s;
  const hex = crypto.randomBytes(16).toString('hex'), url = a.base + '/vid/' + hex + '.mp4';
  MAKING.set(hex, { t: Date.now(), err: '' });
  const bad = (e, say) => { const m = say || oom(e.message); MAKING.set(hex, { t: Date.now(), err: m }); cb({ md: '_(Video generation failed: ' + m + ')_', url: url, tin: 0, tout: 0, vids: 0, usd: 0, err: e.message }); };
  const job = { prompt: String(a.prompt || ''), seconds: a.seconds, last: a.last, refs: a.refs, gen: a.gen };
  const go = (first) => { job.first = first; run(id, job, (e, buf) => {
    if (e) return bad(e);
    save(buf, hex, (e2) => {
      if (e2) return bad(e2);
      MAKING.delete(hex); last[id] = url;
      cb({ md: '[🎬 video](' + url + ')', url: url, tin: 0, tout: 0, vids: 1, usd: 0, wait: job.waited });
    });
  }); };
  // the first frame is read as the image engine reads a picture to edit: ours from disk, a public one fetched
  if (!a.first) go('');
  else fetchImage(a.first, (e, img) => e ? bad(e, 'could not read the picture to animate') : go(img));
  return url;
}

module.exports = { options, has, task, can, need, peak, list, state, load, unload, make, making, run, DIR, DAY, T, jobs: () => pend, held, busyWith: (f) => { chatBusy = f; }, roomWith: (f) => { room = f; }, freedWith: (f) => { freed = f; }, grewWith: (f) => { grew = f; } };
