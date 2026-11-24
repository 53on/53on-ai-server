'use strict';
/*
 * chat_local.js: local chat models through llama.cpp's llama-server, native OpenAI API on 127.0.0.1.
 * One process per loaded model with fixed options (context, batch, drafts): nothing reloads per request.
 * A pinned model that dies is started again; unload, a context change or ia_server's exit kills the process.
 * Config: {bin, models: {id: {args: [...]}}}; args carry -m, --mmproj and tuning, the gateway adds context, host and port.
 * 'name:latest' (old clients and keys) is the same model as a configured 'name'.
 * chat.slot_dir: the prompts survive the process. Every planned kill saves the slots there first (llama-server's own
 * /slots save) and the next start restores them; files are named <key>.s<slot>, key = what gives their bytes a meaning
 * (keyOf: engine, GGUFs, args): a new build, model or unknown flag starts cold, a tuning flag does not. An idle model
 * also saves the slots that grew: insurance against a crash. A file lives an hour from its last save, as the industry's
 * caches expire by themselves (Anthropic's longest lives an hour), and another conversation in its slot takes its place.
 */
const { spawn } = require('child_process');
const http = require('http'), fs = require('fs'), path = require('path'), crypto = require('crypto');
const { gpuGb, freePort, peak } = require('./image_local');
const video = require('./video'), mem = require('./mem');

// start limit, idle unload of a model nobody pinned and uptime that forgives past crashes, in ms; restart pauses in s;
// slot files: save limit, idle time before an insurance save and the growth that earns one, TTL and disk cap (4 full
// slots with their checkpoints ~53 GiB); watchdog: poll period and the time the engine may sit busy without any progress
const T = { load: 600000, idle: 1800000, stable: 600000, back: [5, 30, 120, 600], save: 60000, quiet: 30000, grow: 8192, ttl: 3600000, cap: 64 * 2 ** 30, wd: 60000, stuck: 600000 };
let C = { bin: '', models: {} }, cap = 8192, ver = '';
const run = {};     // id -> {proc, port, c, g (the memory gate let it start), ready, readyAt, pin, busy, idle, wait[], key, saved{slot: tokens}, live: Set(targets), slots[], wd{}}
const known = {};   // id -> what the model said about itself, kept across restarts and unloads
const fail = {};    // id -> last start or crash error
const crash = {};   // id -> deaths since the model last stayed up T.stable
const retry = {};   // id -> pending restart timer
const dying = {};   // id -> a killed process that has not exited yet (its memory is still held)
const used = {};    // id -> GB this model was measured holding once it served (image_local's gpuGb)
const gone = {};    // id -> why its process stopped: {why: idle|owner|yield:<image model>|crash, at, next (retry at), k, of}
const probe = {};   // id -> first KV point {g: GB, p: pool, k: key}

const bare = (n) => String(n || '').replace(/:latest$/, '');
const has = (n) => Object.prototype.hasOwnProperty.call(C.models, bare(n));
const id = (n) => has(n) ? bare(n) : String(n || '');
const argOf = (a, ...k) => { const i = a.findIndex((x) => k.indexOf(x) > -1); return i > -1 ? a[i + 1] || '' : ''; };
// a split model (-00001-of-0000N.gguf) is all N of its shards
const shards = (f) => { const m = /^(.*-)\d+-of-(\d+)\.gguf$/.exec(f); return m ? Array.from({ length: +m[2] }, (_, i) => m[1] + String(i + 1).padStart(m[2].length, '0') + '-of-' + m[2] + '.gguf') : [f]; };
const LOG = (n) => path.join(__dirname, 'llama-' + n.replace(/[^\w.-]/g, '_') + '.log');
const tail = (n) => { try { const l = fs.readFileSync(LOG(n), 'utf8').slice(-6000).split('\n').filter(Boolean); return (l.filter((x) => /error|fail|abort|lost/i.test(x)).pop() || l.pop() || '').slice(0, 300); } catch (_) { return ''; } };
// served context: the owner's cap, never over what the model was trained on
// window: the card's ctx (its own calibration), else the default; never past what it trained
const plan = (n) => { const t = (known[id(n)] || {}).train || Infinity; return Math.min(t, +C.models[id(n)].ctx || cap); };
// what the model takes now: what its engine reported, within its entry's caps (if set), video only with ffmpeg (video.js)
const takes = (n, k) => { const d = (C.models[id(n)] || {}).caps;
  return ((known[id(n)] || {}).caps || []).indexOf(k) > -1 && (!Array.isArray(d) || k === 'completion' || d.indexOf(k) > -1) && (k !== 'video' || video.ok()); };
// a memory knob (kv_pool, cache_ram) of the model, else the global one
const knob = (n, k) => +((C.models[id(n)] || {})[k] ?? C[k]) || 0;
// what this model was MEASURED needing; 0 = never measured, so there is no gate
const need = (n) => used[id(n)] || 0;
// set by the gateway, which owns both engines: may I take gb? Default = nobody asks
let room = (gb, who, live, cb) => cb(null);
// set by the gateway: saves a new measure
let learnt = () => {};
// KV measure key: engine, files, args (no tuning flags)
// a model's own engine: its card's bin, inside the engines folder; else the global one
const binOf = (n) => { const b = C.models[n].bin; if (!b) return C.bin;
  try { const r = fs.realpathSync(b); return r.startsWith(path.dirname(path.dirname(fs.realpathSync(C.bin))) + path.sep) ? r : C.bin; } catch (_) { return C.bin; } };
const kvKey = (n) => keyOf(binOf(n), C.models[n].args, Object.assign({}, process.env, C.env || {}));
// KV pool (tokens): catalog kv_pool, else (free - base - picture peak - cache_ram - reserve) / bytes per token, 1 window..1 per slot
function pool(n, f) {
  const c = plan(n), k = knob(n, 'kv_pool'), kv = (C.models[n].measured || {}).kv;
  if (k) return k;
  if (!kv || kv.key !== kvKey(n)) return (probe[n] || {}).k === kvKey(n) ? c >> 1 : c;
  const gb = f - kv.base - peak() - knob(n, 'cache_ram') / 1024 - (knob(n, 'reserve') || 2);
  return Math.max(c, Math.min(slotsOf(n) * c, Math.floor(gb * 1e9 / kv.b / 4096) * 4096 || 0));
}
// true = restart. Unmeasured: GPU at two pools gives bytes/token; base = what the start took from mem.free() - pool
function weigh(n, me) {
  const m = C.models[n], kv = (m.measured || {}).kv;
  if (me.k || kv && kv.key === me.kk) return false;
  const g = gpuGb(me.proc.pid), p = probe[n];
  if (!p || p.k !== me.kk) { if (!g) return false; probe[n] = { g: g, p: me.p, k: me.kk }; return true; }
  delete probe[n];
  const b = (p.g - g) * 1e9 / (p.p - me.p);
  if (!(b > 0)) return false;
  m.measured = Object.assign({}, m.measured, { kv: { b: Math.round(b), base: +(me.f - mem.free() - me.p * b / 1e9).toFixed(2), key: me.kk } });
  learnt(n);
  return pool(n, me.f) !== me.p;
}

function options(raw, ctxCap) {
  const was = C.bin;
  C = Object.assign({ bin: '' }, raw);
  // only entries with an array of string args are models: a null or malformed entry must not take the server down
  const ms = C.models && typeof C.models === 'object' ? C.models : {};
  C.models = {};
  Object.keys(ms).forEach((n) => { const a = ms[n] && ms[n].args; if (Array.isArray(a) && a.every((x) => typeof x === 'string')) C.models[n] = ms[n]; });
  cap = Math.max(1024, +ctxCap || 8192);
  // the version is the build's folder (~/engines/<engine.dir><version>), whatever the project calls it: bNNNNN, v0.4.1…
  const d = (C.engine || {}).dir;
  try { ver = d ? (fs.realpathSync(C.bin).split(path.sep).slice(0, -1).find((s) => !s.indexOf(d)) || '').slice(d.length) : ''; } catch (_) { ver = ''; }
  // a new build starts with a clean record: deaths on the other one say nothing about it
  if (was !== C.bin) [crash, fail].forEach((o) => Object.keys(o).forEach((n) => delete o[n]));
  // a model gone from the config is killed; one started (or starting) with another context or engine restarts with the new one
  Object.keys(run).forEach((n) => { if (!has(n)) unload(n); else if (run[n].c && (run[n].c !== plan(n) || run[n].k !== knob(n, 'kv_pool') || run[n].r !== knob(n, 'cache_ram') || was !== C.bin || !!run[n].strata !== (C.models[n].engine === 'strata'))) restart(n); });
}
// requests waiting for the old start wait for the new one
function restart(n) { const r = run[n], w = r.wait; r.wait = []; unload(n); load(n, r.pin); run[n].wait.push(...w); }

// GET, or POST of a JSON body. A cut body emits neither 'end' nor an error: 'aborted'/'close' end it, cb runs exactly once.
function call(port, p, body, cb, ms) {
  let once = 0; const done = (e, s, j) => { if (!once++) cb(e, s, j); };
  const r = http.request({ host: '127.0.0.1', port: port, path: p, method: body ? 'POST' : 'GET', headers: body ? { 'content-type': 'application/json' } : {}, timeout: ms || 5000 }, (res) => {
    const b = []; res.on('data', (d) => b.push(d));
    res.on('end', () => { let j = null; try { j = JSON.parse(Buffer.concat(b)); } catch (_) {} done(null, res.statusCode, j); });
    res.on('aborted', () => done(new Error('closed')));
    res.on('close', () => { if (!res.complete) done(new Error('closed')); });
  });
  r.on('error', done); r.on('timeout', () => r.destroy(new Error('timeout')));
  r.end(body ? JSON.stringify(body) : undefined);
}
const get = (port, p, cb) => call(port, p, null, cb);

// Slot files in C.slot_dir: whole ones are <key>.s<slot>, a save writes <key>.s<slot>.<seq>.tmp first; the ckpt engine
// adds '<file>.dft' (draft state), which follows its file. Oldest first, with the size of both.
const SLOT = /^(\w+)\.s(\d+)$/;
function slotFiles(dir) {
  let names = []; try { names = fs.readdirSync(dir); } catch (_) {}
  return names.map((f) => { const m = SLOT.exec(f), p = path.join(dir, f); if (!m) return null;
    let s, d = 0; try { s = fs.statSync(p); d = fs.statSync(p + '.dft').size; } catch (_) { if (!s) return null; }
    return { f: f, p: p, key: m[1], i: +m[2], t: s.mtimeMs, b: s.size + d }; }).filter(Boolean).sort((a, b) => a.t - b.t);
}
const drop = (p) => [p, p + '.dft'].forEach((x) => { try { fs.unlinkSync(x); } catch (_) {} });
const say = (n, w, i, e, s, j, k) => console.log('[chat] ' + n + ' ' + w + ' s' + i + ': ' + (e ? e.message : s === 200 && j ? j[k] + ' tokens ' + Math.round((j.timings || {})[w + '_ms'] || 0) + ' ms' : 'refused'));
// Gone: files past the TTL and, oldest first, the excess over the disk cap; with key, its half-written temporaries too.
function prune(dir, key) {
  if (key) { let names = []; try { names = fs.readdirSync(dir); } catch (_) {} names.forEach((f) => { if (!f.indexOf(key + '.s') && /\.tmp(\.dft)?$/.test(f)) { try { fs.unlinkSync(path.join(dir, f)); } catch (_) {} } }); }
  const all = slotFiles(dir), now = Date.now();
  let total = all.reduce((s, x) => s + x.b, 0);
  all.forEach((x) => { if (now - x.t > T.ttl || total > T.cap) { drop(x.p); total -= x.b; } });
}
// The key holds what gives the files' bytes a meaning, so a speed, memory or scheduling flag keeps every prompt warm.
// llama.cpp checks shapes and types on restore, never whose weights wrote the rows, and replays the checkpoint appendix
// blind (a layout it cannot take aborts it). So: the engine's code (its libraries: the binary is the same launcher in
// every build, and a rebuild can land in place), every GGUF by size and ctime (cp -p and rsync -a keep the old mtime)
// and every arg, from argv or LLAMA_ARG_* (llama.cpp reads both), but OPS and its values: a flag nobody vetted starts
// cold once, never warm on foreign bytes.
// OPS: the tuning flags in use, as spelled here, vetted against llama.cpp: they move speed, memory, scheduling or
// prompt formatting, and capacity and streams (-c, -np, -kvu) a restore refuses, it never misreads them (but -c picks
// a LongRoPE model's RoPE factors: Phi-3, not served). Not --video-fps: a frame's id is the clip's hash + its index,
// and the fps sets the index's time.
const OPS = /^(-[cbt]|-ub|-tb|-np|-ngl|-kvu|--(ctx-checkpoints|cache-ram|kv-unified-per-slot|lazy-mode|reasoning-budget|prefill-budget-busy|spec-max-busy-slots|spec-draft-(n-max|p-min)|chat-template-file|slot-save-path))$/;
const stamp = (f) => shards(f).map((x) => { try { const s = fs.statSync(x); return x + ' ' + s.size + ' ' + s.ctimeMs; } catch (_) { return x; } });
function keyOf(bin, args, env = {}) {
  let real = bin, libs = []; try { real = fs.realpathSync(bin); libs = fs.readdirSync(path.dirname(real)).filter((f) => /\.so\b/.test(f)).sort().map((f) => path.join(path.dirname(real), f)); } catch (_) {}
  let on = true; const k = [real].concat(libs).map(stamp);
  Object.keys(env).filter((v) => !v.indexOf('LLAMA_ARG_')).sort().flatMap((v) => [v, String(env[v])]).concat(args).forEach((x) => {
    if (/^--?[a-z]/i.test(x)) on = !OPS.test(x);
    // adapters come as lists, 'a.gguf,b.gguf' or 'a.gguf:0.5,b.gguf:1'
    if (on) k.push(x, ...x.split(/[,:]/).filter((f) => /\.gguf$/i.test(f)).map(stamp));
  });
  return crypto.createHash('sha1').update(k.flat().join('\n')).digest('hex').slice(0, 12);
}

// Restores this key's files, oldest first (the first ones go to the RAM cache when a task needs the KV), then done().
// A file the engine refuses or dies reading is dropped (me.rf, at its exit); a timeout keeps it for the next start.
function restore(n, me, done) {
  const dir = C.slot_dir; if (!dir) return done();
  me.rs = 1; prune(dir, me.key);
  const q = slotFiles(dir).filter((x) => x.key === me.key);
  const next = () => {
    const x = q.shift(); if (!x) return done();
    me.rf = x.p;
    call(me.port, '/slots/' + x.i + '?action=restore', { filename: x.f }, (e, s, j) => {
      if (run[n] !== me) return;
      if (e && e.message !== 'timeout') q.length = 0; else me.rf = 0;
      if (!e && s !== 200) drop(x.p); else if (!e) me.saved[x.i] = +(j || {}).n_restored || 0;
      say(n, 'restore', x.i, e, s, j, 'n_restored');
      next();
    }, T.save);
  };
  next();
}

// Saves to disk every slot with tokens (all) or the ones that grew T.grow since their last save; cb() once done, T.save at
// most. all cuts the running relays first: llama-server keeps the prompt read so far and the save waits for the slot.
// Each save goes to a temporary name, kept only when the engine says it wrote tokens: a slot cleared by a racing task
// never replaces a good file.
function save(n, r, all, cb) {
  const dir = C.slot_dir; if (!dir || !r.ready) return cb();
  let over = 0; const end = () => { if (!over++) { clearTimeout(tm); prune(dir); cb(); } };
  const tm = setTimeout(end, T.save);
  if (all) r.live.forEach((t) => { if (t.abort) t.abort(); });
  call(r.port, '/slots', null, (e, s, j) => {
    // a slot holding fewer tokens than its file is another conversation: it earns a file of its own the same way
    const q = Array.isArray(j) ? j.filter((x) => { const v = x.n_prompt_tokens, w = r.saved[x.id] || 0; return v > 0 && (all || v - w >= T.grow || v < w && v >= T.grow); }) : [];
    const next = () => {
      const x = q.shift(); if (!x || over) return end();
      const f = r.key + '.s' + x.id, p = path.join(dir, f), tmp = f + '.' + (r.seq = (r.seq || 0) + 1) + '.tmp';
      call(r.port, '/slots/' + x.id + '?action=save', { filename: tmp }, (e2, s2, k) => {
        const ok = !e2 && s2 === 200 && k && +k.n_saved > 0;
        if (ok) { try { fs.renameSync(path.join(dir, tmp), p); r.saved[x.id] = +k.n_saved; } catch (_) {} try { fs.renameSync(path.join(dir, tmp + '.dft'), p + '.dft'); } catch (_) { drop(p + '.dft'); } }
        else drop(path.join(dir, tmp));
        say(n, 'save', x.id, e2, ok ? 200 : s2, k, 'n_saved');
        next();
      }, T.save);
    };
    next();
  }, T.save);
}

// What the running model says about itself: trained context, size, template levels and caps. cb(false) = ask again.
function learn(n, port, cb) {
  get(port, '/v1/models', (e, s, j) => get(port, '/props', (e2, s2, p) => {
    const m = (((j || {}).data || [])[0] || {}).meta, tc = p && p.chat_template_caps;
    if (!m || !tc) return cb(false);
    const tpl = String(p.chat_template || ''), lv = /reasoning_effort\s+not\s+in\s*\(([^)]*)\)/.exec(tpl);
    known[n] = { train: +m.n_ctx_train || 0, ctx: +(p.default_generation_settings || {}).n_ctx || 0, params: +m.n_params || 0, ftype: m.ftype || '', slots: +p.total_slots || 0,
      levels: lv ? lv[1].match(/[a-z]+/g) : null,
      caps: ['completion'].concat(['vision', 'video', 'audio'].filter((k) => (p.modalities || {})[k]), tc.supports_tools ? ['tools'] : [], tc.supports_reasoning_effort ? ['thinking'] : []) };
    cb(true);
  }));
}

// Strata's live block (GET /metrics): what it reads and writes, slot by slot; null if it does not answer
const strataLive = (r, cb) => get(r.port, '/metrics', (e, s, j) => cb(!e && s === 200 && j && j.live || null));
// Strata's slots: its run config's "parallel", else 1
const strataSlots = (p) => { try { return Math.max(1, +JSON.parse(fs.readFileSync(p, 'utf8')).parallel || 1); } catch (_) { return 1; } };
// Starts the model's llama-server; cb(err) once it answers. pin = keep it alive (start it again if it dies).
function loadStrata(n, pin, cb) {
  const cur = run[n];
  if (cur) { if (pin) cur.pin = true; return cur.ready ? cb(null) : cur.wait.push(cb); }
  const mc = C.models[n], port = +mc.strata_port || 8091, bin = C.strata_bin || 'strata', prof = mc.strata_profile;
  known[n] = Object.assign(known[n] || {}, { train: plan(n), caps: ['completion', 'tools'], slots: strataSlots(prof) });
  const me = run[n] = { port: port, c: plan(n), k: knob(n,'kv_pool'), r: knob(n,'cache_ram'), ready: false, pin: !!pin, busy: 0, wait: [cb], proc: null, strata: true, live: new Set(), slots: [], wd: {}, at: Date.now() };
  const done = (e) => { if (run[n] !== me) return; const w = me.wait; me.wait = []; if (e) { unload(n, true); down(n, me, e); } else { me.ready = true; me.readyAt = Date.now(); known[n].up = me.readyAt - me.at; delete fail[n]; touch(n, me); } w.forEach((f) => f(e || null)); };
  if (!prof) return done(new Error('Strata model needs strata_profile'));
  const env = Object.assign({}, process.env, C.env || {});
  // its own words (each request's figures, «N asked, the engine runs M») go to strata-<model>.log, the last run kept as .1
  const lg = path.join(__dirname, 'strata-' + n.replace(/[^\w.-]/g, '_') + '.log');
  try { fs.renameSync(lg, lg + '.1'); } catch (_) {}
  let out = 'ignore'; try { out = fs.openSync(lg, 'w'); } catch (_) {}
  try { me.proc = spawn(bin, ['serve', '--profile', prof, '--host', '127.0.0.1', '--port', String(port)], { stdio: ['ignore', out, out], env: env }); }
  catch (e) { return done(e); } finally { if (typeof out === 'number') fs.closeSync(out); }
  me.proc.on('error', (e) => done(e));
  me.proc.on('exit', (code, sig) => { if (run[n] !== me) return; delete run[n]; clearTimeout(me.idle); down(n, me, new Error('Strata exited ' + (sig || code))); });
  const t0 = Date.now(), poll = () => {
    if (run[n] !== me || me.ready) return;
    if (Date.now() - t0 > T.load) return done(new Error('Strata did not answer'));
    get(port, '/v1/models', (e, st) => { if (run[n] !== me) return; if (e || st !== 200) return setTimeout(poll, 1000); done(null); });
  };
  setTimeout(poll, 1000);
}
function load(n, pin, cb) {
  n = id(n); cb = cb || (() => {});
  if (!has(n)) return cb(new Error('Unknown local chat model: ' + n));
  if (C.models[n].engine === 'strata') return loadStrata(n, pin, cb);
  const cur = run[n];
  if (cur) { if (pin) { cur.pin = true; clearTimeout(cur.idle); } return cur.ready ? cb(null) : cur.wait.push(cb); }
  clearTimeout(retry[n]); delete retry[n];
  const me = run[n] = { port: 0, c: 0, ready: false, pin: !!pin, busy: 0, wait: [cb], proc: null, key: '', saved: {}, live: new Set(), slots: [], wd: {}, at: Date.now() };
  const done = (e) => {
    if (run[n] !== me) return;
    const w = me.wait; me.wait = [];
    // up = what this start took once the gate let it go (weights + slots): the wake a client is told to expect
    if (e) { unload(n, true); down(n, me, e); } else { me.ready = true; me.readyAt = Date.now(); known[n].up = me.readyAt - me.at; delete fail[n]; used[n] = gpuGb(me.proc.pid); touch(n, me); }
    w.forEach((f) => f(e || null));
  };
  const start = () => freePort((port) => {
    if (run[n] !== me) return;
    if (!port) return done(new Error('no free port'));
    try { fs.renameSync(LOG(n), LOG(n) + '.1'); } catch (_) {}
    let out = 'ignore'; try { out = fs.openSync(LOG(n), 'w'); } catch (_) {}
    me.c = plan(n); me.k = knob(n, 'kv_pool'); me.kk = kvKey(n); me.p = pool(n, me.f = mem.free()); me.r = knob(n, 'cache_ram');
    // Slots and conversation memory are the ENGINE's call when nobody interferes: without -np it turns on
    // its unified KV (one shared pool, not fixed pieces) and its cache keeps the context of whoever leaves,
    // so coming back is not reprocessed. Measured: a return cost 4 tokens instead of 23,302.
    if (C.slot_dir) { try { fs.mkdirSync(C.slot_dir, { recursive: true }); } catch (_) {} }
    const base = C.models[n].args.concat(
      // kv_pool: one shared KV for several full contexts at once; each slot still stops at the model's window (me.c)
      me.p > me.c ? ['-c', String(me.p), '--kv-unified-per-slot', String(me.c)] : ['-c', String(Math.min(me.p, me.c))],
      C.slots ? ['-np', String(C.slots)] : [],
      me.r ? ['--cache-ram', String(me.r)] : [],
      C.slot_dir ? ['--slot-save-path', C.slot_dir] : [],
      // video.js already picked each clip's fps: the engine keeps it
      argOf(C.models[n].args, '--mmproj', '-mm') && !argOf(C.models[n].args, '--video-fps') ? ['--video-fps', '0'] : []);
    const env = Object.assign({}, process.env, C.env || {});
    me.key = keyOf(binOf(n), base, env);
    const a = base.concat(['--host', '127.0.0.1', '--port', String(me.port = port), '--no-webui', '--metrics', '--alias', n]);
    try { me.proc = spawn(binOf(n), a, { stdio: ['ignore', out, out], env: env }); }
    catch (e) { return done(e); } finally { if (typeof out === 'number') fs.closeSync(out); }
    me.proc.on('error', (e) => done(e));
    me.proc.on('exit', (code, sig) => {
      if (run[n] !== me) return;
      if (me.rf) drop(me.rf);
      const e = new Error('llama-server exited (' + (sig || code) + ') ' + tail(n));
      if (!me.ready) return done(e);
      // a model that stayed up long enough starts its pauses from the shortest again
      if (Date.now() - me.readyAt > T.stable) crash[n] = 0;
      delete run[n]; clearTimeout(me.idle); down(n, me, e);
      const w = me.wait; me.wait = []; w.forEach((f) => f(e));
    });
    const t0 = Date.now(), poll = () => {
      if (run[n] !== me || me.ready) return;
      if (Date.now() - t0 > T.load) return done(new Error('llama-server did not answer in ' + T.load / 60000 + ' min'));
      // /health answers 503 while the weights load and 200 once the model serves
      get(port, '/health', (e, s) => {
        if (run[n] !== me) return;
        if (e || s !== 200) return setTimeout(poll, 1000);
        // a context learned only now (first start) that is under the one used: restart once with it; then the saved prompts
        learn(n, port, (ok) => run[n] !== me ? 0 : !ok ? setTimeout(poll, 1000) : me.c !== plan(n) || weigh(n, me) ? restart(n) : restore(n, me, () => done(null)));
      });
    };
    setTimeout(poll, 1000);
  });
  // killed processes, of this model or another (single mode frees the rest first), still hold their memory while they
  // save: the new one starts once every one of them exits (6 s after its signal at most, which comes after its save)
  const ds = Object.values(dying);
  // Once nothing dying holds its memory, the gateway says if there is room (a quiet picture lets go).
  let k = ds.length; const go = () => { if (!k--) room(need(n), n, () => run[n] === me, (e) => e ? done(e) : (me.g = 1, me.at = Date.now(), start())); };
  ds.forEach((d) => {
    let once = 0; const one = () => { if (!once++) go(); };
    d.once('exit', one);
    const arm = () => setTimeout(d.killed ? one : arm, d.killed ? 6000 : 1000).unref();
    arm();
  });
  go();
}

// A start that failed or a process that died: the error stays on the panel, and a pinned model is started again with
// growing pauses (a GPU reset takes seconds); after the last pause it stays down until a request or the owner loads it.
function down(n, me, e) {
  fail[n] = e.message;
  const k = crash[n] = (crash[n] || 0) + 1, s = me.pin && k <= T.back.length ? T.back[k - 1] : 0;
  if (s) retry[n] = setTimeout(() => { delete retry[n]; if (!run[n] && has(n)) load(n, true); }, s * 1000).unref();
  gone[n] = { why: 'crash', at: Date.now(), next: s && Date.now() + s * 1000, k: k, of: T.back.length };
}

// A model nobody pinned is unloaded after T.idle with no request running.
function touch(n, r) { clearTimeout(r.idle); if (!r.pin && !r.busy) r.idle = setTimeout(() => { if (run[n] === r && !r.busy) unload(n, false, null, 'idle'); }, T.idle).unref(); }

// Kills the model's process (and any pending restart); the kernel frees its memory when it exits. A serving process
// saves its slots first (the single road of every planned kill); requests that arrive meanwhile wait for the next start.
// keepErr leaves the failure on the panel; cb() once the signal is sent; why: the panel says it (idle, owner, yield:<image>).
function unload(n, keepErr, cb, why) {
  n = id(n); clearTimeout(retry[n]); delete retry[n]; cb = cb || (() => {});
  const r = run[n]; if (!r) { cb(); return false; }
  delete run[n]; clearTimeout(r.idle); clearTimeout(r.quiet); if (!keepErr) delete fail[n];
  gone[n] = { why: why || '', at: Date.now() };
  const w = r.wait; r.wait = []; w.forEach((f) => f(new Error('Chat model unloaded.')));
  const p = r.proc;
  if (!p || p.exitCode !== null || p.signalCode) { cb(); return true; }
  dying[n] = p;
  p.once('exit', () => { if (dying[n] === p) delete dying[n]; });
  save(n, r, true, () => {
    try { p.kill('SIGTERM'); } catch (_) {}
    setTimeout(() => { try { p.kill('SIGKILL'); } catch (_) {} }, 5000).unref();
    cb();
  });
  return true;
}
process.on('exit', () => Object.keys(run).forEach((n) => { try { run[n].proc.kill('SIGKILL'); } catch (_) {} }));
// Service stop: each model saves and is killed; cb() once all exited (memory free for the next start)
function saveAll(cb) {
  let k = 1; const one = () => { if (!--k) cb(); };
  Object.keys(run).forEach((n) => { k++; unload(n, false, () => { const p = dying[n]; p && p.exitCode === null && !p.signalCode ? p.once('exit', one) : one(); }); });
  one();
}

// Crash insurance: T.quiet after its last request, an idle model saves the slots that grew T.grow tokens.
function quiet(n, r) { clearTimeout(r.quiet); if (C.slot_dir && !r.busy) r.quiet = setTimeout(() => { if (run[n] === r && !r.busy) save(n, r, false, () => {}); }, T.quiet).unref(); }

// The engine's slots (GET /slots), compact: {id, busy, done, gen}; done = prompt tokens in place (cache + read so far),
// gen = tokens written, 0 while it reads. The prompt's size is not there: n_prompt_tokens is what the slot's KV holds
// (cache + read + written). cb(ok); a failed read keeps the last list.
function poll(n, r, cb) {
  // Strata's /slots lacks what was read and written: its /metrics live block has them, slot by slot (or the engine's, solo)
  if (r.strata) return strataLive(r, (L) => {
    if (L) { const s = (L.slots || []).slice();
      // a request on Strata's solo path sits in no slot (outside_slots): the engine's own live line stands for it
      if (/^(reading|generating)$/.test(L.state) && (!s.length || L.outside_slots > 0)) s.push({ slot: s.length, state: L.state, prompt_tokens: L.prompt_tokens, prompt_read: L.prompt_read, generated: L.generated });
      r.slots = s.map((x) => ({ id: x.slot, busy: x.state !== 'idle', done: +(x.state === 'reading' ? x.prompt_read : x.prompt_tokens) || 0, gen: +x.generated || 0 })); }
    cb(!!L);
  });
  get(r.port, '/slots', (e, s, j) => {
    if (!e && s === 200 && Array.isArray(j)) r.slots = j.map((x) => { const nt = [].concat(x.next_token || [])[0] || {};
      return { id: x.id, busy: !!x.is_processing, done: (+x.n_prompt_tokens_cache || 0) + (+x.n_prompt_tokens_processed || 0), gen: +nt.n_decoded || 0 }; });
    cb(!e && s === 200);
  });
}
// Every ready model's slots for the panel, tagged with the model: read only while a request runs (or the last read saw
// one, so the end of the work is seen); idle = the last list.
function slots(cb) {
  const ns = Object.keys(run).filter((n) => run[n].ready), out = [];
  let k = ns.length + 1; const one = () => { if (!--k) cb(out); };
  ns.forEach((n) => { const r = run[n], add = () => { r.slots.forEach((s) => out.push(Object.assign({ m: n }, s))); one(); };
    if (r.busy || r.slots.some((s) => s.busy)) poll(n, r, add); else add(); });
  one();
}
// Permits the gateway may hand a model: what its engine serves at once (/props total_slots), else the configured -np, else 4.
const slotsOf = (n) => (known[id(n)] || {}).slots || +C.slots || 4;
// Watchdog, one pass: the engine is judged as a whole, never one slot (the batch fills slot by slot in order, so a slot
// queued behind a long read sits busy with frozen counters): its mark = done + gen over every busy slot. A mark that
// does not move for T.stuck while /health answers 200 is stuck (not dead: the exit hook covers that); so is a /slots
// that stops answering (it waits on the main loop, /health does not), on the same clock. Restart = the usual save + kill.
// A healthy 137K prefill moves the mark every ubatch.
function watch() {
  if (C.slot_dir) prune(C.slot_dir);   // a file past its hour goes even on a quiet night
  Object.keys(run).forEach((n) => { const r = run[n]; if (!r.ready || r.strata) return;
    // base keeps its peak (restored slots come after the measure)
    const kv = !r.k && (C.models[n].measured || {}).kv, b = kv && gpuGb(r.proc.pid) - r.p * kv.b / 1e9;
    if (kv && b > kv.base + 0.1) { kv.base = +b.toFixed(2); learnt(n); }
    poll(n, r, (ok) => {
      if (run[n] !== r) return;
      const now = Date.now(), w = r.wd, b = r.slots.filter((s) => s.busy), m = b.reduce((a, s) => a + s.done + s.gen, 0);
      if (ok && !b.length) { r.wd = {}; return; }
      if (ok ? w.m !== m : !w.t) { r.wd = ok ? { m: m, t: now } : { t: now }; return; }
      if (now - w.t < T.stuck) return;
      get(r.port, '/health', (e, st) => { if (e || st !== 200 || run[n] !== r) return;
        console.log('[chat] ' + n + ' stuck: ' + (ok ? b.length + ' busy at ' + m + ' tokens' : '/slots silent') + ' for ' + Math.round((now - w.t) / 60000) + ' min, health 200: restarting');
        restart(n); });
    });
  });
}
setInterval(watch, T.wd).unref();

// Pins follow the gateway: models in list stay up; the others fall back to the idle unload.
function pins(list) {
  Object.keys(run).forEach((n) => { const r = run[n], p = list.indexOf(n) > -1; if (r.pin !== p) { r.pin = p; touch(n, r); } });
}

// What a model that does not serve yet is doing, for whoever waits on it: memory (dying processes, the gate), load
// (weights), restore (its saved slots); '' when it serves or nothing runs
const phase = (n) => { const r = run[id(n)]; return !r || r.ready ? '' : !r.g ? 'memory' : r.rs ? 'restore' : 'load'; };
// Ready models with no request running and none waiting: the only ones a picture may ask to step aside.
const idle = () => Object.keys(run).filter((n) => run[n].ready && !run[n].busy && !run[n].wait.length);

// Relay target (cloud_relay's OpenAI path) for a ready model, else null. done() must be called when the request ends:
// the idle unload and the insurance save only count from there. The relay sets abort(): a save before a kill cuts it.
function target(n) {
  n = id(n); const r = run[n]; if (!r || !r.ready) return null;
  r.busy++; clearTimeout(r.idle); clearTimeout(r.quiet);
  let over = 0;
  const t = { id: n, prov: 'local', model: n, key: 'local', baseUrl: 'http://127.0.0.1:' + r.port + '/v1', meta: { tools: takes(n, 'tools') }, slots: slotsOf(n),
    // Strata reads a prompt without saying so in its stream: its live read, as llama-server's prompt_progress (null when not reading)
    progress: r.strata ? (cb) => strataLive(r, (L) => { const s = L && !L.prompt_total && (L.slots || []).filter((x) => x.state === 'reading');
      const p = L && (L.prompt_total ? { total: L.prompt_total, processed: L.prompt_read } : s && s.length === 1 ? { total: s[0].prompt_tokens, processed: s[0].prompt_read } : null);
      cb(p && +p.total ? { total: +p.total, cache: 0, processed: +p.processed || 0, time_ms: Math.round((+L.elapsed_s || 0) * 1000) } : null); }) : null,
    done: () => { if (!over++) { r.busy--; r.live.delete(t); if (run[n] === r) { touch(n, r); quiet(n, r); } } } };
  r.live.add(t);
  return t;
}

// OpenAI reasoning_effort -> the model's template level, or undefined for its own default. A model whose template lists
// levels has two states, as in VS Code: low (or minimal) is off at its weakest level (it still thinks enough to call its
// tools), anything higher is on at its strongest; so moving the knob among the high levels never re-reads the prompt.
// The thinking cap is the model's (budget below): the client's reasoning_budget_tokens wins when it sends one.
const RANK = ['minimal', 'low', 'medium', 'high', 'xhigh', 'max'], STATES = ['low', 'high'];
function stateOf(L, e) {
  if (e == null || e === true) return undefined;
  if (e === false || e === 'none' || e === 'off') return 'none';
  const i = RANK.indexOf(e);
  if (i < 0) return undefined;
  if (!L) return e;
  const ok = RANK.filter((x) => L.indexOf(x) > -1);
  return i <= 1 ? ok[0] : ok[ok.length - 1];   // minimal, low: off
}
function effort(n, e) { const k = known[id(n)]; return stateOf(k && k.levels, e); }

const size = {};
const gbOf = (f) => { if (!f) return 0; if (size[f] == null) size[f] = shards(f).reduce((s, x) => { try { return s + fs.statSync(x).size; } catch (_) { return s; } }, 0); return size[f]; };
// every layer on the GPU only when the args ask for all of them; 0 = not known
const gpuPct = (a) => /^(all|9{2,})$/.test(argOf(a, '-ngl', '--n-gpu-layers', '--gpu-layers')) ? 100 : 0;
// Panel rows, one per configured model
function list() {
  return Object.keys(C.models).map((n) => {
    const a = C.models[n].args, r = run[n], inf = known[n] || {}, m = argOf(a, '-m', '--model') || '';
    const mtp = /mtp/.test(argOf(a, '--spec-type')) ? +argOf(a, '--spec-draft-n-max') || 3 : 0;
    const gb = +((gbOf(m) + gbOf(argOf(a, '--mmproj', '-mm'))) / 1e9).toFixed(1), kv = (C.models[n].measured || {}).kv;
    // need = what a start takes: its catalog measure at one full window, else its files
    return { name: n, engine: C.models[n].engine || 'llama.cpp', gb: gb, need: +(kv ? kv.base + plan(n) * kv.b / 1e9 : gb).toFixed(1),
      params: inf.params ? (inf.params / 1e9).toFixed(1) + 'B' : '', quant: (/(?:UD-)?I?Q\d\w*|BF16|F16|F32/i.exec(path.basename(m)) || [inf.ftype || ''])[0],
      fam: '', caps: (inf.caps || []).filter((c) => takes(n, c)), ctx: r && r.ready ? inf.ctx : 0, ctxSrc: r && r.ready ? 'served' : 'planned', ctxMax: inf.train || 0, ctxPlan: plan(n),
      states: inf.levels ? STATES : null, mtp: mtp, mtpHead: mtp && inf.train ? 1 : 0, gguf: path.basename(m || ''), state: r ? (r.ready ? 'loaded' : 'loading') : '', err: fail[n] || '',
      // the panel says what a start is doing and since when, why a model is down, and how long its last start took
      ph: phase(n), since: r ? r.at : 0, off: r ? undefined : gone[n], up: inf.up || 0, pool: r ? r.p : 0 };
  });
}
// Loaded rows for the panel and metrics
const loaded = () => Object.keys(run).filter((n) => run[n].ready).map((n) => ({ name: n, vram_gb: run[n].proc ? gpuGb(run[n].proc.pid) : 0, gpu_pct: gpuPct(C.models[n].args), ctx: known[n].ctx, pinned: run[n].pin }));

// the layer staff wrote for a model (cfg.chat.models[n].sys): it goes before the owner's System Prompt
const layer = (n) => String((C.models[id(n)] || {}).sys || '');
// The brake is the server's: the model's catalog entry says how much it may think per template level
// (think_budget {level: tokens}, 0 = free) and how long it may answer (out_max); a model without them gets DEF. The client's own value wins.
const DEF = { minimal: 128, low: 512, medium: 1024, high: 2048, xhigh: 4096, max: 8192 };
const budgets = (n) => (C.models[id(n)] || {}).think_budget || DEF;
const budget = (n, th) => typeof th === 'string' && th !== 'none' ? +budgets(n)[th] || 0 : 0;
const outMax = (n) => +(C.models[id(n)] || {}).out_max || 0;
const ctxUse = (n) => +(C.models[id(n)] || {}).ctx_use || 0;   // the useful window: the deepest context it re-reads in time (ops/depth_bench.py)
const budgetMsg = (n) => String((C.models[id(n)] || {}).think_msg || 'Considering the limited time by the user, I have to give the solution based on the thinking directly now.');
// the model card's sampling: think_sampling, the floor while thinking (a cold small quant loops, 12-oct); instruct_sampling, without
const card = (n, k) => (C.models[id(n)] || {})[k] || null;
module.exports = { options, has, id, load, unload, saveAll, pins, target, effort, stateOf, list, loaded, plan, takes, layer, budgets, budget, outMax, ctxUse, budgetMsg, card, slots, slotsOf, watch, idle, need, knob, phase, eta: (n) => Math.round(((known[id(n)] || {}).up || 0) / 1000), starting: () => Object.values(run).some((r) => r.g && !r.ready), roomWith: (f) => { room = f; }, learntWith: (f) => { learnt = f; }, names: () => Object.keys(C.models), up: () => !!C.bin && fs.existsSync(C.bin), version: () => ver, T };
