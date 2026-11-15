'use strict';
/*
 * video.js: every clip is sized to the model before llama-server reads it. llama.cpp samples frames at --video-fps and
 * caps nothing per video, so the gateway picks fps, size and frame count within the model's budget and the engine keeps
 * them (chat_local starts vision models with --video-fps 0). Defaults are Qwen3.x-VL's: 2 fps, 12288 tokens.
 * llama.cpp decodes as video any media that is not an image, whatever part carries it, so every such part comes here.
 * Nothing here may throw: the gateway exits on an uncaught error.
 */
const { execFile } = require('child_process');
const fs = require('fs'), os = require('os'), path = require('path'), crypto = require('crypto');

const D = { tokens: 12288, fps: 2, side: 360 };
const on = (b) => (process.env.PATH || '').split(path.delimiter).some((d) => { try { fs.accessSync(path.join(d, b), fs.constants.X_OK); return true; } catch (_) { return false; } });
// llama.cpp decodes video with ffmpeg and ffprobe, and /props says video either way
const ok = () => on('ffmpeg') && on('ffprobe');
// bytes stb or the WebP path read as an image; anything else llama.cpp hands to ffmpeg
const img = (u) => /^(\xff\xd8\xff|\x89PNG|GIF8|BM|RIFF[\s\S]{4}WEBP)/.test(Buffer.from(u.slice(u.indexOf(',') + 1, u.indexOf(',') + 17), 'base64').toString('latin1'));

const memo = new Map();   // clip + budget -> fitted clip: a turn resends its clip on every tool round
const run = (b, a, o, cb) => execFile(b, a, { timeout: 300000, maxBuffer: 1 << 20, signal: o.signal }, cb);

// Frames 32-px aligned (llama.cpp's unit: a frame pair costs one token per 32x32 px) and never more than the budget
// pays for: a long clip gets fewer fps to fit whole, one of unknown length is cut at the budget.
function one(uri, o, cb) {
  let done = 0, dir = '';
  const end = (e, v) => { if (done++) return; if (dir) fs.rm(dir, { recursive: true, force: true }, () => {}); cb(e, v); };
  const safe = (f) => (...a) => { try { f(...a); } catch (e) { end(e); } };
  const k = crypto.createHash('sha1').update(uri).digest('hex') + o.tokens;
  if (memo.has(k)) return end(null, memo.get(k));
  fs.mkdtemp(path.join(os.tmpdir(), 'vid-'), safe((e0, t) => {
    if (e0) return end(e0);
    dir = t; const i = path.join(dir, 'i'), f = path.join(dir, 'o.mp4');
    fs.writeFile(i, Buffer.from(uri.slice(uri.indexOf(',') + 1), 'base64'), safe((we) => {
      if (we) return end(we);
      run('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,duration:stream_side_data=rotation:format=duration', '-of', 'json', i], o, safe((e, out) => {
        let s = {}, d = 0; try { const j = JSON.parse(out); s = (j.streams || [])[0] || {}; d = +(j.format || {}).duration || +s.duration || 0; } catch (_) {}
        if (e || !s.width) return end(new Error('video: unreadable'));
        // ffmpeg turns the frames upright before scaling: a portrait phone clip is stored landscape
        const q = Math.abs(+((s.side_data_list || []).find((x) => 'rotation' in x) || {}).rotation || 0) % 180 === 90;
        const W = q ? s.height : s.width, H = q ? s.width : s.height, r = Math.min(1, o.side / Math.min(W, H));
        const w = Math.max(32, Math.floor(W * r / 32) * 32), h = Math.max(32, Math.floor(H * r / 32) * 32);
        const n = 2 * Math.max(1, Math.floor(o.tokens * 1024 / (w * h)));
        run('ffmpeg', ['-v', 'error', '-i', i, '-an', '-vf', 'fps=' + (d > 0 ? Math.min(o.fps, n / d) : o.fps) + ',scale=' + w + ':' + h,
          '-frames:v', String(n), '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', '-pix_fmt', 'yuv420p', f], o, safe((e2) => {
          if (e2) return end(new Error('video: could not be converted'));
          fs.readFile(f, safe((re, b) => {
            if (re) return end(re);
            const v = 'data:video/mp4;base64,' + b.toString('base64');
            memo.set(k, v); if (memo.size > 8) memo.delete(memo.keys().next().value);
            end(null, v);
          }));
        }));
      }));
    }));
  }));
}

// Body with every clip fitted, as llama.cpp's 'input_video'. pre = the body already parsed, when the gateway has it.
// o: {can (the model takes video now), ctx (served context, it bounds the budget too), signal (aborts the conversion),
// tokens, fps, side}. cb(err, bytes, parsed): the bytes as they came when no clip was fitted, and the one parse made.
function fit(body, pre, o, cb) {
  if (body.indexOf('data:') < 0) return cb(null, body, pre);
  let m = pre; const parts = [];
  try {
    if (!m) m = JSON.parse(body.toString('utf8'));
    (Array.isArray(m.messages) ? m.messages : []).forEach((msg) => Array.isArray(msg && msg.content) && msg.content.forEach((p) => {
      const x = p && (p.video_url || p.image_url || p.image || p.url), u = typeof x === 'string' ? x : x ? x.url : p && (p.input_video || {}).data;
      if (typeof u === 'string' && /^data:/.test(u) && !(/^data:image\//.test(u) && img(u))) parts.push([p, u]);
    }));
  } catch (_) { return cb(null, body, pre); }
  if (!parts.length) return cb(null, body, m);
  if (!o.can) return cb(new Error('this model does not take video'));
  o = Object.assign({}, D, o); if (o.ctx) o.tokens = Math.min(o.tokens, o.ctx >> 1);
  const next = (n) => {
    if (n === parts.length) return cb(null, Buffer.from(JSON.stringify(m)), m);
    one(parts[n][1], o, (e, v) => {
      if (e) return cb(e);
      const p = parts[n][0]; delete p.video_url; delete p.image_url; delete p.image; delete p.url;
      p.type = 'input_video'; p.input_video = { data: v };
      next(n + 1);
    });
  };
  next(0);
}

module.exports = { ok, fit };
