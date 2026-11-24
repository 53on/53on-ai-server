#!/usr/bin/env node
/*
 * ia_server: OpenAI-compatible gateway over local llama.cpp chat models, local image models, cloud providers and Claude Code.
 * Zero dependencies. Every client request needs an API key; a key may be limited to a model list and a token quota.
 * Public: /v1/models, /v1/chat/completions, /v1/completions, /v1/messages (Anthropic), /meta; /img/<32 hex>.png without key.
 * Admin: panel at /admin, API at /admin/api/* (x-admin-token).
 */
'use strict';
const http   = require('http');
const https  = require('https');
const fs     = require('fs');
const path   = require('path');
const os     = require('os');
const crypto = require('crypto');

// Log the cause to crash.log and exit: never keep running in a corrupt state (systemd restarts it).
function _crash(tag, e) { try { fs.appendFileSync(path.join(__dirname, 'crash.log'), new Date().toISOString() + ' ' + tag + ' :: ' + ((e && e.stack) || e) + os.EOL + os.EOL); } catch (_) {} stop(1); }
process.on('uncaughtException', (e) => _crash('uncaughtException', e));
process.on('unhandledRejection', (e) => _crash('unhandledRejection', e));
// exit once every engine has: a sooner restart reads the memory of a process still dying
const down = () => { images.unload(); videos.unload(); chat.saveAll(() => { const w = () => images.held() || videos.held() ? setTimeout(w, 200) : stop(0); w(); }); };
process.on('SIGTERM', down);
process.on('SIGINT', down);

// this server's version and its home: the panel's About asks GitHub for a newer release (the browser asks, never the server)
const VERSION = '1.0.0', REPO = '53on/53on-ai-server';
const CFG_PATH = path.join(__dirname, 'config.json');
let cfg = { port: 11500, public_url: '', max_concurrent: 1, keys: [], admin_token: '', lang: 'en' };
// config.json, or on a first start the template (the first save writes it as config.json)
try { Object.assign(cfg, JSON.parse(fs.readFileSync(fs.existsSync(CFG_PATH) ? CFG_PATH : path.join(__dirname, 'config.example.json'), 'utf8'))); }
catch (e) { console.error('[ia_server] config.json not loaded, using defaults:', e.message); }

const engine    = require('./claude_engine');
const chat      = require('./chat_local');
const cloud     = require('./cloud_relay');
const images    = require('./image_local');
const videos    = require('./video_local');
const mem       = require('./mem');
const video     = require('./video');
const anthropic = require('./anthropic');
const i18n      = require('./i18n');
images.options(cfg.images);
videos.options(Object.assign({ bin: (cfg.images || {}).bin || '' }, cfg.videos));
i18n.lang(cfg.lang);   // the panel's language; one with no lang/<code>.json stays English
// Owner cap (memory): a chat model serves min(its trained context, cap).
const ctxCap    = () => Math.max(1024, +cfg.ctx_cap || 32768);   // default window for a card without ctx
// The engine is a prebuilt llama.cpp in ~/engines/llama-<tag>, never compiled here. ops/update_engine.sh downloads,
// checks and smoke-tests one, then swaps it in through POST /admin/api/engine {bin}: live, no restart, no sudo.
// What is particular to the engine (name, source repo, folder prefix, build file, guide) lives in config chat.engine.
const ENGD = path.join(os.homedir(), 'engines'), EN = () => (cfg.chat || {}).engine || {};
let ENG = null;   // the running or last update: {tag, state: run|ok|failed, line}
// installed builds by folder, whatever their version names look like; newest folder first
const engines = () => { const d = EN().dir; if (!d) return [];
  try { return fs.readdirSync(ENGD).filter((x) => !x.indexOf(d) && /^[\w.-]+$/.test(x)).sort((a, b) => fs.statSync(path.join(ENGD, b)).mtimeMs - fs.statSync(path.join(ENGD, a)).mtimeMs).map((x) => x.slice(d.length)); } catch (_) { return []; } };
chat.options(cfg.chat, ctxCap());
// Names stored with a ':latest' tag are the same model as the bare name.
// local chat turns generating now: a local picture waits for them, the GPU is shared
let chatGen = 0;
images.busyWith(() => chatGen > 0); videos.busyWith(() => chatGen > 0);
// Memory gate (mem.js measures; the gateway owns both engines). A load starts when no chat model is still starting (the
// next one measures what it took), a chat model has no picture model beside it (its own RAM is not in gb) and the GPU
// can take gb. Meanwhile a QUIET engine lets go (a busy one never), measured every 500 ms (a killed process frees on
// exit) while the load is wanted, until the loader's T.load: then said in words. Unread memory (NaN) refuses nothing.
function room(gb, who, live, cb) {
  const img = images.has(who) || videos.has(who), lim = Date.now() + (videos.has(who) ? videos : images.has(who) ? images : chat).T.load;
  let said = 0;
  const again = () => {
    if (!live()) return;
    const f = mem.free();
    if (!(chat.starting() || !img && (images.held() || videos.held()) || f < gb)) return cb(null);
    if (Date.now() >= lim) return cb(new Error('Not enough memory for ' + who + ': needs ' + gb + ' GB, ' + f + ' GB free.'));
    // a picture and a clip model never sit together: the quiet one lets go, as an idle chat does
    if (img) { chat.idle().forEach((n) => chat.unload(n, false, null, 'yield:' + who)); if (videos.has(who) && !images.jobs()) images.unload(); if (images.has(who) && !videos.jobs()) videos.unload(); }
    else { if (!images.jobs()) images.unload(); if (!videos.jobs()) videos.unload(); }
    if (!said++) console.log('[mem] ' + who + ' needs ' + gb + ' GB, ' + f + ' GB free: waiting for a quiet engine to let go');
    setTimeout(again, 500);
  };
  again();
}
// the picture engine let go and owes no picture: the pinned brain comes back
images.roomWith(room); chat.roomWith(room); images.freedWith(pinLocks); videos.roomWith(room); videos.freedWith(pinLocks);
// engines' measures land in the catalog (measured.*)
images.grewWith(() => saveConfig(true)); videos.grewWith(() => saveConfig(true)); chat.learntWith(() => saveConfig(true));
let   CLA       = engine.options(cfg.claude);
function applyClaude() { CLA = engine.options(cfg.claude); }
let   CLOUD     = cloud.options(cfg.cloud);
function applyCloud() { CLOUD = cloud.options(cfg.cloud); }
const isCloudId = (m) => { if (!m) return false; const i = String(m).indexOf('/'); return i > 0 && !!CLOUD.providers[String(m).slice(0, i)]; };
const isClaude  = (m) => !!(CLA.enabled && m && String(m) === CLA.id);
const claudeModel = () => { const o = { id: CLA.id, object: 'model', owned_by: 'claude-code' }; o[CTX_FIELD] = CLA.context_length; o.max_context_length = CLA.context_length; o.context_length_source = 'config'; return o; };
// Panel row with the same shape as a local model; cloud:true = no disk, no VRAM.
const panelModel = () => ({
  name: CLA.id, gb: 0, params: '', quant: '', fam: '', caps: [], ctx: CLA.context_length, cloud: true,
  prov: 'claude', provLabel: 'Claude Code', state: CLA.enabled ? 'ok' : 'off', ctxSrc: 'config',
});
// Inference body cap (vision requests carry base64 images).
const MAXBODY   = Math.max(1, +(cfg.max_body_mb || 25)) * 1024 * 1024;
// Per-key image compatibility, name-free so a new drawer model needs no key edit: '' none, 'create', 'edit', 'both'.
// The chat model is always the loaded one; this only gates the drawer.
const imgFlag = (v) => v === 'create' || v === 'edit' || v === 'both' ? v : '';
// Selected = served, one model or many, as a cloud provider: every key sees the same list and none names models.
const selected = () => (cfg.on || []).filter((m) => chat.has(m) || isClaude(m) || isCloudId(m));
function modelFor(asked) { const on = selected(); if (!asked) return on[0] || null; asked = chat.id(asked); return on.includes(asked) ? asked : null; }
const CTX_FIELD = 'context_length';
let   MAXC      = Math.max(1, cfg.max_concurrent || 1);
// context_length = what the running model serves (or will serve: planned); max_context_length = trained maximum
const ctxOf = (m) => m.state === 'loaded' ? { ctx: m.ctx, src: 'served' } : { ctx: m.ctxPlan, src: 'planned' };
const KEYS      = new Map();   // secret -> key record

function applyKeys() { KEYS.clear(); (cfg.keys || []).forEach(k => KEYS.set(String(k.key), { name: k.name || 'unnamed', image: imgFlag(k.image), quota: +k.quota || 0, quota_at: k.quota_at || '', quota_days: +k.quota_days || 0, quota_day: +k.quota_day || 0 })); }

// Usage lives apart from config.json (it changes on every request) and is keyed by the key secret.
// Written 2 s deferred and atomically; mode 600 because the JSON keys are API secrets.
const USE_PATH = path.join(__dirname, 'usage.json');
let USE = { keys: {}, models: {}, days: {} }, _useT = null;
try { Object.assign(USE, JSON.parse(fs.readFileSync(USE_PATH, 'utf8'))); } catch (_) {}
function writeUsage() {
  _useT = null;
  try { fs.writeFileSync(USE_PATH + '.tmp', JSON.stringify(USE), { mode: 0o600 }); fs.renameSync(USE_PATH + '.tmp', USE_PATH); } catch (_) {}
}
function saveUsage() { if (!_useT) _useT = setTimeout(writeUsage, 2000); }
// Pending usage is flushed on every exit path.
function stop(code) { try { if (_useT) { clearTimeout(_useT); writeUsage(); } } catch (_) {} process.exit(code); }
const today = () => new Date().toISOString().slice(0, 10);
// Pulse timestamps: local HH:MM:SS. quota_at stays UTC ISO because it is stored and compared.
const clock = () => {
  const d = new Date(), p = n => String(n).padStart(2, '0');
  return p(d.getHours()) + ':' + p(d.getMinutes()) + ':' + p(d.getSeconds());
};

// Pulse: the last 60 requests, in memory only (history lives in usage.json).
const PULSE = [];
function pulse(e) { PULSE.unshift(e); if (PULSE.length > 60) PULSE.length = 60; }
// t/s the client saw the engine WRITE. The old out/ms mixed the prompt read into it, so a turn served from cache
// looked slow; a local engine says its own rate (timings.predicted_per_second) and the rest falls back to out/ms.
const wtps = (d) => d.wtps != null ? d.wtps : (d.tout && d.ms) ? Math.round(d.tout / (d.ms / 1000) * 10) / 10 : null;
function usageOf(key) { return USE.keys[key] || { in: 0, out: 0, req: 0, last: '' }; }
// Today's tokens of a key (UTC day): {d, in, out}; a record from another day counts as zero.
function dayOf(key) { const d = usageOf(key).day; return d && d.d === today() ? d : { d: today(), in: 0, out: 0 }; }
// Client-facing reason a key is blocked, or null. quota 0 = no token limit; quota_days 0 = no expiry (counted from quota_at);
// quota_day 0 = no daily limit (else tokens per UTC day, back tomorrow). The plan and its expiry take a top-up; the day's
// cap comes back on its own.
const TOPUP = ' Please contact the administrator to top up your key.';
function blockReason(key, krec) {
  if (!krec) return null;
  const q = +krec.quota || 0;
  if (q) {
    const u = usageOf(key);
    if ((u.in + u.out) >= q) return 'Insufficient credits. You have used ' + (u.in + u.out) + ' of ' + q + ' tokens.' + TOPUP;
  }
  const qd = +krec.quota_day || 0;
  if (qd) {
    const y = dayOf(key);
    if ((y.in + y.out) >= qd) return 'Daily limit reached. You have used ' + (y.in + y.out) + ' of ' + qd + ' tokens today; it resets at 00:00 UTC.';
  }
  const d = +krec.quota_days || 0;
  if (d && krec.quota_at) {
    const end = new Date(krec.quota_at).getTime() + d * 86400000;
    if (Date.now() > end) return 'Your plan expired on ' + new Date(end).toISOString().slice(0, 16).replace('T', ' ') + ' UTC.' + TOPUP;
  }
  return null;
}
// Plan expiry (ISO), or '' when the key does not expire.
function expiresAt(krec) {
  const d = +(krec && krec.quota_days) || 0;
  if (!d || !krec.quota_at) return '';
  return new Date(new Date(krec.quota_at).getTime() + d * 86400000).toISOString();
}
function track(key, model, tin, tout, imgs, usd) {
  if (!tin && !tout && !imgs) return;
  const k = USE.keys[key] || (USE.keys[key] = { in: 0, out: 0, req: 0, last: '' });
  k.in += tin; k.out += tout; k.req++; if (imgs) k.img = (k.img || 0) + imgs; if (usd) k.usd = +((k.usd || 0) + usd).toFixed(6); k.last = new Date().toISOString();
  // the day's count (quota_day) starts over when the UTC date changes
  const y = k.day = dayOf(key); y.in += tin; y.out += tout;
  const m = USE.models[model] || (USE.models[model] = { in: 0, out: 0, req: 0 });
  m.in += tin; m.out += tout; m.req++; if (imgs) m.img = (m.img || 0) + imgs; if (usd) m.usd = +((m.usd || 0) + usd).toFixed(6);
  const d = USE.days[today()] || (USE.days[today()] = { in: 0, out: 0, req: 0 });
  d.in += tin; d.out += tout; d.req++; if (imgs) d.img = (d.img || 0) + imgs; if (usd) d.usd = +((d.usd || 0) + usd).toFixed(6);
  const ds = Object.keys(USE.days).sort();
  while (ds.length > 60) delete USE.days[ds.shift()];   // keep 60 days
  saveUsage();
}
// atomic like writeUsage: a crash mid-write must never leave a truncated config.json (admin_token '' = panel locked out).
// The version before the last admin edit stays as config.json.prev (one step back, same 600 mode): the only backup that
// holds keys. auto = a write nobody asked for (traffic pinning a model back): it leaves .prev alone.
function saveConfig(auto) { try { cfg.max_concurrent = MAXC; if (!auto) { try { fs.copyFileSync(CFG_PATH, CFG_PATH + '.prev'); } catch (_) {} } fs.writeFileSync(CFG_PATH + '.tmp', JSON.stringify(cfg, null, 2), { mode: 0o600 }); fs.renameSync(CFG_PATH + '.tmp', CFG_PATH); return true; } catch (e) { return false; } }
const genKey = (pfx) => (pfx || 'sk-pc-') + crypto.randomBytes(15).toString('hex');
// LAN IPv4 addresses; virtual adapters (matched by interface name) are unreachable from other devices.
const VIRT_RE = /vethernet|virtualbox|vmware|hyper-?v|loopback|\bwsl\b|docker|bluetooth|\bvpn\b|tailscale|zerotier/i;
const lanIPs = () => {
  const out = [];
  for (const [name, addrs] of Object.entries(os.networkInterfaces())) {
    if (VIRT_RE.test(name)) continue;
    for (const i of addrs || []) if (i.family === 'IPv4' && !i.internal) out.push(i.address);
  }
  return out;
};
function readBody(req, cb) { const c = []; req.on('data', d => c.push(d)); req.on('end', () => { let b = {}; try { b = JSON.parse(Buffer.concat(c).toString('utf8') || '{}'); } catch (_) {} cb(b && typeof b === 'object' ? b : {}); }); }
applyKeys();
applyClaude();
applyCloud();
const prices = require('./prices');
// only the served models are kept: the source table is 2 MB
// Prices are asked for what is CONFIGURED, on or not today: a provider switched on mid-afternoon
// bills its first request with a price, not the next day.
prices.refresh(cloud.panelModels(CLOUD).map((m) => m.name), () => {});

const ALLOW = [
  { m: 'GET',  re: /^\/v1\/models\/?$/ },
  { m: 'POST', re: /^\/v1\/chat\/completions\/?$/ },
  { m: 'POST', re: /^\/v1\/completions\/?$/ },
];
const pathOf  = (u) => (u || '').split('?')[0];
// requests browsers make on their own
const BROWSER_NOISE = /^\/(favicon\.ico|robots\.txt|sitemap\.xml|apple-touch-icon.*\.png|browserconfig\.xml|\.well-known\/)/;
const allowed = (req) => ALLOW.some(a => a.m === req.method && a.re.test(pathOf(req.url)));
// Local chat relayed like a provider, with no output floor (a 1-token warmup stays 1 token) and a long idle limit:
// a 20K-token prompt is read for ~2 min before the first byte.
// Local chat relayed like a provider, with no output floor (a 1-token warmup stays 1 token) and no idle limit: a long
// prompt or a request waiting for the model's slot sends no byte for minutes, and a dead llama-server resets the socket.
const LOCAL = () => Object.assign({}, CLOUD, { min_output: 0, timeout_ms: 0 });
// Media reaches llama-server only as data URLs: it would fetch any other URL itself, LAN addresses included. Our own
// /img/<hex>.png is read from disk, other URLs are dropped (the model does not see them), legacy part shapes rewritten.
// A file pasted as TEXT is read as text: one screenshot's base64 is ~700K tokens (measured), minutes of GPU for
// every turn that resends it, and the model still cannot see it. Inside text, an image data URL in a user turn becomes
// a real image part (~1K tokens) and any other base64 a short marker. Logged with the key, to find what sends it.
const B64 = /data:([\w.+-]+\/[\w.+-]+);base64,[A-Za-z0-9+/=]{256,}/g;
// Every special of the vocab (27 CONTROL + 6 USER_DEFINED, the same in both local GGUFs): /tokenize turns each into
// ONE id even inside a user's text. Lookalikes the vocab lacks (<tools>, <function=..>, <|user|>) are plain text and
// stay out. A zero-width space after the '<' makes one plain text.
const CTRL = /<(?=\|(?:endoftext|im_start|im_end|(?:object_ref|box|quad|vision|audio)_(?:start|end)|(?:vision|image|video|audio|fim)_pad|fim_(?:prefix|middle|suffix)|repo_name|file_sep)\|>|tts_(?:pad|text_bos|text_eod|text_bos_single)>|\/?(?:think|tool_call|tool_response)>)/g;
// true when a part changed (the relay serializes only then); a part already in the engine's shape is kept
function localParts(msgs, who) {
  let cut = 0, chg = false;
  const drop = () => { chg = true; return null; };
  // s with the zero-width space after the '<' of each literal that starts in it; nx = what the template writes next
  const z = (s, nx) => { let n = 0; const r = (s + nx).replace(CTRL, (x, i) => { if (i < s.length) n++; return '<\u200b'; }); if (!n) return s; chg = true; return r.slice(0, s.length + n); };
  (Array.isArray(msgs) ? msgs : []).forEach((m) => {
    if (m && typeof m.content === 'string' && m.content.indexOf(';base64,') > -1) { m.content = [{ type: 'text', text: m.content }]; chg = true; }
    if (m && Array.isArray(m.content)) m.content = m.content.reduce((a, p) => {
      if (typeof p === 'string') { p = { type: 'text', text: p }; chg = true; }
      if (!p || p.type !== 'text' || typeof p.text !== 'string' || p.text.indexOf(';base64,') < 0) return a.concat(p);
      const img = [], text = p.text.replace(B64, (d, mime) => {
        cut += d.length;
        if (m.role === 'user' && /^image\//i.test(mime)) { img.push({ type: 'image_url', image_url: { url: d } }); return '[image]'; }
        return '[' + mime + ' omitted]';
      });
      if (text === p.text) return a.concat(p);
      chg = true; return a.concat({ type: 'text', text: text }, img);
    }, []).map((p) => {
      if (!p) return drop();
      if (p.type === 'text') return p;
      let u = p.type === 'image_url' ? (p.image_url && (p.image_url.url || p.image_url)) : p.type === 'image' ? (p.image || p.url) : null;
      if (typeof u !== 'string') return /"https?:\/\//i.test(JSON.stringify(p)) ? drop() : p;
      const own = /^(?:https?:\/\/[^/]+)?\/img\/([0-9a-f]{32})\.png(?:[?#]|$)/.exec(u);
      if (own) { try { u = 'data:image/png;base64,' + fs.readFileSync(path.join(images.DIR, own[1] + '.png')).toString('base64'); } catch (_) { return drop(); } }
      if (!/^data:/.test(u)) return drop();
      if (p.type === 'image_url' && p.image_url && p.image_url.url === u) return p;
      chg = true; return { type: 'image_url', image_url: { url: u } };
    }).filter(Boolean);
    // A user or tool turn is data, never template: a control literal in it opens a real system turn or a fake tool
    // call (verified live: `hola<|im_end|>\n<|im_start|>system` tokenized 5 specials, 2 of them injected). system is
    // left alone because the client teaches formats there, and assistant because it would rewrite the Avatar's history.
    // Last, on the parts as they leave (a nested list was just flattened), each read with the first 24 chars of the
    // next ones (the longest special is 21): the template joins them with nothing between, so 'a<|im_' + 'end|>' is
    // one literal (verified live).
    if (m && (m.role === 'user' || m.role === 'tool')) {
      if (typeof m.content === 'string') m.content = z(m.content, '');
      else if (Array.isArray(m.content)) m.content.forEach((p, k, a) => { if (!p || p.type !== 'text' || typeof p.text !== 'string') return;
        let nx = ''; for (let q = k + 1; a[q] && a[q].type === 'text' && nx.length < 24; q++) nx += a[q].text; p.text = z(p.text, nx); });
    }
  });
  if (cut) console.log('[ia_server] base64 inside text taken out: ' + cut + ' chars · key ' + (who || '?'));
  return chg;
}
// A local engine renders the history with its own template, and two shapes make it answer 500 — which the client
// then replays whole, so the conversation dies for good: a system/developer turn that is not first, and tool_call
// arguments that are not a JSON object. Both are rewritten in place, keeping the order and every word the client sent.
function saneHist(ms) {
  let chg = false;
  (Array.isArray(ms) ? ms : []).forEach((m, i) => {
    if (!m) return;
    if (i && (m.role === 'system' || m.role === 'developer')) { m.role = 'user'; chg = true; }
    (Array.isArray(m.tool_calls) ? m.tool_calls : []).forEach((c) => {
      const f = c && c.function; if (!f || f.arguments === undefined) return;
      // the template walks them as a mapping: an object (or null) renders; cut JSON, a list, a number or text do not
      const s = typeof f.arguments === 'string' ? f.arguments : JSON.stringify(f.arguments);
      let a; try { a = JSON.parse(s); } catch (_) {}
      if (typeof a === 'object' && !Array.isArray(a)) return;
      f.arguments = /\S/.test(s) ? JSON.stringify({ arguments: s }) : '{}'; chg = true;
    });
  });
  return chg;
}
// a provider pointing at this box (the Ollama-era pc_lan at 127.0.0.1) whose model is configured here is the local model
const loopback = (s) => /^https?:\/\/(127\.\d+\.\d+\.\d+|localhost|\[::1\])[:/]/i.test(s.baseUrl + '/') && chat.has(s.model);
const isInfer = (req) => req.method === 'POST' && /^\/v1\/(chat\/completions|completions)\/?$/.test(pathOf(req.url));

function clientKey(req) {
  const auth = req.headers['authorization'] || '';
  if (auth.startsWith('Bearer ')) return auth.slice(7).trim();
  if (req.headers['x-api-key']) return String(req.headers['x-api-key']).trim();
  return '';   // never from the URL: a query string lands in every proxy log
}
// OpenAI-style rate-limit headers on every authenticated response; only known values are sent. With a daily limit
// the headers describe whichever runs out first: the plan, or the day (reset = next 00:00 UTC).
function rateHeaders(krec, key) {
  const h = { 'x-request-id': reqId() };
  const q = +(krec && krec.quota) || 0, qd = +(krec && krec.quota_day) || 0;
  const u = usageOf(key), left = q ? Math.max(0, q - (u.in || 0) - (u.out || 0)) : Infinity;
  const y = qd ? dayOf(key) : null, dleft = qd ? Math.max(0, qd - y.in - y.out) : Infinity;
  if (qd && dleft <= left) {
    h['x-ratelimit-limit-tokens'] = String(qd);
    h['x-ratelimit-remaining-tokens'] = String(dleft);
    h['x-ratelimit-reset-tokens'] = new Date(Math.floor(Date.now() / 864e5 + 1) * 864e5).toISOString();
  } else if (q) {
    h['x-ratelimit-limit-tokens'] = String(q);
    h['x-ratelimit-remaining-tokens'] = String(left);
    const exp = expiresAt(krec);
    if (exp) h['x-ratelimit-reset-tokens'] = exp;
  }
  return h;
}
let _rid = 0;
const reqId = () => 'req_' + Date.now().toString(36) + (++_rid % 1e4).toString(36);

// internal tag -> the {type, code} pair official SDKs read
const ERR_MAP = {
  auth:               { type: 'invalid_request_error', code: 'invalid_api_key' },
  invalid:            { type: 'invalid_request_error', code: 'invalid_request' },
  too_large:          { type: 'invalid_request_error', code: 'request_too_large' },
  insufficient_quota: { type: 'insufficient_quota',    code: 'insufficient_quota' },
  forbidden:          { type: 'invalid_request_error', code: 'route_not_allowed' },
  not_found:          { type: 'invalid_request_error', code: 'not_found' },
  upstream:           { type: 'api_error',             code: 'upstream_error' },
  file:               { type: 'api_error',             code: 'internal_error' },
};

const jres = (res, code, obj) => {
  // a stream opened while its client waited ends with the error as an event, then [DONE] (Strata's way)
  if (res.headersSent) return res.writableEnded || res.end('data: ' + JSON.stringify(obj) + '\n\ndata: [DONE]\n\n');
  const h = Object.assign({ 'content-type': 'application/json' }, res._rl || { 'x-request-id': reqId() });
  res.writeHead(code, h); res.end(JSON.stringify(obj));
};
// Errors also land in the pulse, so the owner sees rejected requests.
const jerr = (res, code, msg, tag, param) => {
  const m = ERR_MAP[tag] || { type: 'api_error', code: tag || null };
  try { pulse({ t: clock(), key: '', model: '', in: 0, out: 0, ms: 0, queue: 0, tps: null, code: code, ok: false, err: m.code + ': ' + msg }); } catch (_) {}
  // the only 429 is an exhausted quota: SDKs retry 429 unless told not to
  if (code === 429) res._rl = Object.assign({}, res._rl, { 'x-should-retry': 'false' });
  return jres(res, code, { error: { message: msg, type: m.type, param: param || null, code: m.code } });
};

// Permits per engine, each its own line: 'local:<model>' = the slots its llama-server serves (the gateway sees when the
// server is full instead of hiding requests in the engine's deferred queue), 'cloud' and 'claude' max_concurrent each, so
// the cloud never waits behind the GPU. A waiting request whose client leaves is dropped from the line and never runs.
const Q = {};
const qOf = (e) => Q[e] || (Q[e] = { n: 0, w: [] });
const capOf = (e) => e.indexOf('local:') ? MAXC : chat.slotsOf(e.slice(6));
const acquire = (res, e) => new Promise(r => {
  const q = qOf(e);
  if (q.n < capOf(e)) { q.n++; return r(); }
  q.w.push(res._w = r); res.once('close', () => { const i = q.w.indexOf(r); if (i > -1) q.w.splice(i, 1); });
});
// the permit passes to the next in line, unless the engine now serves fewer than are running (a model restarted smaller)
const release = (e) => { const q = qOf(e); if (q.w.length && q.n <= capOf(e)) q.w.shift()(); else q.n--; };
// [running, waiting, permits] per engine, and the totals the panel and ops/ias.py read
const permits = () => { const o = {}; Object.keys(Q).forEach((e) => { o[e] = [Q[e].n, Q[e].w.length, capOf(e)]; }); return o; };
const inFlight = () => Object.keys(Q).reduce((a, e) => a + Q[e].n, 0), queued = () => Object.keys(Q).reduce((a, e) => a + Q[e].w.length, 0);

const TRANSMIT_STANDARD = {
  endpoint: 'POST /v1/chat/completions',
  required: { model: 'string, an id from /v1/models', messages: 'array of {role, content}' },
  optional: { stream: 'bool', max_tokens: 'int (output)', temperature: '0..2', top_p: '0..1', top_k: 'int', stop: 'string|[]', presence_penalty: '-2..2', frequency_penalty: '-2..2', seed: 'int', response_format: '{type:json_object}', tools: 'array' },
  token_limit: 'context_length per model, shared by input and output; it is what the loaded model actually serves',
  context_fields: { context_length: 'served now (running model or provider)', max_context_length: 'trained maximum, not what is served', context_length_source: 'served | planned | provider | catalog | default | config' }
};

// Sensors: Linux sysfs first (hwmon found by name), nvidia-smi only as fallback. Sampled on
// request, cached, coalesced. A missing sensor is null, never an error.
const { execFile, spawn } = require('child_process');
const HW = '/sys/class/hwmon/';
let _hw = null, _amd = 0, _smi = true, _cpu0 = null, _snap = null, _snapT = 0, _wait = null;
const rd = (f) => { try { return fs.readFileSync(f, 'utf8').trim(); } catch (_) { return null; } };
const num = (f, d) => { const v = rd(f); return v === null || v === '' || isNaN(v) ? null : v / (d || 1); };
const r1 = (v) => v == null ? null : Math.round(v * 10) / 10;
const toMb = (b) => Math.round(b / 1048576);
function hwmon() {
  if (_hw) return _hw;
  _hw = {};
  const vt = (p) => +rd(p + 'device/mem_info_vram_total');
  // several amdgpu (APU + discrete): keep the one with the most VRAM
  try { fs.readdirSync(HW).sort().forEach(d => { const p = HW + d + '/', n = rd(p + 'name'); if (n === 'amdgpu') _amd++; if (n && (!_hw[n] || n === 'amdgpu' && vt(p) > vt(_hw[n]))) _hw[n] = p; }); } catch (_) {}
  return _hw;
}
function cpuPct(cs) {
  let idle = 0, all = 0;
  cs.forEach(c => { const t = c.times; idle += t.idle; all += t.user + t.nice + t.sys + t.idle + t.irq; });
  const p = _cpu0 && all > _cpu0[1] ? r1(100 - (idle - _cpu0[0]) / (all - _cpu0[1]) * 100) : null;
  _cpu0 = [idle, all];
  return p;
}
function diskRoot() {
  try { const s = fs.statfsSync(path.parse(__dirname).root), u = (s.blocks - s.bfree) * s.bsize; return [toMb(u), toMb(u + s.bavail * s.bsize)]; } catch (_) { return null; }
}
function gpuSys(d, cs) {
  if (!d) return null;
  const v = d + 'device/', pair = (a) => { const u = num(v + 'mem_info_' + a + '_used'), t = num(v + 'mem_info_' + a + '_total'); return u == null || t == null ? null : [toMb(u), toMb(t)]; };
  const n = _amd < 2 && ((cs[0] || {}).model || '').match(/w\/ (Radeon [^,]+)/);
  return { src: 'amdgpu', apu: !!n, name: n ? n[1].trim() : 'AMD GPU', c: r1(num(d + 'temp1_input', 1000)), busy: num(v + 'gpu_busy_percent'),
    w: r1(num(d + 'power1_average', 1e6) ?? num(d + 'power1_input', 1e6)), vram: pair('vram'), gtt: pair('gtt') };
}
function gpuSmi(cb) {
  if (!_smi) return cb(null);
  execFile('nvidia-smi', ['--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw', '--format=csv,noheader,nounits'],
    { timeout: 4000, windowsHide: true }, (e, out) => {
      if (e) { if (e.code === 'ENOENT') _smi = false; return cb(null); }
      const f = String(out).split(/\r?\n/)[0].split(',').map(x => x.trim()), n = (i) => f[i] === '' || isNaN(f[i]) ? null : +f[i];
      cb(f[0] ? { src: 'nvidia-smi', name: f[0], c: n(4), busy: n(3), w: r1(n(5)), vram: n(1) == null || n(2) == null ? null : [n(1), n(2)], gtt: null } : null);
    });
}
function sample(cb) {
  if (Date.now() - _snapT < 1800) return cb(_snap);
  if (_wait) return _wait.push(cb);
  _wait = [cb];
  const h = hwmon(), cs = os.cpus(), k = h.k10temp || h.zenpower || h.coretemp, tm = os.totalmem();
  const s = { t: Date.now(), up: Math.round(os.uptime()), srv_up: Math.round(process.uptime()),
    host: { cpu: cs[0] ? cs[0].model.trim() : null, cores: cs.length, os: process.platform },
    cpu: { pct: cpuPct(cs), c: k ? r1(num(k + 'temp1_input', 1000)) : null, load: process.platform === 'win32' ? null : os.loadavg().map(x => Math.round(x * 100) / 100) },
    ram: [toMb(tm - os.freemem()), toMb(tm)], disk: diskRoot(), gpu: gpuSys(h.amdgpu, cs), nvme_c: h.nvme ? r1(num(h.nvme + 'temp1_input', 1000)) : null };
  const done = (g) => { if (g) s.gpu = g; _snap = s; _snapT = Date.now(); const w = _wait; _wait = null; w.forEach(f => f(s)); };
  // hybrid (AMD iGPU + NVIDIA): report the card with the most VRAM; without nvidia-smi this is one ENOENT per process
  const a = s.gpu;
  let n2 = 2, g0 = null; const fin = () => { if (!--n2) done(g0); };
  gpuSmi(g => { g0 = g && (!a || !a.vram || g.vram && g.vram[1] >= a.vram[1]) ? g : null; fin(); });
  // the engine's slots (what each one reads or writes now) ride the same 2 s sample, never a request
  chat.slots((l) => { s.slots = l; fin(); });
}
// The monitor's graph does not start empty after a reload: the server keeps ITS OWN samples on disk while nobody
// watches, the way status_server keeps ss_metrics_history.json. [t, cpu °C, gpu °C, gpu %] · 15 min at the panel's own pace.
const HIST_F = path.join(__dirname, 'metrics_history.json'), HIST_N = 450, HIST_MS = 2000;
let hist = [], histW = 0;
// what is older than an hour is not history, it is rubbish: it does not come back on start
try { hist = JSON.parse(fs.readFileSync(HIST_F, 'utf8')).slice(-HIST_N).filter(p => Date.now() - p[0] < 36e5); } catch (_) {}
setInterval(() => sample((s) => {
  const g = s.gpu || {};
  hist.push([s.t, s.cpu.c, g.c == null ? null : g.c, g.busy == null ? null : g.busy]);
  if (hist.length > HIST_N) hist.splice(0, hist.length - HIST_N);
  if (++histW >= 15) { histW = 0; fs.writeFile(HIST_F, JSON.stringify(hist), () => {}); }   // one write per 30 s, not per sample
}), HIST_MS).unref();
// A model's thinking floor (config chat.models[m].think_sampling): temperature, top_p, top_k and presence_penalty never below
// it (top_k 0 = no limit, already open), min_p never above it; a value the client did not send takes the floor. True if o changed.
// her kit (her card): with `use`, her engine gets the tools her template shows; her own mg and ma carry no card, her template writes it
function kitTools(o, kit) {
  const tn = (x) => (x.function || x).name, had = new Set(o.tools.map(tn));
  o.tools = (kit.own || []).filter((x) => !had.has(x.name)).map((x) => ({ type: 'function', function: x })).concat(o.tools.filter((x) => (kit.tools || []).includes(tn(x))));
}
function liftToFloor(o, fl) {
  if (!fl) return false;
  let d = false;
  for (const k of ['temperature', 'top_p', 'top_k', 'presence_penalty']) {
    if (fl[k] == null || (o[k] != null && ((k === 'top_k' && +o[k] === 0) || +o[k] >= +fl[k]))) continue;
    o[k] = fl[k]; d = true;
  }
  if (fl.min_p != null && (o.min_p == null || +o.min_p > +fl.min_p)) { o.min_p = fl.min_p; d = true; }
  return d;
}
// a set fills only what the client left out; true if o changed
function fill(o, s) { let d = false; if (s) for (const k in s) if (o[k] == null) { o[k] = s[k]; d = true; } return d; }
// Thinking for a local request that names no level: config think_default (a value or
// {model: value, '*': value}), else off. Qwen3.8's template alone thinks at its maximum: 787 tokens, 75 s for one word.
function thinkDefault(n) {
  let t = cfg.think_default;
  if (t && typeof t === 'object') t = t[n] != null ? t[n] : t[n + ':latest'] != null ? t[n + ':latest'] : t['*'];
  return t == null ? false : t;
}
const fmtK = (n) => { n = +n || 0; if (n >= 1e6 && n % 1e6 === 0) return (n / 1e6) + 'M'; if (n % 1048576 === 0) return (n / 1048576) + 'M'; if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M'; if (n % 1024 === 0) return (n / 1024) + 'K'; return n >= 1000 ? Math.round(n / 1000) + 'K' : n; };
// A local model as an OpenAI list entry
function localModel(m) {
  const c = ctxOf(m), o = { id: m.name, object: 'model', owned_by: 'llama.cpp' };
  o[CTX_FIELD] = c.ctx || m.ctxPlan; o.max_context_length = m.ctxMax || o[CTX_FIELD]; o.context_length_source = c.src;
  o.context_length_display = fmtK(o[CTX_FIELD]);   // one truth: the server formats, every view shows this
  o.capabilities = m.caps;   // what the model takes now, so a client offers only that (Ollama's names)
  // llama-server's router names (Strata lists an unloaded model the same way): a client can say 'waking up (~14 s)' before it sends
  o.status = { value: m.state || 'unloaded' }; if (m.state !== 'loaded' && m.up) o.status.eta_s = Math.round(m.up / 1000);
  if (m.states) o.reasoning_levels = m.states;   // the thinking states it tells apart: the client's knob shows only these
  const om = chat.outMax(m.name); if (om) o.default_parameters = { max_tokens: om };   // OpenRouter's name: what applies when the client sends none
  // Anthropic's names, as Anthropic means them: the hard input window (the real one) and the longest answer. The useful window is a
  // FACT for the AI to decide when to sleep (ms); it never moves the faint at 1% of the real context (no threshold decides for her)
  o.max_input_tokens = o[CTX_FIELD]; if (om) o.max_tokens = om;
  const cu = chat.ctxUse(m.name); if (cu) o.ctx_use = cu;
  if (m.states) o.reasoning_budgets = chat.budgets(m.name);   // the thinking cap per level the server applies (the client's reasoning_budget_tokens wins)
  return o;
}
// /v1/models, with context_length.
const cloudRow = (id) => { const i = String(id).indexOf('/'), pv = String(id).slice(0, i), ci = cloud.ctxInfo(pv, String(id).slice(i + 1), null), o = { id: id, object: 'model', owned_by: pv }; o[CTX_FIELD] = ci.ctx; o.max_context_length = ci.ctx; o.context_length_source = ci.src; return o; };
const served = () => { const l = chat.list(); return [].concat(...selected().map((m) => isClaude(m) ? [claudeModel()] : isCloudId(m) ? [cloudRow(m)] : l.filter((r) => r.name === m).map(localModel))); };
const handleModels = (res) => jres(res, 200, { object: 'list', data: served() });

// The admin panel does not exist for the internet: it answers only private, loopback and VPN addresses (the image
// module's LAN list). A router that does SNAT makes the internet arrive from the gateway's own address, so that one
// is refused too; the gateway is read from the system, never hard-coded.
const ipOf = (req) => String(req.socket.remoteAddress || '').replace(/^::ffff:/, '');
const GW = (() => { try {
  const r = fs.readFileSync('/proc/net/route', 'utf8').split('\n').find(l => l.split(/\s+/)[1] === '00000000');
  return r ? r.split(/\s+/)[2].match(/../g).reverse().map(x => parseInt(x, 16)).join('.') : '';
} catch (_) { return ''; } })();
const fromNet = (req) => { const ip = ipOf(req); return (!!GW && ip === GW) || !images.LAN.check(ip, ip.includes(':') ? 'ipv6' : 'ipv4'); };
// Constant-time equality (no length or prefix leaks through timing); different lengths are simply unequal.
const same = (a, b) => { a = Buffer.from(String(a)); b = Buffer.from(String(b)); return a.length === b.length && crypto.timingSafeEqual(a, b); };

// ADMIN APPS: local web tools the owner lists in config.json "apps" (none by default), served on this port under their own path and
// behind this server's lock: the house network only (like /admin) and the admin token, typed once into a small sign-in page that trades
// it for an HttpOnly cookie scoped to the app's path (12 h; a restart signs everyone out). GETs pass through, event streams included; a
// POST passes only if its path matches one of the app's "post" patterns, comes from this server's own pages (same Origin) and is small.
// "open": true drops the sign-in for that app (the house network still only), for a tool that holds nothing to guard.
//   "apps": [{ "path": "/tool", "port": 39000, "post": ["^/run$"], "open": false }]
const APP_S = new Map();   // session id -> { path, exp }
const appOf = (p) => (cfg.apps || []).find((a) => a && typeof a.path === 'string' && /^\/[a-z0-9_-]+$/i.test(a.path) && !/^\/(v1|admin|health|meta)$/.test(a.path)
  && +a.port > 0 && (p === a.path || p.indexOf(a.path + '/') === 0));
const htm = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
function appSignIn(res, a, bad) {
  const t = i18n.bag('ui');
  res.writeHead(401, { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store' });
  res.end(`<!doctype html><html lang="${htm(i18n.lang())}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${htm(a.path)}</title><style>
:root{--bg:#f9fafb;--card:#fff;--ln:#dee5ee;--tx:#221e1f;--mut:#6f6f6f;--ac:#10b981;--aci:#fff;--dz:#e0283f;--fo:#405de6;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#0e1113;--card:#161a1d;--ln:#2a3036;--tx:#eef1f3;--mut:#8e979f;--ac:#34d399;--aci:#052e20;--dz:#ff5a6e;--fo:#7b93ff;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;padding:16px;background:var(--bg);color:var(--tx);font:14px/1.55 "GT Walsheim Pro",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
form{width:100%;max-width:360px;background:var(--card);border:1px solid var(--ln);border-radius:20px;padding:24px}
h1{margin:0 0 16px;font-size:18px;font-weight:900;letter-spacing:-.01em}label{display:block;font-size:12px;font-weight:500;color:var(--mut);margin-bottom:6px}
input{width:100%;padding:10px 12px;border:1px solid var(--ln);border-radius:10px;background:var(--bg);color:var(--tx);font:inherit}
input:focus{outline:none;border-color:var(--fo);box-shadow:0 0 0 3px color-mix(in srgb,var(--fo) 22%,transparent)}
button{margin-top:14px;width:100%;height:42px;border:0;border-radius:14px;background:var(--ac);color:var(--aci);font:inherit;font-weight:700;cursor:pointer}
.bad{color:var(--dz);font-size:13px;margin:8px 0 0}
</style></head><body><form method="post" action="${htm(a.path)}/_signin"><h1>${htm(a.path)}</h1>
<label for="t">${htm(t.admin_token)}</label><input id="t" name="token" type="password" autocomplete="current-password" autofocus required>
${bad ? `<p class="bad">${htm(t.bad_token)}</p>` : ''}<button>${htm(t.log_in)}</button></form></body></html>`);
}
function appDoor(req, res, p, a) {
  const now = Date.now(); for (const [k, s] of APP_S) if (s.exp < now) APP_S.delete(k);
  const q = req.url.indexOf('?') < 0 ? '' : req.url.slice(req.url.indexOf('?'));
  if (p === a.path) { res.writeHead(302, { location: a.path + '/' + q }); return res.end(); }   // the cookie's path ends in /
  const name = 'app' + a.path.replace(/[^a-z0-9]/gi, '_');
  if (p === a.path + '/_signin' && req.method === 'POST') {
    let b = ''; req.on('data', (d) => { b += d; if (b.length > 4096) req.destroy(); });
    return req.on('end', () => {
      if (!cfg.admin_token || !same(new URLSearchParams(b).get('token') || '', cfg.admin_token)) return setTimeout(() => appSignIn(res, a, true), 800);
      const id = crypto.randomBytes(24).toString('hex'); APP_S.set(id, { path: a.path, exp: now + 12 * 3600e3 });
      res.writeHead(303, { location: a.path + '/', 'cache-control': 'no-store',
        'set-cookie': `${name}=${id}; Path=${a.path}/; HttpOnly; SameSite=Strict; Max-Age=43200${req.socket.encrypted ? '; Secure' : ''}` });
      res.end();
    });
  }
  const m = new RegExp('(?:^|;\\s*)' + name + '=([0-9a-f]{48})(?:;|$)').exec(req.headers.cookie || ''), s = m && APP_S.get(m[1]);
  if (a.open !== true && (!s || s.path !== a.path)) return appSignIn(res, a, false);
  const sub = req.url.slice(a.path.length) || '/';   // path and query, as the tool sees them
  if (req.method === 'POST') {
    if (!(a.post || []).some((r) => { try { return new RegExp(r).test(sub); } catch (_) { return false; } })) return jerr(res, 404, 'Not found.', 'not_found');
    const o = req.headers.origin;
    if (o && o !== (req.socket.encrypted ? 'https://' : 'http://') + req.headers.host) return jerr(res, 403, 'Cross-origin request refused.', 'origin');
  } else if (req.method !== 'GET') return jerr(res, 405, 'Method not allowed.', 'method');
  const up = http.request({ host: '127.0.0.1', port: +a.port, method: req.method, path: sub,
    headers: Object.assign({ 'accept-language': req.headers['accept-language'] || '' }, req.method === 'POST' ? { 'content-type': req.headers['content-type'] || 'application/json' } : {}) }, (r) => {
    const h = { 'cache-control': 'no-store' };
    for (const n of ['content-type', 'content-length', 'content-disposition']) if (r.headers[n]) h[n] = r.headers[n];
    res.writeHead(r.statusCode, h); r.pipe(res);
  });
  up.on('error', () => { if (!res.headersSent) jerr(res, 502, 'The app at ' + a.path + ' is not running.', 'app_down'); else res.end(); });
  res.on('close', () => up.destroy());   // the tab closed: its event stream to the app closes too
  if (req.method !== 'POST') return up.end();
  let n = 0; req.on('data', (d) => { n += d.length; if (n > 4096) { up.destroy(); req.destroy(); } }); req.pipe(up);
}
// One line per admin mutation in audit.log (600, rotated at 256 KB to .1): time, ip, route and what it touched by
// NAME: a key's label, a model, a provider, a plan size. Never a value that is a secret (the key itself, a provider
// key, the next admin token, a model layer). Written when the body arrives, so a refused call leaves its trace too.
const AUD = path.join(__dirname, 'audit.log');
const AUD_F = ['name', 'model', 'models', 'action', 'mode', 'tool', 'provider', 'tag', 'bin', 'on', 'edit', 'id', 'from', 'delete', 'quota', 'quota_days', 'quota_day', 'max_concurrent', 'ctx_cap', 'lang', 'reset_usage', 'save', 'force'];
function audit(req, p, b) {
  const w = [];
  if ('key' in b) { const k = (cfg.keys || []).find((x) => x.key === b.key); w.push('key=' + (k ? k.name : b.key ? 'set' : 'none')); }
  AUD_F.forEach((f) => { if (b[f] != null) w.push(f + '=' + String(b[f]).replace(/\s+/g, ' ').slice(0, 60)); });
  try { if (fs.statSync(AUD).size > 262144) fs.renameSync(AUD, AUD + '.1'); } catch (_) {}
  try { fs.appendFileSync(AUD, new Date().toISOString() + ' ' + ipOf(req) + ' ' + p.slice(11) + (w.length ? ' ' + w.join(' ') : '') + '\n', { mode: 0o600 }); } catch (_) {}
}
// Catalog: models are put by hand, one folder each, in chat.models_dir; nothing is uploaded or moved (as Strata's discovery).
const MDIR = () => (cfg.chat || {}).models_dir || path.join(os.homedir(), 'models'), GG = {};
// a GGUF's architecture and trained window, from its KV header up to the tokenizer
function ggufMeta(f) {
  let st; try { st = fs.statSync(f); } catch (_) { return {}; }
  if (GG[f] && GG[f].t === st.mtimeMs) return GG[f].m;
  const b = Buffer.alloc(Math.min(st.size, 4 << 20)), kv = {}, SZ = [1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8];
  let o = 24;
  const str = () => { const n = Number(b.readBigUInt64LE(o)); o += 8 + n; return b.toString('utf8', o - n, o); };
  const val = (t) => { if (t === 8) return str();
    if (t === 9) { const at = b.readUInt32LE(o), n = Number(b.readBigUInt64LE(o + 4)); o += 12; if (at === 8 || at === 9) for (let i = 0; i < n; i++) val(at); else o += n * SZ[at]; return; }
    const v = t === 4 || t === 5 ? b.readUInt32LE(o) : t === 10 ? Number(b.readBigUInt64LE(o)) : 0; o += SZ[t]; return v; };
  try { const fd = fs.openSync(f, 'r'); try { fs.readSync(fd, b, 0, b.length, 0); } finally { fs.closeSync(fd); }
    if (b.toString('latin1', 0, 4) === 'GGUF') for (let i = Number(b.readBigUInt64LE(16)), k; i-- && (k = str()).indexOf('tokenizer.'); ) { o += 4; kv[k] = val(b.readUInt32LE(o - 4)); }
  } catch (_) {}
  const a = kv['general.architecture'] || '';
  return (GG[f] = { t: st.mtimeMs, m: { arch: a, train: +kv[a + '.context_length'] || 0 } }).m;
}
// every weight file one folder down: GGUF (a split one = its first shard) and safetensors (a split one = its index),
// sized with all their shards; used = the entries naming it, text, image or video
function catalogScan() {
  const dir = MDIR(), ms = (cfg.chat || {}).models || {}, use = {}, files = [];
  const real = (f) => { try { return fs.realpathSync(f); } catch (_) { return path.resolve(f); } }, mark = (f, who) => (use[real(f)] = use[real(f)] || []).push(who);
  const ls = (p) => { try { return fs.readdirSync(p).sort(); } catch (_) { return []; } }, base = (x) => x.replace(/-\d+-of-\d+\.(gguf|safetensors)$/i, '');
  ['chat', 'images', 'videos'].forEach((s) => { const m = (cfg[s] || {}).models || {}; Object.keys(m).forEach((id) => (m[id].args || []).forEach((x) => path.isAbsolute(x) && mark(x, id))); });
  // a folder is a model; a variant or a part (IQ3_XXS/, vae/…) may sit one level down, still its folder's
  const scan = (d, sub) => { const at = path.join(dir, d, sub), fl = ls(at); fl.forEach((f) => {
    const sh = /-(\d+)-of-\d+\.(gguf|safetensors)$/i.exec(f), ix = /\.safetensors\.index\.json$/i.test(f), p = path.join(at, f);
    if (!ix && !/\.(gguf|safetensors)$/i.test(f)) { if (!sub) { try { if (fs.statSync(p).isDirectory()) scan(d, f); } catch (_) {} } return; }
    if (sh && (+sh[1] !== 1 || /safetensors$/i.test(f))) return;
    const g = sh ? base(f) : ix && f.replace(/\.safetensors\.index\.json$/i, ''), m = /\.gguf$/i.test(f) ? ggufMeta(p) : {};
    const gb = (g ? fl.filter((x) => /\.(gguf|safetensors)$/i.test(x) && base(x) === g) : [f]).reduce((s, x) => { try { return s + fs.statSync(path.join(at, x)).size; } catch (_) { return s; } }, 0);
    files.push({ dir: d, file: sub ? sub + '/' + f : f, path: p, gb: +(gb / 1e9).toFixed(1), train: m.train || 0, used: use[real(p)] || [],
      kind: m.arch === 'clip' || /mmproj/i.test(f) ? 'eyes' : /(^|[-_.])(mtp|draft|dflash)/i.test(f) ? 'draft' : 'model' });
  }); };
  ls(dir).forEach((d) => scan(d, ''));
  // presets: 53ON's own calibration per model file (catalog.json), filled in when a matching .gguf is added
  let presets = {}; try { presets = JSON.parse(fs.readFileSync(path.join(__dirname, 'catalog.json'), 'utf8')); } catch (_) {}
  // all text models: a stopped one is editable
  return { dir: dir, real: real(dir), files: files, presets: presets, text: ms, image: (cfg.images || {}).models || {}, video: (cfg.videos || {}).models || {}, cloud: (cfg.cloud || {}).catalog || {} };
}
// a kind is what a model makes (OpenRouter's output modality) and the config section that holds it; video has no engine yet
const KIND = { text: 'chat', image: 'images', video: 'videos' };
// uncensored: the owner's word (the editor's 🔓); never set, the model's name, files and note tell
const unc = (e, name) => e.uncensored ?? /ablit|uncens|uncnr|unfilter|heretic|dolphin|nsfw/i.test(name + ' ' + (e.args || []).join(' ') + ' ' + (e.note || ''));
const kindOn = (k, n) => k === 'text' ? (cfg.on || []).includes(n) || !!(chat.list().find((r) => r.name === n) || {}).state
  : k === 'image' ? !!(images.list().find((r) => r.id === n) || {}).state : k === 'video' && !!(videos.list().find((r) => r.id === n) || {}).state;
function handleAdmin(req, res, p) {
  const tok = req.headers['x-admin-token'] || (req.headers['authorization'] || '').replace(/^Bearer /, '');
  if (!cfg.admin_token || !same(tok, cfg.admin_token)) return jerr(res, 401, 'Invalid admin token.', 'auth');
  const body = (cb) => readBody(req, (b) => { audit(req, p, b); cb(b); });
  if (p === '/admin/api/pulse' && req.method === 'GET') {
    const oks = PULSE.filter(e => e.ok && e.tps);
    return jres(res, 200, {
      list: PULSE,
      tps_avg: oks.length ? Math.round(oks.reduce((a, e) => a + e.tps, 0) / oks.length * 10) / 10 : null,
      queue_max: PULSE.length ? Math.max.apply(null, PULSE.map(e => e.queue || 0)) : 0,
      errors: PULSE.filter(e => !e.ok).length,
      stranded: PULSE.reduce((a, e) => a + (e.str || 0), 0),   // O: tool calls written inside a <think> that never closed
      permits: permits(),
    });
  }
  if (p === '/admin/api/usage' && req.method === 'GET') {
    const keys = (cfg.keys || []).map(k => {
      const u = usageOf(k.key);
      const kr = KEYS.get(k.key);
      return { key: k.key, name: k.name, quota: +k.quota || 0, quota_days: +k.quota_days || 0, quota_day: +k.quota_day || 0, day: dayOf(k.key),
               quota_at: k.quota_at || '', expires: expiresAt(kr), expired: !!(kr && blockReason(k.key, kr) || '').toString().startsWith('Your plan expired'),
               in: u.in || 0, out: u.out || 0, req: u.req || 0, last: u.last || '' };
    });
    return jres(res, 200, { keys: keys, models: USE.models, days: USE.days, today: USE.days[today()] || { in: 0, out: 0, req: 0 } });
  }
  // every 2 s for the dock: what is on its way up (phase, since), the line and the pictures pending
  if (p === '/admin/api/metrics' && req.method === 'GET') return sample((s) => jres(res, 200, Object.assign({}, s, { engine_up: chat.up(), models: chat.loaded(), image: images.state(),
    wake: chat.list().filter((m) => m.state === 'loading').map((m) => [m.name, m.ph, m.since]), queued: queued(), jobs: images.jobs() })));
  if (p === '/admin/api/history' && req.method === 'GET') return jres(res, 200, { ms: HIST_MS, h: hist });
  if (p === '/admin/api/history' && req.method === 'DELETE') { audit(req, p, {}); hist = []; histW = 0; fs.unlink(HIST_F, () => {}); return jres(res, 200, { ok: true }); }
  // the accumulated usage (per key, model and day) and the pulse start over; keys, plans and config stay
  if (p === '/admin/api/usage' && req.method === 'DELETE') { audit(req, p, {}); USE = { keys: {}, models: {}, days: {} }; writeUsage(); PULSE.length = 0; return jres(res, 200, { ok: true }); }
  if (p === '/admin/api/state' && req.method === 'GET') return jres(res, 200, {
    on: selected(), free: mem.free(), version: VERSION, repo: REPO,
    // port is the street door (tls); lan_port the house one, plain http, so the panel names it right whichever it came through
    public_url: cfg.public_url || '', port: cfg.port, lan_port: server.setSecureContext ? cfg.port + 1 : cfg.port,
    lan: lanIPs(), max_concurrent: MAXC, engine: (EN().name || 'engine') + ' ' + chat.version(), engines: engines(), engine_job: ENG, engine_src: EN(), engine_dir: ENGD,
    engine_up: chat.up(), inflight: inFlight(), queued: queued(), permits: permits(), max_body_mb: MAXBODY / 1048576,
    keys: (cfg.keys || []).map(k => ({ name: k.name, key: k.key, image: imgFlag(k.image) })),
    // title and note from the model's own entry: the server's catalog is its config
    downloaded: chat.list().map(m => { const e = cfg.chat.models[m.name] || {};
      return Object.assign(m, { ttl: e.ttl || '', note: e.note || '', uncensored: unc(e, m.name + ' ' + m.gguf) });
    // cloud and Claude rows share this shape, so the panel lists them with no extra code; off, they come with state 'off'
    }).concat([panelModel()], cloud.panelModels(CLOUD)),
    loaded: chat.loaded(),
    // local image models (sd-server spawned on demand) with live state; video ones are catalog only, no engine yet
    images: images.list().map((m) => Object.assign(m, { uncensored: unc(((cfg.images || {}).models || {})[m.id] || {}, m.id) })), videos: videos.list().map((m) => Object.assign(m, { uncensored: unc(((cfg.videos || {}).models || {})[m.id] || {}, m.id) })),
    ctx_cap: ctxCap(), rates: Object.keys(prices.table()).length,
    // the drawer paired with the main model, normalized: same object as GET /tools
    tools: { image: cloud.toolCfg(CLOUD, 'image') }
  });
  // The model's own layer: what its chat template wraps around the System Prompt and the tools. Nothing hidden.
  if (p === '/admin/api/template' && req.method === 'GET') {
    const n = (chat.loaded()[0] || {}).name, t = n && chat.target(n);
    if (!t) return jerr(res, 503, 'No local model loaded.', 'model');
    const port = +new URL(t.baseUrl).port, eff = new URL(req.url, 'http://x').searchParams.get('effort') || 'low';
    const post = (path, body, cb) => { const r = http.request({ host: '127.0.0.1', port, path, method: 'POST', headers: { 'content-type': 'application/json' }, timeout: 10000 }, (rs) => { let d = ''; rs.on('data', (c) => { d += c; }); rs.on('end', () => { try { cb(null, JSON.parse(d)); } catch (e) { cb(e); } }); }); r.on('error', cb); r.on('timeout', () => r.destroy(new Error('timeout'))); r.end(JSON.stringify(body)); };
    const ly = chat.layer(n);
    return post('/apply-template', { reasoning_effort: eff, messages: [{ role: 'system', content: (ly ? ly + '\n\n' : '') + '{YOUR SYSTEM PROMPT}' }, { role: 'user', content: '{USER}' }],
      tools: [{ type: 'function', function: { name: '{tool}', description: '{what it does}', parameters: { type: 'object', properties: { '{arg}': { type: 'string' } } } } }] }, (e, a) => {
      if (e || !a || !a.prompt) { t.done(); return jerr(res, 502, 'The model did not render its template.', 'model'); }
      post('/tokenize', { content: a.prompt }, (e2, k) => { t.done(); jres(res, 200, { model: n, effort: eff, sys: ly, tokens: ((k && k.tokens) || []).length, prompt: a.prompt }); });
    });
  }
  // the layer a local model carries: staff writes it by hand, empty = none
  if (p === '/admin/api/model/sys' && req.method === 'POST') return body((b) => {
    const m = cfg.chat && cfg.chat.models && cfg.chat.models[b.id];
    if (!m) return jerr(res, 404, 'Unknown model.', 'model');
    if (String(b.sys || '').trim()) m.sys = String(b.sys); else delete m.sys;
    saveConfig(); jres(res, 200, { ok: true, id: b.id, sys: m.sys || '' });
  });
  // the catalog: the models folder and its entries (✏️ and + of Modelos)
  if (p === '/admin/api/catalog' && req.method === 'GET') return jres(res, 200, catalogScan());
  // {type, id, entry, from?, from_type?} saves, {type, id, delete:true} drops. A serving model is never renamed, moved or
  // dropped, an answer is never cut, and every absolute path in args must exist.
  if (p === '/admin/api/catalog' && req.method === 'POST') return body((b) => {
    const ty = b.type || 'text', fty = b.from_type || ty, id = String(b.id || '').trim(), from = String(b.from || id).trim(), e = b.entry, a = e && e.args;
    // the server's cloud catalog, whole: {providers: {id: {label, baseUrl, defaultCtx, ctx: {model: n}, ...}}, ctx: {model: n}}
    if (ty === 'cloud') { const k = b.catalog, ob = (o) => !!o && typeof o === 'object' && !Array.isArray(o);
      if (!ob(k) || !ob(k.providers) || !Object.values(k.providers).every(ob)) return jerr(res, 400, 'The catalog needs its providers.', 'invalid', 'catalog');
      (cfg.cloud = cfg.cloud || {}).catalog = { providers: k.providers, ctx: ob(k.ctx) ? k.ctx : {} }; saveConfig(); applyCloud(); return jres(res, 200, { ok: true, id: 'cloud' }); }
    if (!KIND[ty] || !KIND[fty]) return jerr(res, 400, "type must be 'text', 'image' or 'video'.", 'invalid', 'type');
    if (!(ty === 'image' ? /^local\/[\w.-]{1,64}$/ : /^[\w.-]{1,64}$/).test(id)) return jerr(res, 400, 'A name takes letters, digits, dots, dashes or underscores' + (ty === 'image' ? ', after local/.' : '.'), 'invalid', 'id');
    const sec = (k) => { const s = cfg[KIND[k]] = cfg[KIND[k]] || {}; return s.models = s.models || {}; }, ms = sec(ty), fms = sec(fty), moved = from !== id || fty !== ty;
    const off = (n) => jerr(res, 409, n + ' is serving. Stop it first.', 'model_busy', 'id');
    if (b.delete) { if (!ms[id]) return jerr(res, 404, 'Unknown model.', 'not_found', 'id'); if (kindOn(ty, id)) return off(id); delete ms[id]; }
    else {
      if (!e || typeof e !== 'object' || !Array.isArray(a) || !a.every((x) => typeof x === 'string')) return jerr(res, 400, 'args must be a list of strings.', 'invalid', 'args');
      const lost = a.find((x) => path.isAbsolute(x) && !fs.existsSync(x)); if (lost) return jerr(res, 400, 'No file at ' + lost + '.', 'invalid', 'args');
      if (ty === 'text' && e.engine !== 'strata' && !a.some((x, i) => /^(-m|--model)$/.test(x) && a[i + 1])) return jerr(res, 400, 'A text model needs -m <file.gguf>.', 'invalid', 'args');
      if ((b.from == null || moved) && ms[id]) return jerr(res, 409, id + ' already exists.', 'exists', 'id');
      if (moved && kindOn(fty, from)) return off(from);
      if (kindOn(fty, from)) return off(from);   // edit only while stopped
      if ((Q['local:' + from] || {}).n) return jerr(res, 409, 'An answer is running on ' + from + '. Save when it ends.', 'model_busy', 'id');
      // the entry keeps its place in its list
      const nm = {}; if (fty !== ty) delete fms[from];
      Object.keys(ms).forEach((k) => (fty === ty && k === from ? (nm[id] = e) : (nm[k] = ms[k]))); nm[id] = e; cfg[KIND[ty]].models = nm;
    }
    saveConfig(); if ([ty, fty].includes('text')) chat.options(cfg.chat, ctxCap()); if ([ty, fty].includes('image')) images.options(cfg.images); if ([ty, fty].includes('video')) videos.options(Object.assign({ bin: (cfg.images || {}).bin || '' }, cfg.videos));
    jres(res, 200, { ok: true, id: id });
  });
  // Gateway tools: what a text model can delegate, and to which model.
  if (p === '/admin/api/tools' && req.method === 'GET')
    return jres(res, 200, { tools: cloud.toolList().map((x) => Object.assign(x, cloud.toolCfg(CLOUD, x.id))) });
  // Stored as {on, model, edit}; old panels still send {provider, model:bare id}. Old shapes migrate here.
  if (p === '/admin/api/tools' && req.method === 'POST') return body((b) => {
    const id = String(b.tool || '').trim();
    if (!cloud.toolList().some((x) => x.id === id)) return jerr(res, 400, 'Unknown tool.', 'invalid', 'tool');
    cfg.cloud = cfg.cloud || {}; cfg.cloud.tools = cfg.cloud.tools || {};
    const cur = cloud.toolCfg(cfg.cloud, id);
    // a local image model for its task, or a cloud image model (they create and edit) while the cloud is on
    const fits = (m, task) => images.can(m, task) || cloud.isImageModel(CLOUD, m);
    if (b.on != null) cur.on = !!b.on;
    // only a new model is validated: turning off must work even if the stored one went stale
    if (b.model != null) {
      const m = String(b.model).trim(); cur.model = m && b.provider ? b.provider + '/' + m : m;
      if (m && !fits(cur.model, 'create')) return jerr(res, 400, 'Not an image model.', 'invalid', 'model');
    }
    if (b.edit != null) {
      cur.edit = String(b.edit).trim();
      if (cur.edit && !fits(cur.edit, 'edit')) return jerr(res, 400, 'Not an image editing model.', 'invalid', 'edit');
    }
    cfg.cloud.tools[id] = cur; if (id === 'image') delete cfg.cloud.image_model;
    saveConfig(); applyCloud();
    jres(res, 200, Object.assign({ ok: true, id: id }, cur));
  });
  // Cloud providers from the catalog. A provider key is never returned, only whether one is set.
  if (p === '/admin/api/cloud' && req.method === 'GET') {
    const cat = cloud.catalog(), hp = cloud.health();
    const list = Object.keys(cat || {}).map((k) => {
      const c = CLOUD.providers[k] || {}, h = hp[k];
      return { id: k, label: (cat[k].label || k), baseUrl: c.baseUrl || cat[k].baseUrl || '',
               editable: !!cat[k].editable, keyUrl: cat[k].keyUrl || '',
               hasKey: !!c.key, on: c.enabled !== false && !!c.key, models: c.models || [],
               state: h ? h.state : '', code: h ? h.code : 0, msg: h ? h.msg : '',
               at: h ? h.at : 0, limit: h ? h.limit : {},
               nModels: (Array.isArray(c.models) && c.models.length) || cloud.modelCount(k) };
    });
    return jres(res, 200, { providers: list });
  }
  // Probe one provider, only on the owner's request (their keys, their balance).
  if (p === '/admin/api/cloud/probe' && req.method === 'POST') return readBody(req, (b) => {
    const id = String(b.provider || '').trim();
    if (!id || !CLOUD.providers[id]) return jerr(res, 400, 'Unknown provider.', 'invalid', 'provider');
    cloud.probe(CLOUD, id, (h) => jres(res, 200, Object.assign({ ok: true, id: id }, h)));
  });
  if (p === '/admin/api/cloud' && req.method === 'POST') return body((b) => {
    const id = String(b.provider || '').trim();
    if (!id) return jerr(res, 400, "Missing required parameter: 'provider'.", 'invalid', 'provider');
    cfg.cloud = cfg.cloud || { enabled: true, providers: {} };
    cfg.cloud.providers = cfg.cloud.providers || {};
    const cur = cfg.cloud.providers[id] || {};
    // an empty key deletes the stored credential
    if (b.key != null) { const k = String(b.key).trim(); if (k) cur.key = k; else delete cur.key; }
    if (b.baseUrl != null) cur.baseUrl = String(b.baseUrl).trim();
    if (b.on != null) cur.enabled = !!b.on;
    if (Array.isArray(b.models)) cur.models = b.models;
    cfg.cloud.providers[id] = cur;
    saveConfig(); applyCloud();
    jres(res, 200, { ok: true, id: id, hasKey: !!cur.key });
  });
  // {model, action:'load'|'unload'} or {model, on}: selects a model or takes it out. Selected = served.
  if (p === '/admin/api/model' && req.method === 'POST') return body((b) => {
    const name = String(b.model || '').trim(), act = b.action;
    if (!name) return jerr(res, 400, "Missing required parameter: 'model'.", 'invalid', 'model');
    if (act != null && act !== 'load' && act !== 'unload') return jerr(res, 400, "action must be 'load' or 'unload'.", 'invalid', 'action');
    const on = act ? act === 'load' : b.on !== false, free = mem.free();
    // a play asks first: the catalog says what it takes, mem.js what is free (unread memory refuses nothing)
    const short = (gb, up) => on && !up && gb > free, noRoom = (gb) => jerr(res, 409, name + ' needs ' + gb + ' GB, ' + free + ' GB free. Stop a model first.', 'no_memory', 'model');
    // video model: the image model's twin, its sd-server starts in the background or is killed now, never under a clip
    if (videos.has(name)) {
      if (videos.jobs()) return jerr(res, 409, 'A video is being made. Load or unload a video model when it ends.', 'video_busy', 'model');
      const vm = videos.list().find((x) => x.id === name) || {};
      if (short(vm.gb, vm.state)) return noRoom(vm.gb);
      if (on) videos.load(name); else videos.unload();
      return jres(res, 200, { ok: true, video: videos.state() });
    }
    // image model: sd-server starts in the background (the panel polls /state) or is killed now;
    // never under a pending picture, which the kill would fail
    if (images.has(name)) {
      if (images.jobs()) return jerr(res, 409, 'An image is being generated. Load or unload an image model when it ends.', 'image_busy', 'model');
      const im = images.list().find((x) => x.id === name) || {};
      if (short(im.gb, im.state)) return noRoom(im.gb);
      if (on) images.load(name); else images.unload();
      return jres(res, 200, { ok: true, image: images.state() });
    }
    const pick = (m) => { if ((cfg.on || []).includes(m) !== on) { cfg.on = (cfg.on || []).filter((x) => x !== m).concat(on ? [m] : []); saveConfig(); pinLocks(); } }, ok = () => jres(res, 200, { ok: true, on: selected() });
    // Claude and cloud models have no VRAM to load. One turned off (claude.enabled false) can be taken out, never selected.
    const sky = isClaude(name) || isCloudId(name), px = name.slice(0, Math.max(0, name.indexOf('/')));
    if (sky || name === CLA.id || (px && ((cfg.cloud || {}).providers || {})[px])) {
      if (on && !sky) return jerr(res, 409, name + ' is turned off.', 'model_off', 'model');
      pick(name); return ok();
    }
    const n = chat.id(name);
    if (!chat.has(n)) return jerr(res, 404, "Unknown model '" + name + "'.", 'not_found', 'model');
    // like a picture (409 image_busy) and Strata's unload ('busy'): a click never cuts an answer; force:true says it may
    const bz = on ? 0 : (Q['local:' + n] || {}).n || 0;
    if (bz && !b.force) return jerr(res, 409, bz + ' answer(s) running would be cut. Try again when they end, or send force:true.', 'model_busy', 'model');
    const row = chat.list().find((r) => r.name === n);
    if (short(row.need, row.state)) return noRoom(row.need);
    pick(n);
    if (!on) { chat.unload(n, false, null, 'owner'); return ok(); }
    chat.load(n, true, (e) => e ? jerr(res, 502, 'llama-server: ' + e.message, 'upstream') : ok());
  });
  if (p === '/admin/api/model/engine' && req.method === 'POST') return body((b) => {
    const name = chat.id(String(b.model || '')), eng = String(b.engine || '');
    if (!chat.has(name)) return jerr(res, 404, "Unknown model '" + name + "'.", 'not_found', 'model');
    if (eng !== 'strata' && eng !== 'llama.cpp') return jerr(res, 400, "engine must be strata or llama.cpp.", 'invalid', 'engine');
    const m = (cfg.chat.models || {})[name]; if (!m) return jerr(res, 404, "no config", 'not_found', 'model');
    m.engine = eng; saveConfig(); chat.options(cfg.chat, ctxCap());
    chat.unload(name, false, null, 'engine'); if ((cfg.on || []).includes(name)) chat.load(name, true, () => {});
    jres(res, 200, { ok: true, model: name, engine: eng });
  });
  if (p === '/admin/api/key' && req.method === 'POST') return body((b) => {
    const key = genKey(); cfg.keys = cfg.keys || [];
    cfg.keys.push({ key: key, name: String(b.name || 'unnamed').slice(0, 40),
                    image: imgFlag(b.image),
                    quota: Math.max(0, +b.quota || 0), quota_days: Math.max(0, +b.quota_days || 0), quota_day: Math.max(0, +b.quota_day || 0),
                    quota_at: new Date().toISOString() });   // the plan period starts now
    applyKeys(); saveConfig(); jres(res, 200, { ok: true, key: key });
  });
  if (p === '/admin/api/key/update' && req.method === 'POST') return body((b) => {
    const k = (cfg.keys || []).find(x => x.key === b.key); if (k) { if ('image' in b) k.image = imgFlag(b.image); if (b.name) k.name = String(b.name).slice(0, 40); if ('quota' in b) { k.quota = Math.max(0, +b.quota || 0); k.quota_at = new Date().toISOString(); }
      if ('quota_days' in b) { k.quota_days = Math.max(0, +b.quota_days || 0); if (!k.quota_at) k.quota_at = new Date().toISOString(); }
      if ('quota_day' in b) k.quota_day = Math.max(0, +b.quota_day || 0); }   // a daily limit never restarts the plan
    // reset_usage restores the full quota without deleting the key
    if (b.reset_usage && USE.keys[b.key]) { USE.keys[b.key] = { in: 0, out: 0, req: 0, last: '' }; saveUsage(); }
    applyKeys(); saveConfig(); jres(res, 200, { ok: true });
  });
  if (p === '/admin/api/key/delete' && req.method === 'POST') return body((b) => {
    cfg.keys = (cfg.keys || []).filter(k => k.key !== b.key); applyKeys(); saveConfig(); jres(res, 200, { ok: true });
  });
  // a new secret for an existing key: name, plan and usage (keyed by the secret) move with it; the old value dies now
  if (p === '/admin/api/key/rotate' && req.method === 'POST') return body((b) => {
    const k = (cfg.keys || []).find(x => x.key === b.key); if (!k) return jerr(res, 404, 'Unknown key.', 'not_found', 'key');
    const n = genKey(); if (USE.keys[k.key]) { USE.keys[n] = USE.keys[k.key]; delete USE.keys[k.key]; saveUsage(); }
    k.key = n; applyKeys(); saveConfig(); jres(res, 200, { ok: true, key: n, name: k.name });
  });
  // the users in one file, out and back. Import never removes: a new secret comes with its usage, a known one takes the
  // file's name and plan and keeps its counters. The file holds the secrets, as the panel does; the audit line does not.
  if (p === '/admin/api/keys/export' && req.method === 'GET') {
    audit(req, p, {});
    return jres(res, 200, { ia_server_keys: 1, at: new Date().toISOString(), keys: (cfg.keys || []).map((k) => Object.assign({}, k, { usage: usageOf(k.key) })) });
  }
  if (p === '/admin/api/keys/import' && req.method === 'POST') return body((b) => {
    if (b.ia_server_keys !== 1 || !Array.isArray(b.keys)) return jerr(res, 400, 'Not an ia_server keys file.', 'invalid_request');
    if (b.keys.some((k) => !k || typeof k.key !== 'string' || !/^[\x21-\x7e]{8,128}$/.test(k.key))) return jerr(res, 400, 'A key in the file is not valid.', 'invalid_request', 'keys');
    let added = 0, updated = 0; cfg.keys = cfg.keys || [];
    for (const k of b.keys) {
      const row = { key: k.key, name: String(k.name || 'unnamed').slice(0, 40), image: imgFlag(k.image), quota: Math.max(0, +k.quota || 0),
                    quota_days: Math.max(0, +k.quota_days || 0), quota_day: Math.max(0, +k.quota_day || 0), quota_at: String(k.quota_at || '') };
      const had = cfg.keys.find((x) => x.key === k.key);
      if (had) { Object.assign(had, row); updated++; continue; }
      cfg.keys.push(row); added++;
      if (k.usage && typeof k.usage === 'object' && !Object.prototype.hasOwnProperty.call(USE.keys, k.key)) USE.keys[k.key] = k.usage;
    }
    applyKeys(); saveConfig(); saveUsage(); jres(res, 200, { ok: true, added: added, updated: updated });
  });
  // the panel's own token: {next} chosen (printable ASCII, so it fits a header) or {} generated. Answered once, never
  // logged; the old one is dead as soon as the file holds the new one (ops/ias.py reads config.json on every run)
  if (p === '/admin/api/admin/token' && req.method === 'POST') return body((b) => {
    const t = b.next == null ? 'adm-' + crypto.randomBytes(12).toString('hex') : String(b.next), old = cfg.admin_token;
    if (!/^[!-~]{16,64}$/.test(t)) return jerr(res, 400, 'Token: 16-64 characters, no spaces.', 'invalid', 'next');
    cfg.admin_token = t; if (!saveConfig()) { cfg.admin_token = old; return jerr(res, 500, 'config.json not written.', 'file'); }
    jres(res, 200, { ok: true, token: t });
  });
  // {tag}: an installed or published version, latest (nightly) or stable, through ops/update_engine.sh, one at a
  // time (argv, no shell: the name only needs to be a plain word); {bin}: the script's live swap
  if (p === '/admin/api/engine' && req.method === 'POST') return body((b) => {
    if (b.bin) {
      const ch = cfg.chat = cfg.chat || {};
      let f = ''; try { f = fs.realpathSync(String(b.bin)); } catch (_) {}
      if (!ch.bin || path.basename(f) !== path.basename(ch.bin) || f.indexOf(ENGD + path.sep)) return jerr(res, 400, 'bin must be the same engine binary under ' + ENGD + '.', 'invalid');
      // the swap cuts whatever the old build is serving: it happens only in a quiet moment (the script retries)
      if (f !== ch.bin && (inFlight() || queued())) return jerr(res, 409, 'Requests are running; try again.', 'busy');
      // restart what runs on the new build and bring back every pinned model, even one that died on the other build
      ch.bin = f; chat.options(ch, ctxCap()); pinLocks();
      // two phases: applied now, written once the script has verified it, so a restart meanwhile boots the previous one
      if (b.save) saveConfig();
      return jres(res, 200, { ok: true, engine: chat.version() });
    }
    const t = String(b.tag || '');
    if (!/^\w[\w.-]{0,39}$/.test(t)) return jerr(res, 400, 'tag must be a version name, latest or stable.', 'invalid');
    if (ENG && ENG.state === 'run') return jerr(res, 409, 'An engine update is already running.', 'busy');
    const e = ENG = { tag: t, state: 'run', line: 'starting' };
    const out = (d) => { const l = String(d).trim().split('\n').pop(); if (l) e.line = l.slice(0, 160); };
    let j;
    try {
      j = spawn('bash', [path.join(__dirname, 'ops', 'update_engine.sh'), t], { timeout: 1800000,
        env: Object.assign({}, process.env, { IA_BASE: 'http://127.0.0.1:' + (server.setSecureContext ? cfg.port + 1 : cfg.port) }) });
    } catch (x) { e.state = 'failed'; e.line = x.message; return jres(res, 202, e); }
    if (j.stdout) j.stdout.on('data', out); if (j.stderr) j.stderr.on('data', out);
    j.on('error', (x) => { e.state = 'failed'; e.line = x.message; });
    // only a clean exit is a success: a signal (timeout, OOM, kill) leaves code null
    j.on('close', (c) => { if (e.state === 'run') e.state = c === 0 ? 'ok' : 'failed'; });
    jres(res, 202, e);
  });
  if (p === '/admin/api/config' && req.method === 'POST') return body((b) => {
    // the panel's language: one that has its lang/<code>.json, as the 53ON Harness's Options take it
    if ('lang' in b) { if (!i18n.lang(b.lang)) return jerr(res, 400, 'Unknown language.', 'invalid', 'lang'); cfg.lang = b.lang; }
    if ('public_url' in b) cfg.public_url = String(b.public_url || '').replace(/\/+$/, '');
    if ('max_concurrent' in b) MAXC = Math.max(1, +b.max_concurrent || 1);
    // a new cap restarts the running chat models with it
    if ('ctx_cap' in b) { cfg.ctx_cap = Math.max(1024, +b.ctx_cap || 32768); chat.options(cfg.chat, ctxCap()); }
    saveConfig(); jres(res, 200, { ok: true, public_url: cfg.public_url, max_concurrent: MAXC, ctx_cap: ctxCap(), lang: i18n.lang() });
  });
  return jerr(res, 404, 'Unknown admin endpoint.', 'not_found');
}

// Two doors, on purpose. The street gets TLS on cfg.port: the api key never travels in the clear.
// The house gets plain http on the next port, because a CA only signs names, never a LAN ip, so
// https at home would just mean a certificate warning on every phone. Missing or unreadable certs
// leave everything on plain http instead of dying: a cert problem never takes the service down.
const _mkServer = (handler) => {
  const dir = process.env.IA_TLS_DIR || path.join(__dirname, 'certs');
  const cf = path.join(dir, 'fullchain.pem'), kf = path.join(dir, 'privkey.pem');
  // idle connections kept 90 s (Node's 5 s made a client redo TLS every turn); headersTimeout above it
  const door = (s) => { s.keepAliveTimeout = 90000; s.headersTimeout = 95000; return s; };
  try {
    const srv = door(https.createServer({ cert: fs.readFileSync(cf), key: fs.readFileSync(kf) }, handler));
    // A certificate renewer signals SIGHUP after it drops a renewed pair here. A sleeping handler:
    // costs nothing until the signal lands, then swaps the cert live, no restart.
    process.on('SIGHUP', () => {
      try { srv.setSecureContext({ cert: fs.readFileSync(cf), key: fs.readFileSync(kf) }); console.log('[ia_server] TLS cert reloaded (SIGHUP)'); }
      catch (e) { console.log('[ia_server] TLS reload skipped: ' + e.message); }
    });
    door(http.createServer(handler)).listen(cfg.port + 1, '0.0.0.0');
    console.log('[ia_server] TLS on (https) from ' + dir);
    return srv;
  } catch (e) {
    console.log('[ia_server] TLS off, plain http: ' + e.message);
    return door(http.createServer(handler));
  }
};
const server = _mkServer((req, res) => {
  const p = pathOf(req.url);
  // CORS: HTTPS browser clients must be allowed to send Authorization
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', String(req.headers['access-control-request-headers'] || 'Authorization, Content-Type, x-api-key, anthropic-version, anthropic-beta').slice(0, 1000));   // what the browser asks for (SDKs add their own)
  res.setHeader('Access-Control-Max-Age', '86400');
  if (req.method === 'OPTIONS') { res.writeHead(204); return res.end(); }
  // generated pictures: public like a provider's image URL, unguessable names, served 24 h from creation at most
  // (shared caches keep none: they can be edits of real faces)
  // generated clips: the pictures' rules, mp4 (a webm from before it), streamed
  const vid = req.method === 'GET' && /^\/vid\/([0-9a-f]{32})\.(mp4|webm)$/.exec(p);
  if (vid) { const f = path.join(videos.DIR, vid[1] + '.' + vid[2]), m = videos.making(vid[1]);
    // a clip still being made answers 202, a failed one 410 with why: a client that left polls its link until it is there
    if (m === '') { res.writeHead(202, { 'content-type': 'application/json', 'retry-after': '15', 'cache-control': 'no-store' }); return res.end('{"status":"in_progress"}'); }
    if (m) return jres(res, 410, { error: { message: 'Video generation failed: ' + m, type: 'api_error', param: null, code: 'video_failed' } });
    return fs.stat(f, (e, st) => {
    const left = e ? 0 : videos.DAY - (Date.now() - st.mtimeMs);
    if (left <= 0) { if (!e) fs.unlink(f, () => {}); return jres(res, 404, { error: { message: 'Not found.', type: 'invalid_request_error', param: null, code: 'not_found' } }); }
    // byte ranges (206): Safari plays a video only from a server that serves them, and players seek with them
    const r = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range || ''), n = st.size;
    let a = 0, b = n - 1;
    if (r) { a = r[1] ? +r[1] : n - +r[2]; if (r[1] && r[2]) b = Math.min(+r[2], n - 1); }
    if (r && (a < 0 || a > b)) { res.writeHead(416, { 'content-range': 'bytes */' + n }); return res.end(); }
    res.writeHead(r ? 206 : 200, Object.assign({ 'content-type': 'video/' + vid[2], 'content-length': b - a + 1, 'accept-ranges': 'bytes', 'cache-control': 'private, max-age=' + Math.floor(left / 1000) + ', immutable' },
      r ? { 'content-range': 'bytes ' + a + '-' + b + '/' + n } : {}));
    fs.createReadStream(f, { start: a, end: b }).pipe(res);
  }); }
  const img = req.method === 'GET' && /^\/img\/([0-9a-f]{32})\.png$/.exec(p);
  if (img) { const f = path.join(images.DIR, img[1] + '.png'); return fs.stat(f, (e, st) => {
    const left = e ? 0 : images.DAY - (Date.now() - st.mtimeMs);
    if (left <= 0) { if (!e) fs.unlink(f, () => {}); return jres(res, 404, { error: { message: 'Not found.', type: 'invalid_request_error', param: null, code: 'not_found' } }); }
    fs.readFile(f, (e2, buf) => {
      if (e2) return jres(res, 404, { error: { message: 'Not found.', type: 'invalid_request_error', param: null, code: 'not_found' } });
      res.writeHead(200, { 'content-type': 'image/png', 'content-length': buf.length, 'cache-control': 'private, max-age=' + Math.floor(left / 1000) + ', immutable' }); res.end(buf);
    });
  }); }

  // public_url = the https door: a client that reached the http one (an app on the LAN, where http is mixed content)
  // streams there instead of waiting for the whole reply
  if (p === '/health' || p === '/') return jres(res, 200, { ok: true, service: 'ia_server', max_concurrent: MAXC, admin: '/admin', meta: '/meta', public_url: cfg.public_url || undefined });
  // 404, not 403: an outsider is not even told there is something here to attack
  // the owner's admin apps (config.json "apps"): the house network only, like /admin, then appDoor's own lock
  const app = appOf(p); if (app) return fromNet(req) ? jerr(res, 404, 'Not found.', 'not_found') : appDoor(req, res, p, app);
  if (p.indexOf('/admin') === 0 && fromNet(req)) return jerr(res, 404, 'Not found.', 'not_found');
  // the panel with its words: i18n.js hands it the bag of its surface (ui.), as the 53ON Harness does with its pages
  // the panel's own files, cached a week: 53ON's logo (128 px) and the house help icon (Onboarding's help.svg)
  const pub = /^\/admin\/(logo\.png|help\.svg)$/.exec(p);
  if (pub) return fs.readFile(path.join(__dirname, pub[1]), (e, b) => {
    if (e) return jerr(res, 404, pub[1] + ' not found.', 'file');
    res.writeHead(200, { 'content-type': pub[1].endsWith('.png') ? 'image/png' : 'image/svg+xml', 'cache-control': 'max-age=604800' }); res.end(b); });
  if (p === '/admin' || p === '/admin/') return fs.readFile(path.join(__dirname, 'admin.html'), 'utf8', (e, h) => { if (e) return jerr(res, 500, 'admin.html not found.', 'file');
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store' });
    res.end(h.replace('/*i18n*/{}', () => JSON.stringify({ lang: i18n.lang(), langs: i18n.langs(), t: i18n.bag('ui') }).replace(/</g, '\\u003c'))); });
  if (p.indexOf('/admin/api') === 0) return handleAdmin(req, res, p);

  // browser noise: silent 404, kept out of the pulse
  if (BROWSER_NOISE.test(p)) return jres(res, 404, { error: { message: 'Not found.', type: 'invalid_request_error', param: null, code: 'not_found' } });
  // Anthropic's door (Claude Code): translated onto the OpenAI door by loopback, so key, quota, brake and pulse are the same
  if (req.method === 'POST' && /^\/v1\/messages(\/count_tokens)?\/?$/.test(p)) return anthropic.serve(req, res, p, { port: server instanceof https.Server ? cfg.port + 1 : cfg.port, max: MAXBODY, ok: (k) => !!KEYS.get(k), key: clientKey(req) });
  const apiKey = clientKey(req), krec = KEYS.get(apiKey);
  if (!krec) return jerr(res, 401, 'Invalid or missing API key.', 'auth');
  res._rl = rateHeaders(krec, apiKey);   // every later response carries them
  if (p === '/meta') {
    const ms = served();
    return jres(res, 200, { service: 'ia_server', max_concurrent: MAXC, transmit_standard: TRANSMIT_STANDARD, ctx_field: CTX_FIELD,
      models: ms.map(m => ({ id: m.id, context_length: m.context_length, max_context_length: m.max_context_length, context_length_source: m.context_length_source, capabilities: m.capabilities })) });
  }
  if (!allowed(req)) return jerr(res, 403, 'Route not allowed. Only inference and model listing are available.', 'forbidden');
  // Quota is checked before the GPU starts and fails with a clear error. max_tokens is never trimmed to
  // fit: reasoning models spend a small budget thinking and return empty text without an error.
  if (isInfer(req)) {
    const _m = blockReason(apiKey, krec);
    if (_m) return jerr(res, 429, _m, 'insufficient_quota');
  }
  if (req.method === 'GET' && /^\/v1\/models\/?$/.test(p)) return handleModels(res);

  // Inference: buffer body -> model per key -> queue -> engine
  if (isInfer(req)) {
    // The body is counted while it arrives and cut at MAXBODY, before the queue: concurrent uploads
    // (base64 images) must not exhaust memory.
    const chunks = []; let received = 0, tooBig = false;
    req.on('data', d => {
      if (tooBig) return;
      received += d.length;
      if (received > MAXBODY) {
        tooBig = true;
        jerr(res, 413, 'Request body too large. Maximum is ' + (MAXBODY / 1048576) + ' MB.', 'too_large');
        try { req.destroy(); } catch (_) {}
        return;
      }
      chunks.push(d);
    });
    req.on('end', () => {
      if (tooBig) return;
      const body = Buffer.concat(chunks);
      // One parse, shared with the relay (target.pre): under 256 KB here (exact 'model', it may come after 'messages';
      // JSON-object check: null, "x" or 1 crashed the engines), a larger one (base64 media) is scanned for the model,
      // must start with '{' and is parsed once by the relay. Bytes go up as sent unless the relay changes a field.
      const _small = body.length < 262144;
      let _asked = null, _pre = null;
      let _b = 0; while (body[_b] <= 32) _b++;
      if (_small) { try { _pre = JSON.parse(body.toString('utf8')); } catch (_) {} }
      if (body[_b] !== 123 || (_small && !_pre)) return jerr(res, 400, 'Request body must be a JSON object.', 'invalid');
      if (_pre && typeof _pre.model === 'string') _asked = _pre.model.slice(0, 120);
      // large body: Buffer.indexOf does not copy it
      if (!_asked) { try { const i = body.indexOf('"model"'); if (i >= 0) { const m = body.slice(i, i + 160).toString('utf8').match(/^"model"\s*:\s*"([^"]{1,120})"/); if (m) _asked = m[1]; } } catch (_) {} }
      const _mdl = modelFor(_asked);
      const _t0 = Date.now();
      // who sent it, for the owner's pulse: forwarded-for>socket address, then the user agent
      const _xf = String(req.headers['x-forwarded-for'] || '').split(',')[0].trim();
      const _from = (_xf ? _xf + '>' : '') + String(req.socket.remoteAddress || '').replace(/^::ffff:/, '') + ' ' + String(req.headers['user-agent'] || '').slice(0, 60);
      const _chat = /^\/v1\/chat\/completions\/?$/.test(p);
      // The drawer runs the generate_image tool a client declares itself (the gateway adds none) for whatever text model
      // answers (cloud or local): one model creates, one edits, each a local image model (no key, no catalog) or a cloud
      // one resolved with its own provider key.
      const _drawer = (id) => {
        const t = cloud.toolCfg(CLOUD, 'image'), f = krec.image;   // the key's image compatibility gates the drawer, not its chat list
        if (!t.on || !f || images.has(id) || cloud.isImageModel(CLOUD, id)) return null;
        const pick = (m) => !m ? null : images.has(m) ? m : cloud.isImageModel(CLOUD, m) ? cloud.resolve(CLOUD, m) : null;
        const mk = f === 'create' || f === 'both' ? pick(t.model) : null;
        let ed = f === 'edit' || f === 'both' ? pick(t.edit) : null;
        // a cloud model or a local 'both' one edits with the create model too
        if ((f === 'edit' || f === 'both') && !ed) { const cm = pick(t.model); if (cm && (typeof cm === 'object' || images.can(cm, 'edit'))) ed = cm; }
        return mk || ed ? { run: (a, cb, b, say) => drawWith(mk, ed, a, b, req, res, cb, say) } : null;
      };
      // The filmer runs generate_video the same way: the tool's local video model, else the first one that makes from text
      const _filmer = (id) => {
        const t = cloud.toolCfg(CLOUD, 'video');
        if (!t.on || videos.has(id) || images.has(id)) return null;
        const vm = videos.has(t.model) ? t.model : videos.list().map((m) => m.id).find((m) => videos.can(m, 'fl2va'));
        if (!vm) return null;
        return { run: (a, cb, b, say) => {
          // the picture to animate is found as the drawer finds the one to edit: an invented URL stops here
          const f = imageInput(a, b, baseOf(req));
          if (f && f.err) return cb({ md: '_(Video generation failed: ' + f.err + ')_', tin: 0, tout: 0, vids: 0, usd: 0, err: f.err });
          const t0 = Date.now();
          // a background job: its link goes out with the status, and the clip is kept if this caller leaves
          const url = videos.make(vm, { prompt: String(a.prompt || ''), seconds: +a.seconds || 0, first: f || '', base: baseOf(req) }, (r) => {
            try { fs.appendFileSync(path.join(__dirname, 'draw.log'), new Date().toISOString() + ' video ' + vm + ' ' + (Date.now() - t0) + 'ms ' + (r.err ? 'ERR ' + r.err : 'ok') + (f ? ' from picture' : '') + (res.destroyed ? ' kept, caller left' : '') + ' ' + r.url + '\n'); } catch (_) {}
            cb(r);
          });
          if (say) say('Making a video with ' + vm, { video: url });
        } };
      };
      // A local stream that waits hears why (Strata's phases: queued, loading): the SSE opens with the status, repeated every
      // 10 s as data, never a comment (idle timers count events). The Anthropic door ('(messages)') pings on its own.
      const _sse = (_pre ? _pre.stream === true : body.indexOf('"stream":true') > -1) && !/\(messages\)$/.test(req.headers['user-agent'] || '');
      // Local chat and completions: the model's own llama-server through the OpenAI relay. label = the id the client used.
      const serveLocal = (name, label) => {
        const ln = 'local:' + name, say = () => {
          const i = qOf(ln).w.indexOf(res._w), ph = chat.phase(name), eta = chat.eta(name);
          const s = i > -1 ? 'Waiting in line' + (i ? ' (' + i + ' ahead)' : '') : ph === 'memory' ? 'Waiting for memory' + (images.jobs() ? ': a picture is being made' : '')
            : ph === 'load' ? 'Loading ' + label + (eta ? ' (~' + eta + ' s)' : '') : ph === 'restore' ? 'Restoring conversations' : '';
          if (!_sse || !s || res.writableEnded) return;
          if (!res.headersSent) res.writeHead(200, Object.assign({ 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', 'x-accel-buffering': 'no' }, res._rl));
          res.write('data: ' + JSON.stringify({ id: 'chatcmpl-wait', object: 'chat.completion.chunk', created: Math.floor(Date.now() / 1000), model: label, choices: [{ index: 0, delta: { status: s }, finish_reason: null }] }) + '\n\n');
        }, tk = setInterval(say, 10000);
        res.once('close', () => clearInterval(tk));
        const wq = acquire(res, ln).then(() => {
        const _q = Date.now() - _t0; let out = false, left = false, th; const relL = () => { if (!out) { out = true; release('local:' + name); } };
        // generating once its engine serves it, until it ends or hands the engine to its own picture
        let gen = 0; const ug = () => { if (gen) { gen = 0; chatGen--; } };
        res.on('close', () => { left = true; relL(); ug(); });
        const fin = (d) => {
          relL(); ug();
          if (d.tin || d.tout || d.imgs) track(apiKey, label, d.tin || 0, d.tout || 0, d.imgs || 0, d.usd || 0);
          pulse({ t: clock(), key: krec.name, model: label, from: _from,
            in: d.tin || 0, out: d.tout || 0, ms: d.ms || 0, queue: _q, cut: d.cut || undefined,
            tps: wtps(d), cn: d.cn, pn: d.pn, pms: d.pms, dn: d.dn, da: d.da, str: d.str,
            code: d.code, ok: !!d.ok, err: d.err ? 'llama: ' + d.err : undefined, th: th });
          // the engine's own numbers and O's stranded call stay in the journal too: the pulse keeps 60 rows and dies
          // with every restart, and O is decided on 1-2 weeks of them
          console.log('POST', p, d.cut ? 'cut' : d.code, krec.name, label, 'ctx=' + chat.plan(name)
            + (d.pn != null ? ' ms=' + d.ms + ' tps=' + d.wtps + ' cache=' + d.cn + '/' + (d.cn + d.pn) + ' mtp=' + d.da + '/' + d.dn : '') + (d.fr ? ' fin=' + d.fr : '') + (d.str ? ' stranded' : ''));
        };
        chat.load(name, true, (e) => {
          clearInterval(tk);
          if (left) return;   // it left while the model started: nothing ran
          const t = !e && chat.target(name);
          if (!t) { relL(); return jerr(res, 503, "Local model '" + label + "' is not available: " + (e ? e.message : 'unloaded') + '.', 'upstream'); }
          gen = 1; chatGen++;
          if (!_chat) t.path = '/completions';
          // its picture starts once the engine's turn is over: slot, permit and GPU go free (a quiet brain may yield)
          else if (t.meta.tools) { const dr = _drawer(label); if (dr) t.drawer = { run: (a, cb, b, say) => { ug(); relL(); t.done(); dr.run(a, cb, b, say); } }; const fm = _filmer(label); if (fm) t.filmer = { run: (a, cb, b, say) => { ug(); relL(); t.done(); fm.run(a, cb, b, say); } }; }
          // Thinking: the client's level ('think' first, as before; the pack fills reasoning_effort) mapped to what the
          // template takes, none asked = think_default. Media only as data URLs. Returns true when the body changed.
          t.prep = (o) => {
            const rs = o.reasoning && typeof o.reasoning === 'object' ? o.reasoning : null;   // OpenRouter's reasoning: {effort, max_tokens}
            const lv = o.think != null ? o.think : o.reasoning_effort != null ? o.reasoning_effort : rs && rs.effort, asked = lv == null ? thinkDefault(name) : lv;
            let d = o.think !== undefined; delete o.think;
            th = chat.effort(name, asked);
            if (th === undefined) { if (o.reasoning_effort !== undefined) d = true; delete o.reasoning_effort; } else if (o.reasoning_effort !== th) { o.reasoning_effort = th; d = true; }
            // the border speaks OpenAI/OpenRouter: max_completion_tokens and reasoning.max_tokens become llama-server's names
            if (o.max_completion_tokens != null) { if (o.max_tokens == null) o.max_tokens = o.max_completion_tokens; delete o.max_completion_tokens; d = true; }
            if (rs) { if (rs.max_tokens != null && o.reasoning_budget_tokens == null) o.reasoning_budget_tokens = rs.max_tokens; delete o.reasoning; d = true; }
            // the brake is the server's: its output cap and its thinking cap for this level, when the client sends none; and no cap
            // so near max_tokens that the answer has no room (Strata serve/server.py:4199-4215: a quarter, 512 to half; from 1024
            // on, a thinking turn with no cap gets one). A client's 0 or -1 is its own choice.
            const om = chat.outMax(name); if (om && o.max_tokens == null && o.n_predict == null) { o.max_tokens = om; d = true; }
            if (th !== 'none') {
              const mt = +(o.max_tokens ?? o.n_predict) || 0, room = Math.min(Math.max(512, mt >> 2), mt >> 1);
              const k = o.reasoning_budget_tokens != null ? 'reasoning_budget_tokens' : o.thinking_budget_tokens != null ? 'thinking_budget_tokens' : '';
              let cap = k ? +o[k] : chat.budget(name, th || asked);
              if (mt && (cap > mt - room || !k && !cap && mt >= 1024)) cap = Math.max(1, mt - room);
              if (cap > 0 && cap !== (k ? +o[k] : 0)) { o[k || 'reasoning_budget_tokens'] = cap; if (!k) o.reasoning_budget_message = chat.budgetMsg(name); d = true; }
            }
            // the model card's sampling: thinking, its floor (colder loops, 12-oct: lifted; warmer stays); else its set fills the gaps
            if (th === 'none' ? fill(o, chat.card(name, 'instruct_sampling')) : liftToFloor(o, chat.card(name, 'think_sampling'))) d = true;
            // her DNA (think_more in the model's card) only while it thinks: the relay reads it (cloud_relay.js dnaReply)
            t.thinkMore = th !== 'none' ? chat.card(name, 'think_more') : null;
            const kit = chat.card(name, 'kit');
            if (kit && Array.isArray(o.tools) && o.tools.some((x) => x && (x.function || x).name === 'use')) { kitTools(o, kit); d = true; }
            if (localParts(o.messages, krec && krec.name)) d = true;
            // the model's own layer (staff wrote it here) goes before the owner's System Prompt
            const ly = chat.layer(name);
            if (ly && Array.isArray(o.messages)) { const s0 = o.messages[0]; if (s0 && s0.role === 'system' && typeof s0.content === 'string') s0.content = ly + '\n\n' + s0.content; else o.messages.unshift({ role: 'system', content: ly }); d = true; }
            if (saneHist(o.messages)) d = true;   // J, last: the history the template renders is this final one
            return d;
          };
          res.once('close', t.done);
          // clips are fitted to the model first; a client that leaves stops the conversion
          const ac = new AbortController(); res.once('close', () => ac.abort());
          video.fit(body, _pre, Object.assign({ can: chat.takes(name, 'video'), ctx: chat.plan(name), signal: ac.signal }, (cfg.chat || {}).video), (ve, b, o) => {
            if (left) return;
            if (ve) { t.done(); relL(); ug(); return jerr(res, 400, ve.message + '.', 'invalid'); }
            t.pre = o;
            cloud.relay(res, LOCAL(), t, b, (d) => { t.done(); fin(d); });
          });
        });
        say();
        });
        if (res._w) say(); return wq;   // in line: said now; the rest once the load starts
      };
      const _local = chat.id(_mdl || _asked);
      if (images.has(_local)) return jerr(res, 400, "'" + _local + "' is an image model: it draws through the generate_image tool of a text model.", 'invalid', 'model');
      if (!_mdl) return jerr(res, 404, "The model '" + (_asked || '') + "' does not exist.", 'not_found', 'model');
      if (chat.has(_mdl)) return serveLocal(_mdl, _mdl);
      const _sky = cloud.resolve(CLOUD, _mdl);
      if (_sky && loopback(_sky)) return serveLocal(chat.id(_sky.model), _sky.id);
      if (_sky) {
        return acquire(res, 'cloud').then(() => {
          const _qw = Date.now() - _t0;
          let out = false; const relS = () => { if (!out) { out = true; release('cloud'); } };
          res.on('close', relS);
          // the catalog says which providers take tools
          _sky.drawer = _chat && _sky.meta.tools && _drawer(_sky.id); _sky.filmer = _chat && _sky.meta.tools && _filmer(_sky.id);
          _sky.pre = _pre;
          cloud.relay(res, CLOUD, _sky, body, (d) => {
            relS();
            if (d.tin || d.tout || d.imgs) track(apiKey, _sky.id, d.tin || 0, d.tout || 0, d.imgs || 0, d.usd || 0);
            pulse({ t: clock(), key: krec.name, model: _sky.id, from: _from,
              in: d.tin || 0, out: d.tout || 0, ms: d.ms || 0, queue: _qw, cut: d.cut || undefined,
              tps: wtps(d),
              code: d.code, ok: !!d.ok, err: d.err ? _sky.prov + ': ' + d.err : undefined });
            console.log('POST', p, d.cut ? 'cut' : d.code, krec.name, _sky.id);
          });
        });
      }
      if (isClaude(_mdl)) {
        // queued as well: no VRAM, but it bounds concurrent claude processes
        return acquire(res, 'claude').then(() => {
          const waitC = Date.now() - _t0;
          let freed = false; const relC = () => { if (!freed) { freed = true; release('claude'); } };
          res.on('close', relC);
          engine.serve(res, CLA, body, (d) => {
            relC();
            // Claude is the only engine that reports the exact cost in dollars: it is kept
            if (d.tin || d.tout) track(apiKey, CLA.id, d.tin || 0, d.tout || 0, 0, d.usd || 0);
            pulse({
              t: clock(), key: krec.name, model: CLA.id, from: _from,
              in: d.tin || 0, out: d.tout || 0, ms: d.ms || 0, queue: waitC, cut: d.cut || undefined,
              tps: wtps(d),
              code: d.code, ok: !!d.ok, err: d.err ? 'claude: ' + d.err : undefined,
            });
            console.log('POST', p, d.cut ? 'cut' : d.code, krec.name, CLA.id);
          });
        });
      }
      if (_local) return jerr(res, 503, "Model '" + _local + "' is not available right now.", 'upstream', 'model');
      jerr(res, 400, "Missing required parameter: 'model'. No active model is set.", 'invalid', 'model');
    });
    return;
  }
  jerr(res, 404, 'Not found.', 'not_found');
});

// What is selected stays up: chat_local starts it again if it dies.
function pinLocks() {
  const locks = selected().filter((m) => chat.has(m));
  chat.pins(locks);   // a model that left the locks falls back to the idle unload
  locks.forEach(m => chat.load(m, true));
  return locks;
}

// The picture to edit: ours (rebuilt from its id on our own base, read only from disk while it lives), or an http(s)
// URL the conversation really carries. One the model invented is REFUSED: not fetched, not replaced by another photo,
// nothing drawn -- falling back to "the last photo" would turn it into an edit of somebody's upload.
// No URL, or "attached", still means a picture in the conversation (data URL or http image part), the last one when
// the tool names an image it cannot link, or one in the last user turn.
function imageInput(a, body, base) {
  const u = String(a.image || '').trim(), own = /^https?:\/\/[^/?#]+\/img\/([0-9a-f]{32})\.png(?:[?#]|$)/.exec(u);
  if (own) return base + '/img/' + own[1] + '.png';
  let last = '', turn = '', seen = false;
  const hit = (s) => { if (u && typeof s === 'string' && s.indexOf(u) > -1) seen = true; };
  ((body && Array.isArray(body.messages)) ? body.messages : []).forEach((m) => {
    if (!m) return;
    let x = '';
    hit(m.content);
    (Array.isArray(m.content) ? m.content : []).forEach((p) => {
      if (!p) return;
      const v = p.image_url ? p.image_url.url || p.image_url : p.image;
      if (typeof v === 'string' && /^(data:image\/|https?:\/\/)/.test(v)) x = v;
      hit(v); hit(p.text);
    });
    if (x) last = x;
    if (m.role === 'user') turn = x;
  });
  // an http(s) URL nobody in this conversation wrote is invented: an error, never a fallback to another photo
  if (/^https?:\/\//.test(u)) return seen ? u : { err: 'that image URL is not in this conversation' };
  return u ? last : turn;
}
// Absolute base for links to this server's files, from the Host the client used.
const baseOf = (req) => { const h = String(req.headers.host || '');
  // the socket itself is the truth when we terminate TLS; the header only rules behind a proxy
  return (req.headers['x-forwarded-proto'] || (req.socket.encrypted ? 'https' : 'http')) + '://' + (/^([\w.-]+|\[[\da-f:.]+\])(:\d+)?$/i.test(h) ? h : (lanIPs()[0] || '127.0.0.1') + ':' + cfg.port); };
// One picture with the drawer's create model mk or edit model ed (local id or resolved cloud model), result logged.
// A local picture whose client left is skipped while queued; one already drawing is dropped when it ends.
function drawWith(mk, ed, a, body, req, res, cb, say) {
  a = a || {};
  let left = false; res.once('close', () => { left = !res.writableEnded; });
  const found = imageInput(a, body, baseOf(req));
  // an invented URL stops here: nothing is fetched, nothing is drawn, and draw.log says why
  if (found && found.err) {
    try { fs.appendFileSync(path.join(__dirname, 'draw.log'), new Date().toISOString() + ' REFUSED ' + found.err + ' | ' + String(a.prompt || '').slice(0, 70).replace(/\s+/g, ' ') + '\n'); } catch (_) {}
    return cb({ md: '_(Image generation failed: ' + found.err + ')_', tin: 0, tout: 0, imgs: 0, usd: 0, err: found.err });
  }
  // edit needs a picture and an edit model; a cloud one only takes the picture by URL
  const src = found, edit = !!(src && ed && (typeof ed === 'string' || /^https?:/.test(src))), d = edit ? ed : mk, t0 = Date.now(), prompt = String(a.prompt || '');
  // create and edit cost very differently, so draw.log records which one, with what, how long and how it ended
  const log = (r) => { try { fs.appendFileSync(path.join(__dirname, 'draw.log'), new Date().toISOString() + ' ' + (edit ? 'EDIT ' : 'CREATE ') + ((d && d.id) || d)
    + ' ' + (Date.now() - t0) + 'ms ' + (r.imgs ? 'ok' : 'error: ' + String(r.err || '').slice(0, 120)) + (r.wait != null ? ' wait_ms ' + r.wait : '') + (edit ? ' ' + (/^data:/.test(src) ? 'upload' : src.slice(0, 70)) : '')
    + ' | ' + prompt.slice(0, 70).replace(/\s+/g, ' ') + '\n'); } catch (_) {} cb(r); };
  if (!d) return log({ md: '_(Image generation failed: no model to create images)_', tin: 0, tout: 0, imgs: 0, err: 'no create model' });
  if (say) say(edit ? 'Editing image' : 'Creating image');   // the instruction leaves for the image model now
  if (typeof d === 'string') return images.draw(d, { prompt: prompt, image: edit ? src : '', base: baseOf(req), quality: CLOUD.image_quality, gone: () => left }, log);
  cloud.draw(CLOUD, d, Object.assign({}, a, { image: edit ? src : '' }), log);
}

server.listen(cfg.port, '0.0.0.0', () => {
  console.log('ia_server listening on 0.0.0.0:' + cfg.port + ' -> ' + (EN().name || 'engine') + ' ' + chat.version() + ' (' + chat.names().join(', ') + ')');
  console.log('public_url: ' + (cfg.public_url || 'not set'));
  console.log('max_concurrent: ' + MAXC);
  console.log('keys: ' + (cfg.keys || []).map(k => k.name || 'unnamed').join(', '));
  lanIPs().forEach(ip => console.log('lan: http://' + ip + ':' + (cfg.port + 1) + '/v1'));   // the local door, no cert needed
  // first start: no admin token yet, so one is made and shown once here (config.json keeps it)
  if (!cfg.admin_token) { cfg.admin_token = 'adm-' + crypto.randomBytes(12).toString('hex'); saveConfig(); console.log('admin token (panel /admin): ' + cfg.admin_token); }
  const locks = pinLocks(); if (locks.length) console.log('pinned: ' + locks.join(', '));
  const ims = images.list(); if (ims.length) console.log('local images (on demand): ' + ims.map(m => m.id + '=' + m.task).join(', '));
});

