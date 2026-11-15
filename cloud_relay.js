'use strict';
/*
 * cloud_relay.js: cloud providers served through ia_server. The owner's provider key stays on the
 * server; clients use their own ia_server key and quota.
 * The provider catalog is the server's own (config cloud.catalog {providers, ctx}): it depends on no one. Adapter rules:
 *  - endpoint = baseUrl + /chat/completions (every provider is OpenAI-compatible), Bearer auth
 *  - api.openai.com takes max_completion_tokens, the others max_tokens
 *  - output floor of 2048 tokens: reasoning models spend a small budget thinking and return empty text
 * Responses are piped; only the tail is kept to read usage.
 */

const https = require('https');
const http = require('http');
const { URL } = require('url');
const { StringDecoder } = require('string_decoder');
const prices = require('./prices');
// keep-alive pools (Node 18's default agent connects per request): the local engines (maxSockets per port = slots + 2) and the
// cloud, whose TLS handshake is then paid once per host; an idle cloud socket is dropped at 30 s (a reset on reuse is retried, see relay)
const localAgent = new http.Agent({ keepAlive: true }), cloudAgent = new https.Agent({ keepAlive: true, timeout: 30000 });

const MODELS_TTL  = 30 * 60 * 1000;

let catalog = null, modelCtx = {};   // modelCtx: the catalog's root ctx{}, each model's window
const modelsCache = Object.create(null);   // provider -> { at, ids[], ctx{} }

const DEF = { providers: {}, timeout_ms: 120000, min_output: 2048, tokens_per_image: 0 };

function options(raw) {
  const c = Object.assign({}, DEF, raw || {});
  c.providers = c.providers || {};
  c.timeout_ms = Math.max(10000, +c.timeout_ms || DEF.timeout_ms);
  c.min_output = Math.max(0, +c.min_output || DEF.min_output);
  const k = c.catalog || {}; catalog = k.providers || null; modelCtx = k.ctx || {};
  return c;
}

// single network helper: raw text and response
function fetchRaw(url, headers, payload, cb) {
  let u; try { u = new URL(url); } catch (_) { return cb(new Error('bad url')); }
  const mod = u.protocol === 'https:' ? https : http;
  const h = Object.assign({}, headers || {});
  if (payload) { h['content-type'] = 'application/json'; h['content-length'] = payload.length; }
  const req = mod.request(u, { method: payload ? 'POST' : 'GET', headers: h, timeout: 20000 }, (r) => {
    let s = '';
    r.on('data', (d) => { if (s.length < 1048576) s += d; });
    r.on('end', () => cb(null, r, s));
  });
  req.on('error', cb);
  req.on('timeout', () => { req.destroy(new Error('timeout')); });
  if (payload) req.write(payload);
  req.end();
}
function getJson(url, headers, cb) {
  fetchRaw(url, headers, null, (e, r, s) => {
    if (e) return cb(e);
    try { cb(null, JSON.parse(s), r.statusCode); } catch (e2) { cb(e2); }
  });
}

// Provider health, learned from traffic already paid for: prov -> { at, code, state, msg, limit{} }
const health = Object.create(null);
// Output token cap per provider, learned from its own errors; none is hardcoded.
const outCaps = Object.create(null);
// "Range of max_tokens should be [1, 131072]" | "max_tokens must be <= 8192" | "at most 4096"
const capFromMsg = (m) => {
  if (!m || !/max_?tokens|max_completion_tokens/i.test(m)) return 0;
  const r = m.match(/\[\s*\d+\s*,\s*(\d+)\s*\]/) || m.match(/(?:<=|less than or equal to|at most|maximum of)\s*(\d+)/i);
  return r ? +r[1] || 0 : 0;
};
const BALANCE_RE = /insufficient|balance|quota|credit|recharge|billing|arrears|expired/i;

function stateOf(code, msg) {
  if (!code) return 'down';
  if (code === 401 || code === 403) return 'bad_key';
  if (code === 402) return 'no_balance';
  if (code === 429) return BALANCE_RE.test(msg || '') ? 'no_balance' : 'rate_limit';
  if (code >= 500) return 'down';
  return code < 400 ? 'ok' : 'error';
}

// the provider's message verbatim: that is the text to search on its site
function errMsg(txt, n) {
  const t = String(txt || '');
  try { const j = JSON.parse(t.trim()), e = j.error || j; if (e && e.message) return String(e.message).slice(0, n || 160); } catch (_) {}
  const m = t.match(/"message"\s*:\s*"([^"]{1,160})"/);
  return m ? m[1] : '';
}

function note(prov, code, msg, headers) {
  const h = headers || {}, limit = {};
  // rate-limit headers are collected by pattern (x-ratelimit-*, anthropic-ratelimit-*)
  Object.keys(h).forEach((k) => {
    if (/ratelimit-/.test(k)) limit[k.replace(/^.*?ratelimit-/, '')] = String(h[k]).slice(0, 40);
  });
  health[prov] = { at: Date.now(), code: code || 0, state: stateOf(code, msg), msg: msg || '', limit: limit };
  return health[prov];
}

// Client ids are "provider/model": two providers may serve the same model name (glm-4.6 on zai and bigmodel).
const splitId = (id) => { const i = String(id || '').indexOf('/'); return i < 1 ? null : { prov: id.slice(0, i), model: id.slice(i + 1) }; };

// Call details for an id; null when it is not a cloud id or the provider is off.
function resolve(cfg, id) {
  if (!catalog) return null;
  const s = splitId(id);
  if (!s) return null;
  const p = cfg.providers[s.prov], meta = catalog[s.prov];
  if (!p || p.enabled === false || !meta) return null;
  const baseUrl = String(p.baseUrl || meta.baseUrl || '').replace(/\/+$/, '');
  if (!baseUrl || !p.key) return null;
  // a non-empty models list limits what is shared
  if (Array.isArray(p.models) && p.models.length && p.models.indexOf(s.model) === -1) return null;
  return { id: id, prov: s.prov, model: s.model, baseUrl: baseUrl, key: p.key, meta: meta };
}

// Configured ids are added to the provider's /models list: some variants exist but are not listed.
const mergeIds = (fixed, live) => [...new Set([].concat(Array.isArray(fixed) ? fixed : [], (live && live.ids) || []))];
function providerModels(cfg, prov, cb) {
  const c = modelsCache[prov];
  if (c && Date.now() - c.at < MODELS_TTL) return cb(c);
  const p = cfg.providers[prov], meta = catalog && catalog[prov];
  if (!p || !meta) return cb(null);
  const baseUrl = String(p.baseUrl || meta.baseUrl || '').replace(/\/+$/, '');
  if (!baseUrl || !p.key) return cb(null);
  getJson(baseUrl + '/models', { accept: 'application/json', authorization: 'Bearer ' + p.key }, (e, j) => {
    if (e || !j) return cb(null);
    const arr = j.data || j.models || [], ids = [], ctx = {};
    arr.forEach((m) => {
      const id = m && (m.id || m.name); if (!id) return;
      ids.push(id);
      // the catalog names the field that carries each model's context (ctxField)
      if (meta.ctxField && +m[meta.ctxField] > 0) ctx[id] = +m[meta.ctxField];
    });
    const rec = { at: Date.now(), ids: ids, ctx: ctx };
    modelsCache[prov] = rec;
    cb(rec);
  });
}

// Context priority: the provider's list in the server's catalog > provider /models > the model's own window > provider default > 128k.
function ctxInfo(prov, model, live) {
  const meta = (catalog && catalog[prov]) || {}, cur = meta.ctx || {};
  const k = Object.keys(cur).find((x) => x.toLowerCase() === String(model).toLowerCase());
  if (k && +cur[k] > 0) return { ctx: +cur[k], src: 'catalog' };
  if (live && live.ctx && +live.ctx[model] > 0) return { ctx: +live.ctx[model], src: 'provider' };
  // the model's own window (as the app's _mkey: no house prefix, no date suffix)
  const mk = (x) => String(x).toLowerCase().replace(/^[^/]*\//, '').replace(/-\d{4,}$/, '');
  const g = Object.keys(modelCtx).find((x) => mk(x) === mk(model));
  if (g && +modelCtx[g] > 0) return { ctx: +modelCtx[g], src: 'model' };
  return { ctx: +meta.defaultCtx || 128000, src: 'default' };   // numeric: the catalog once sent a string
}
function ctxOf(prov, model, live) { return ctxInfo(prov, model, live).ctx; }

// For /v1/models; a failing provider does not hide the others.
function listModels(cfg, cb) {
  const provs = Object.keys(cfg.providers).filter((p) => {
    const c = cfg.providers[p];
    return c && c.enabled !== false && c.key && catalog && catalog[p];
  });
  if (!provs.length) return cb([]);
  const out = [];
  let pend = provs.length;
  provs.forEach((prov) => {
    const fixed = cfg.providers[prov].models;
    const push = (live) => {
      const ids = mergeIds(fixed, live);
      ids.forEach((m) => out.push({
        id: prov + '/' + m, prov: prov, model: m,
        label: (catalog[prov] || {}).label || prov,
        ctx: ctxOf(prov, m, live), ctxSrc: ctxInfo(prov, m, live).src,
      }));
      if (--pend === 0) cb(out);
    };
    providerModels(cfg, prov, push);
  });
}

// An image model: the owner priced it per image, or the rates table says so. Explicit, not guessed.
const isImageModel = (cfg, id) => !!((cfg.usd_per_image || {})[id]) || (prices.rateOf(id) || {}).mode === 'image_generation';
// DashScope image reply -> OpenAI. The picture goes as markdown so any chat renders it as-is.
function toOpenAI(j, model) {
  const outs = ((j.output || {}).choices) || [];
  const urls = [];
  outs.forEach((c) => (((c.message || {}).content) || []).forEach((p) => { if (p && p.image) urls.push(p.image); }));
  const text = ((outs[0] || {}).message || {}).content;
  const words = Array.isArray(text) ? text.map((p) => p && p.text).filter(Boolean).join('\n') : '';
  const body = urls.map((u) => '![](' + u + ')').join('\n') + (words ? (urls.length ? '\n\n' : '') + words : '');
  const u = j.usage || {};
  const tin = +u.input_tokens || +u.prompt_tokens || 0, tout = +u.output_tokens || +u.completion_tokens || 0;
  return {
    id: 'chatcmpl-' + (j.request_id || '').slice(0, 24), object: 'chat.completion',
    created: Math.floor(Date.now() / 1000), model: model,
    choices: [{ index: 0, message: { role: 'assistant', content: body }, finish_reason: 'stop' }],
    usage: { prompt_tokens: tin, completion_tokens: tout, total_tokens: tin + tout, image_count: +u.image_count || urls.length, size: u.size },
  };
}
// DashScope content: a LIST of items, each {text} OR {image}, never both (400 otherwise).
function dsContent(c) {
  const out = [];
  [].concat(c == null ? [] : c).forEach((p) => {
    if (typeof p === 'string') return p && out.push({ text: p });
    if (!p) return;
    const img = p.image_url ? p.image_url.url || p.image_url : p.image;
    if (p.text) out.push({ text: String(p.text) });
    if (typeof img === 'string' && img) out.push({ image: img });
  });
  return out;
}

// What the gateway EXECUTES, the OpenAI way: a client that wants a picture declares generate_image among its own tools
// ({prompt, size: 1K|2K, image: url to edit}); the gateway adds none (an injected card cost 394 prompt tokens and a cold
// prefill per chat, measured). cloud.tools[id] = { on, model, edit }: model creates, edit changes a given picture
// ('provider/modelId' or a local image model). A new capability is an entry plus its runner.
const TOOLS = { image: { label: 'Image generation', fn: 'generate_image' } };
const declares = (b, fn) => Array.isArray(b.tools) && b.tools.some((x) => ((x || {}).function || {}).name === fn);
const pick = (o) => { const p = Object.keys(o || {}).find((k) => o[k]); return p ? p + '/' + o[p] : ''; };
// Any stored shape -> {on, model, edit}. Old shapes ({prov:id} map, legacy image_model) are read
// here and migrated on write. An explicit OFF wins over every fallback.
function toolCfg(cfg, id) {
  const t = (cfg.tools || {})[id] || {}, m = t.model;
  return { on: t.on !== false, model: typeof m === 'string' ? m : pick(m) || (id === 'image' ? pick(cfg.image_model) : ''), edit: typeof t.edit === 'string' ? t.edit : '' };
}
// Cloud drawer runner: the picture with resolved model d, always calling back {md,tin,tout,imgs,usd}
// (err on failure): md is the picture markdown or the failure note, so callers only glue.
function draw(cfg, d, args, cb) {
  args = args || {};
  const edit = /^https?:\/\//.test(args.image || ''), prompt = String(args.prompt || '');
  let once = 0;
  const out = (r, err) => { if (!once++) cb(r || { md: '_(Image generation failed' + (err ? ': ' + err : '') + ')_', tin: 0, tout: 0, imgs: 0, err: err || 'failed' }); };
  // EDIT is the same endpoint: the picture travels as its own item next to the text.
  const body = Buffer.from(JSON.stringify({ model: d.model,
    messages: [{ role: 'user', content: dsContent(edit ? [{ image: args.image }, prompt] : prompt) }],
    parameters: { size: args.size === '2K' ? '2K' : '1K' } }));
  let u; try { u = new URL(d.baseUrl + '/chat/completions'); } catch (_) { return out(null, 'bad url'); }
  const q = (u.protocol === 'https:' ? https : http).request(u, { method: 'POST', timeout: cfg.timeout_ms, headers: {
    authorization: 'Bearer ' + d.key, 'content-type': 'application/json', 'content-length': body.length } }, (r) => {
    let raw = '';
    r.on('data', (x) => { if (raw.length < 262144) raw += x; });
    // a body cut upstream never ends
    r.on('close', () => { if (!r.complete) out(null, 'aborted'); });
    r.on('end', () => {
      let j = null; try { j = JSON.parse(raw); } catch (_) {}
      const urls = [];
      (((j || {}).output || {}).choices || []).forEach((c) => (((c.message || {}).content) || [])
        .forEach((p) => { if (p && p.image) urls.push(p.image); }));
      if (!urls.length) return out(null, errMsg(raw) || 'HTTP ' + r.statusCode);
      const us = j.usage || {}, imgs = +us.image_count || urls.length;
      out({ md: urls.map((x) => '![](' + x + ')').join('\n'), tin: +us.input_tokens || 0,
        tout: (+us.output_tokens || 0) + (prices.imageChips(imgs, us.size, cfg.image_quality) || 0),
        imgs: imgs, usd: prices.costOf(d.id, us, cfg) });
    });
  });
  q.on('error', (e) => out(null, e.message));
  q.on('timeout', () => q.destroy(new Error('timeout')));
  q.write(body); q.end();
}

// onDone({tin,tout,imgs,usd,ms,code,ok,err,cut}) lets the server track and pulse. Optional on target: prep(body)
// (true = it changed the body), path, pre (the body the server parsed) and slots (local engines).
function relay(res, cfg, target, rawBody, onDone, retried) {
  const t0 = Date.now();
  // gateway overhead = this request's ms minus the engine's own prompt + generation ms, once it says them
  let ov = -1; const over = (tm) => { if (tm && tm.predicted_ms != null) ov = Math.max(0, Date.now() - t0 - Math.round((+tm.prompt_ms || 0) + (+tm.predicted_ms || 0))); };
  // one report per request: a cut stream can end, close and error in any order
  let told = 0;
  const report = (d) => { if (told++) return; try { if (onDone) onDone(d); } catch (_) {} };
  // the upstream failed before a whole reply: a 502 while nothing went out, else an open stream ends with the error as an
  // event and [DONE] (Strata's way); never a clean end, a silent cut or a hang
  const fail = (m) => {
    const eo = JSON.stringify({ error: { message: target.prov + ': ' + m, type: 'api_error', param: null, code: 'upstream_error' } });
    if (!res.headersSent) { res.writeHead(502, { 'content-type': 'application/json' }); res.end(eo); }
    else if (body.stream === true) res.writableEnded || res.end('\n\ndata: ' + eo + '\n\ndata: [DONE]\n\n'); else res.destroy();
    report({ tin: 0, tout: 0, ms: Date.now() - t0, code: 502, ok: false, err: m });
  };
  // a retry can start after the client left: nothing is generated for nobody
  if (!res.socket || res.socket.destroyed) return report({ tin: 0, tout: 0, ms: 0, code: 200, ok: true, cut: true });

  // One parse per request (the server hands its own as pre; a retry parses the client's bytes again). The bytes go up
  // AS SENT unless a field below changes (dirty): a rewrite is ~1 ms per 60 KB and two more copies of a picture body.
  let body = target.pre, dirty = false; target.pre = null;
  if (!body) { try { body = JSON.parse(rawBody.toString('utf8')); } catch (_) {} }
  if (!body || typeof body !== 'object' || Array.isArray(body)) { body = {}; dirty = true; }
  if (body.model !== target.model) { body.model = target.model; dirty = true; }   // the provider's own name, without our prefix
  // target-specific preparation after the parse (the local chat maps the thinking level here)
  if (target.prep && target.prep(body)) dirty = true;
  if (target.prov !== 'local') {
    // a cloud knows low..high: the top of our scale reaches it as high (the local chat fitted its own level above)
    if (body.reasoning_effort === 'xhigh' || body.reasoning_effort === 'max') { body.reasoning_effort = 'high'; dirty = true; }
    // llama-server's prompt-reading progress: a cloud never sees the flag
    if (body.return_progress !== undefined) { delete body.return_progress; dirty = true; }
  }
  // Image models speak DashScope (same assumption as the reply below): OpenAI parts -> native items.
  if (isImageModel(cfg, target.id) && Array.isArray(body.messages)) { body.messages.forEach((m) => { if (m && m.role) m.content = dsContent(m.content); }); dirty = true; }

  // output floor for reasoning models, clipped to the provider cap once learned (sending more is a sure 400)
  const asked = +body.max_tokens || +body.max_completion_tokens || 0, had = body.max_tokens, hadC = body.max_completion_tokens;
  let maxTok = Math.max(asked, cfg.min_output);
  if (outCaps[target.prov] && maxTok > outCaps[target.prov]) maxTok = outCaps[target.prov];
  delete body.max_tokens; delete body.max_completion_tokens;
  // OpenAI deprecated max_tokens and rejects it on reasoning models; 0 (no floor, none asked) = the server's own limit
  if (maxTok) body[/api\.openai\.com/.test(target.baseUrl) ? 'max_completion_tokens' : 'max_tokens'] = maxTok;
  if (body.max_tokens !== had || body.max_completion_tokens !== hadC) dirty = true;

  // The drawer comes from the server (it owns the key policy) as {run(args, cb, body)}: cloud or local, never for an
  // image model. Only a request that declares generate_image itself takes it; the rest are piped, timings included.
  const drawer = target.drawer && body.tool_choice !== 'none' && declares(body, TOOLS.image.fn) ? target.drawer : null;

  // her DNA (card think_more): a doubt (empty, capped, past the gate) brings k-1 more thoughts, the surest by its tail answers
  // llama-server refuses logprobs on a stream with tools: only whole replies ask them; a stream asks reading progress, it bills a stopped thought
  const dna = target.thinkMore && !drawer && !(+body.n > 1) ? target.thinkMore : null, dnaReqs = [];
  let dnaSse = false, dnaLp = false, dnaPp = false;
  if (dna) {
    dnaSse = body.stream === true; dnaLp = body.logprobs === true; dnaPp = body.return_progress === true;
    if (!dnaSse) { body.logprobs = true; body.top_logprobs = Math.max(5, +body.top_logprobs || 0); } else if (target.prov === 'local') body.return_progress = true;
    dirty = true;
  }

  // usage is always asked: the quota is billed from it (a client's include_usage:false only drops its own reading)
  if (body.stream === true && !(body.stream_options && body.stream_options.include_usage === true)) { body.stream_options = Object.assign({}, body.stream_options, { include_usage: true }); dirty = true; }

  const payload = dirty ? Buffer.from(JSON.stringify(body)) : rawBody;
  let u; try { u = new URL(target.baseUrl + (target.path || '/chat/completions')); } catch (_) {
    res.writeHead(500, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: { message: 'Invalid provider baseUrl.', type: 'api_error', param: null, code: 'config_error' } }));
    return report({ tin: 0, tout: 0, ms: 0, code: 500, ok: false, err: 'invalid baseUrl' });
  }

  const mod = u.protocol === 'https:' ? https : http, local = target.prov === 'local';
  if (local) localAgent.maxSockets = (+target.slots || 4) + 2;
  let gone = false, upEnd = false;
  const up = mod.request(u, {
    method: 'POST',
    headers: {
      'authorization': 'Bearer ' + target.key,
      'content-type': 'application/json',
      'content-length': payload.length,
      'accept': body.stream === true ? 'text/event-stream' : 'application/json',
    },
    timeout: cfg.timeout_ms, agent: local ? localAgent : mod === https ? cloudAgent : undefined,
  }, (ur) => {
    ur.once('end', () => { upEnd = true; });
    // an error is read fully before replying: the only chance to learn from it and retry
    if (ur.statusCode >= 400) {
      let bad = '';
      ur.on('data', (d) => { if (bad.length < 8192) bad += d; });
      // 'close' follows 'end', and also comes alone when the body is cut: what arrived is the answer
      ur.on('close', () => {
        const msg = errMsg(bad);
        note(target.prov, ur.statusCode, msg, ur.headers);
        // the output cap is learned from the message and the request is repeated, clipped
        if (ur.statusCode === 400 && !retried) {
          const cap = capFromMsg(msg);
          if (cap > 0 && cap < maxTok) { outCaps[target.prov] = cap; return relay(res, cfg, target, rawBody, onDone, true); }
        }
        // Always the gateway shape, provider text kept: bodies come native (DashScope), as HTML or cut at 8 KB.
        const s = ur.statusCode;
        // a prompt over the context keeps OpenAI's code: clients compact the history on it (llama-server: exceed_context_size_error)
        const code = /exceed_context_size|context_length_exceeded/.test(bad) ? 'context_length_exceeded' : 'upstream_error';
        const eo = JSON.stringify({ error: { message: target.prov + ': ' + (errMsg(bad, 2000) || 'HTTP ' + s), type: s === 429 ? 'rate_limit_error' : s < 500 ? 'invalid_request_error' : 'api_error', param: null, code: code } });
        if (!res.headersSent) { res.writeHead(s, { 'content-type': 'application/json' }); res.end(eo); } else res.writableEnded || res.end('data: ' + eo + '\n\ndata: [DONE]\n\n');
        report({ tin: 0, tout: 0, ms: Date.now() - t0, code: ur.statusCode, ok: false, err: msg || ('HTTP ' + ur.statusCode) });
      });
      return;
    }
    if (dna) return dnaReply(ur);
    // Text model with the drawing tool offered: the reply is inspected, and if it asked for a
    // picture the gateway draws it and glues it in. The client only ever sees one answer.
    if (drawer) {
      const sse = body.stream === true;
      if (sse && !res.headersSent) res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache, no-transform', connection: 'keep-alive', 'x-accel-buffering': 'no' });
      // All traffic crosses this path while a drawer is set, so reasoning and the client's tool calls must
      // be forwarded here or they are lost with the tokens already billed.
      // In SSE `said` is only a yes/no for the glue; the text itself already went out.
      // seen = deltas passed on (billed when the stream is cut before its usage); fr = the model's own finish reason;
      // tm = the engine's timings (kept in the final chunk); hold = the client is behind: upstream waits for 'drain'
      let buf = '', said = '', thought = '', calls = {}, u2 = null, tm = null, cid = '', started = false, ended = false, seen = 0, fr = '', hold = false;
      const dec = new StringDecoder('utf8'), created = Math.floor(Date.now() / 1000);
      const w = (s) => { if (!res.write(s)) hold = true; };
      const ch = (delta, fin) => 'data: ' + JSON.stringify({ id: cid || 'chatcmpl-img', object: 'chat.completion.chunk', created: created, model: target.id, choices: [{ index: 0, delta: delta, finish_reason: fin || null }] }) + '\n\n';
      const open = () => { if (!started) { started = true; w(ch({ role: 'assistant' })); } };
      // in SSE the reasoning goes out as it comes; only its tail is kept, which is all the stranded-call check needs
      const keep = (s) => { thought = (thought + s).slice(-24); };
      const eat = (o) => {
        if (!o) return;
        cid = cid || o.id || '';
        if (o.usage) u2 = o.usage;
        if (o.timings) tm = o.timings;
        const c = (o.choices || [])[0]; if (!c) return;
        if (c.finish_reason) fr = c.finish_reason;
        const d = c.delta || c.message || {};
        // reasoning keeps its original field name (reasoning_content or reasoning)
        const think = d.reasoning_content || d.reasoning || '';
        if (think) { if (!sse) thought += think; else { keep(think); open(); w(ch(d.reasoning_content ? { reasoning_content: think } : { reasoning: think })); } }
        if (d.content) { if (!sse) said += d.content; else { said = '1'; open(); w(ch({ content: d.content })); } }
        // a non-stream message has no index: its position is the key, or two calls merge into one
        (d.tool_calls || []).forEach((t, i) => {
          const k = t.index != null ? t.index : i;
          const cur = calls[k] || (calls[k] = { name: '', args: '' });
          if (t.function) { if (t.function.name) cur.name += t.function.name; if (t.function.arguments) cur.args += t.function.arguments; }
        });
      };
      ur.on('data', (dd) => {
        buf += dec.write(dd);   // a char split across chunks waits for its other half (no U+FFFD)
        if (!sse && !/^\s*data:/.test(buf)) return;   // plain JSON: parsed at the end
        let i;
        while ((i = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
          if (line.indexOf('data:')) continue;
          // text/thought deltas go out as sent, like the pipe (a thought leaves its tail): only lines that can carry a tool call, usage or finish are parsed
          if (sse && !/"tool_calls"|"usage":\s*\{|"finish_reason":\s*"|\[DONE\]/.test(line)) {
            const r = /"(content|reasoning(?:_content)?)":\s*("(?:[^"\\]|\\.)+")/.exec(line);
            if (r) { seen++; if (/"content":\s*"[^"]/.test(line)) said = '1'; else try { keep(JSON.parse(r[2])); } catch (_) {} }
            started = true; w(line + '\n\n'); continue;
          }
          const p = line.slice(5).trim(); if (p && p !== '[DONE]') { try { eat(JSON.parse(p)); } catch (_) {} }
        }
        if (hold) { hold = false; ur.pause(); res.once('drain', () => ur.resume()); }
      });
      ur.on('end', () => {
        if (ended) return; ended = true;
        buf += dec.end();
        if (!sse) { try { eat(JSON.parse(buf)); } catch (_) {} }
        over(tm);
        const wanted = Object.keys(calls).map((k) => calls[k]).filter((c) => c.name === TOOLS.image.fn);
        let tin = u2 ? (+u2.prompt_tokens || +u2.input_tokens || 0) : 0;
        let tout = u2 ? (+u2.completion_tokens || +u2.output_tokens || 0) : 0;
        const hit = cachedOf(u2);
        // the client's own tool calls pass through; only generate_image is intercepted
        const pass = Object.keys(calls).map((k) => calls[k]).filter((c) => c.name && c.name !== TOOLS.image.fn);
        // the engine's own word wins: a turn cut for length is 'length' even if it managed to emit a tool call
        const fin0 = fr === 'length' ? 'length' : pass.length ? 'tool_calls' : 'stop';
        // a tool call written INSIDE a <think> that never closed: the parser leaves it as reasoning, the tool never
        // runs and the turn answers nothing. Counted only; rescuing it needs a state machine.
        const stranded = fr === 'stop' && !said && !pass.length && !wanted.length && /<\/tool_call>\s*$/.test(thought) ? 1 : 0;
        const finish = (extra, imgs, usd) => {
          const text = said + (extra ? (said ? '\n\n' : '') + extra : ''), left = res.destroyed;
          const usage = { prompt_tokens: tin, completion_tokens: tout, total_tokens: tin + tout };
          if (hit) usage.prompt_tokens_details = { cached_tokens: hit };
          if (!res.headersSent) res.writeHead(200, Object.assign({ 'content-type': 'application/json' }, ov < 0 ? {} : { 'x-ias-overhead-ms': ov }));
          const tail = { id: cid || 'chatcmpl-img', object: 'chat.completion.chunk', created: created, model: target.id, choices: [], usage: usage };
          if (tm) tail.timings = tm;
          if (sse) {
            open();
            if (extra) res.write(ch({ content: (said ? '\n\n' : '') + extra }));
            if (pass.length) res.write(ch({ tool_calls: pass.map((c, i) => ({ index: i, id: 'call_' + i, type: 'function', function: { name: c.name, arguments: c.args } })) }));
            res.write(ch({}, fin0));
            res.write('data: ' + JSON.stringify(tail) + '\n\n');
            res.end('data: [DONE]\n\n' + (ov < 0 ? '' : ': x-ias-overhead-ms=' + ov + '\n\n'));
          } else {
            const msg = { role: 'assistant', content: text };
            if (thought) msg.reasoning_content = thought;
            if (pass.length) msg.tool_calls = pass.map((c, i) => ({ id: 'call_' + i, type: 'function', function: { name: c.name, arguments: c.args } }));
            res.end(JSON.stringify(Object.assign(tail, { object: 'chat.completion', choices: [{ index: 0, message: msg, finish_reason: fin0 }] })));
          }
          report(Object.assign({ tin: tin - hit, tout: tout, imgs: imgs || undefined, usd: (prices.costOf(target.id, u2, cfg) || 0) + (usd || 0), ms: Date.now() - t0, code: 200, ok: true, cut: left || undefined, str: stranded || undefined, fr: fr || undefined }, perfOf(tm)));
        };
        if (!wanted.length) return finish('', 0, 0);
        let args = {}; try { args = JSON.parse(wanted[0].args || '{}') || {}; } catch (_) {}
        // a call the engine cut for length carries half a JSON, and a picture nobody described is ~52 s of GPU for noise
        if (fr === 'length' || typeof args.prompt !== 'string' || !args.prompt.trim()) return finish('', 0, 0);
        // A local picture takes up to minutes: every 15 s an SSE comment, or a leading space before the JSON (valid JSON),
        // keeps proxies and idle-read timeouts from dropping the client meanwhile.
        const ka = setInterval(() => { if (res.destroyed) return; if (!sse && !res.headersSent) res.writeHead(200, { 'content-type': 'application/json' }); res.write(sse ? ': drawing\n\n' : ' '); }, 15000);
        // The drawing's input is paid too (9400 tokens on an edit): it lands on the same key.
        // say: the drawer names the job the moment it hands it to the image model; the picture chunk ends it.
        drawer.run(args, (r) => { clearInterval(ka); tin += r.tin; tout += r.tout; finish(r.md, r.imgs, r.usd); }, body,
          (s) => { if (sse && res.writable) { open(); res.write(ch({ status: s })); } });
      });
      // cut before its end (the client left or the provider died): nothing sent yet is a 502,
      // else the reply is broken and what the provider already counted is reported
      ur.on('close', () => { if (!ended) { ended = true; if (!gone && !res.headersSent) return fail('aborted'); res.destroy(); report(Object.assign({ tin: u2 ? (+u2.prompt_tokens || 0) - cachedOf(u2) : 0, tout: u2 ? +u2.completion_tokens || 0 : seen, ms: Date.now() - t0, code: 200, ok: true, cut: true }, perfOf(tm))); } });
      return;
    }
    // Image models: the whole reply is small and must be rewritten, so it is read before answering.
    if (isImageModel(cfg, target.id)) {
      let raw = '';
      ur.on('data', (d) => { if (raw.length < 262144) raw += d; });
      ur.on('end', () => {
        let j = null; try { j = JSON.parse(raw); } catch (_) {}
        const done = toOpenAI(j || {}, target.id);
        const u2 = j && j.usage;
        const tin = u2 ? (+u2.input_tokens || +u2.prompt_tokens || 0) : 0;
        const imgs = u2 ? (+u2.image_count || 0) : (done.choices[0].message.content.match(/!\[\]\(/g) || []).length;
        const tout = (u2 ? (+u2.output_tokens || 0) : 0)
                   + (prices.imageChips(imgs, u2 && u2.size, cfg.image_quality) || prices.imageTokens(target.id, imgs, cfg));
        done.usage.completion_tokens = tout; done.usage.total_tokens = tin + tout;
        if (!res.headersSent) {
          // the provider ignores stream:true, so SSE is built here for clients that asked for it
          if (body.stream === true) {
            res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache, no-transform', connection: 'keep-alive', 'x-accel-buffering': 'no' });
            const ch = (delta, fin) => 'data: ' + JSON.stringify({ id: done.id, object: 'chat.completion.chunk', created: done.created, model: target.id, choices: [{ index: 0, delta: delta, finish_reason: fin || null }] }) + '\n\n';
            res.write(ch({ role: 'assistant' }));
            res.write(ch({ content: done.choices[0].message.content }));
            res.write(ch({}, 'stop'));
            res.write('data: ' + JSON.stringify({ id: done.id, object: 'chat.completion.chunk', created: done.created, model: target.id, choices: [], usage: done.usage }) + '\n\n');
            res.write('data: [DONE]\n\n'); res.end();
          } else {
            res.writeHead(200, { 'content-type': 'application/json' });
            res.end(JSON.stringify(done));
          }
        }
        report({ tin: tin, tout: tout, imgs: imgs || undefined, usd: prices.costOf(target.id, u2, cfg), ms: Date.now() - t0, code: 200, ok: true });
      });
      ur.on('close', () => { if (!ur.complete) fail('aborted'); });
      return;
    }
    // A stream is piped as it comes (with the headers a proxy needs); its overhead figure can only follow it, an SSE
    // comment after [DONE] (parsers skip ': ' lines). A local whole reply is held (bounded) for the header; a cloud's is piped.
    const ct = ur.headers['content-type'] || 'application/json', sse = /event-stream/.test(ct), h = { 'content-type': ct }, whole = !sse && local ? [] : null;
    if (sse) { h['cache-control'] = 'no-cache'; h['x-accel-buffering'] = 'no'; }
    if (!whole) { if (!res.headersSent) res.writeHead(ur.statusCode, h); ur.pipe(res, { end: false }); }
    // only the last 8 KB are kept to read usage and timings
    let tail = '', seen = 0, closed = false, nb = 0;
    ur.on('data', (d) => {
      const t = d.toString('utf8');
      tail = (tail + t).slice(-8192);
      if (whole) { whole.push(d); if ((nb += d.length) > 16777216) ur.destroy(); }   // one JSON, never 16 MB
      // chunks are counted as they pass: a client that disconnects is still billed for what was generated
      if (t.indexOf('"content"') !== -1) { const m = t.match(/"content"/g); seen += m ? m.length : 0; }
    });
    const finish = (cut) => {
      if (closed) return; closed = true;
      const tm = tailObj(tail, '"timings"');
      over(tm);
      // a pipe does not end its target when the source dies (the client would wait forever); nothing sent yet is a 502
      // cut mid-stream with the client still there: the error event says so (billing below counts what went out)
      if (cut) { if (!res.headersSent && res.socket && !res.socket.destroyed) return fail('aborted'); if (sse && !gone) res.writableEnded || res.end('\n\ndata: ' + JSON.stringify({ error: { message: target.prov + ': the reply was cut before its end (the engine stopped); try again', type: 'api_error', param: null, code: 'upstream_error' } }) + '\n\ndata: [DONE]\n\n'); else res.destroy(); }
      else if (whole) { if (ov >= 0) h['x-ias-overhead-ms'] = ov; res.writeHead(ur.statusCode, h); res.end(Buffer.concat(whole)); }
      else res.end(sse && ov >= 0 ? ': x-ias-overhead-ms=' + ov + '\n\n' : undefined);
      note(target.prov, ur.statusCode, ur.statusCode >= 400 ? errMsg(tail) : '', ur.headers);
      const u2 = tailObj(tail, '"usage"');
      // Chat says prompt/completion_tokens; Images and Alibaba say input/output_tokens. Reading
      // only the first pair lost every input token of an image request (measured: 1274 -> 0).
      const tin = (u2 ? (+u2.prompt_tokens || +u2.input_tokens || 0) : 0) - cachedOf(u2);
      // Images are counted AS IMAGES, never converted: Alibaba bills per picture and says its
      // own tokens are "not billed" (output_tokens is a flat 2 per image, same for 1K and 4K).
      const imgs = u2 ? (+u2.image_count || 0) : 0;
      // The chip is the token: an image is converted by its price, so the quota spends one unit.
      const raw = u2 ? (+u2.completion_tokens || +u2.output_tokens || 0) : Math.max(0, seen - 1);
      // Same standard for every image AI. Only if the provider hides the size we fall back to
      // converting by price, and if there is no price either, the picture counts as 0; never guessed.
      const chips = prices.imageChips(imgs, u2 && u2.size, cfg.image_quality)
                 || prices.imageTokens(target.id, imgs, cfg);
      const tout = raw + chips;
      report(Object.assign({ tin: tin, tout: tout, imgs: imgs || undefined, usd: prices.costOf(target.id, u2, cfg), ms: Date.now() - t0,
        code: ur.statusCode, ok: ur.statusCode < 400, cut: cut || undefined, fr: (/[^]*"finish_reason":\s*"(\w+)"/.exec(tail) || [])[1],
        err: ur.statusCode >= 400 ? (errMsg(tail) || ('HTTP ' + ur.statusCode)) : undefined }, perfOf(tm)));
    };
    ur.on('end', () => finish(false));
    ur.on('close', () => finish(true));
    res.on('close', () => finish(true));
  });
  up.on('timeout', () => { up.destroy(new Error('timeout')); });
  // a local model about to be killed cuts the request: the engine keeps the prompt read so far, the client gets the error
  target.abort = () => up.destroy(new Error('model unloading'));
  if (dna) { const a0 = target.abort; target.abort = () => { a0(); dnaReqs.forEach((q) => q.destroy(new Error('model unloading'))); }; }
  // A client that leaves stops the generation nobody reads: a local GPU is freed, a provider stops billing.
  // Once the provider answered in full (a picture may still be drawing) there is nothing left to stop; a cloud picture
  // is billed by the provider anyway, so it is read to its end.
  res.on('close', () => { if (!res.writableEnded && !upEnd && !isImageModel(cfg, target.id)) { gone = true; up.destroy(); } });
  up.on('error', (e) => {
    if (gone) return report({ tin: 0, tout: 0, ms: Date.now() - t0, code: 200, ok: true, cut: true });
    // a pooled socket reset before any reply (an engine restarted, a cloud host dropped it idle): once more on a fresh one
    if (e.code === 'ECONNRESET' && up.reusedSocket && !up.res && !target.reset) { target.reset = 1; return relay(res, cfg, target, rawBody, onDone, retried); }
    note(target.prov, 0, e.message, null);
    // a reply already begun ends in its own 'close', which reports what was generated
    if (!up.res) fail(e.message);
  });

  // her DNA: the first reply read whole; no doubt, it goes out as it is; a doubt brings k-1 more thoughts and the surest answers
  function dnaReply(ur) {
    if (dnaSse) return dnaLive(ur);
    let raw = '';
    ur.on('data', (d) => { if (raw.length < 16777216) raw += d; });
    ur.on('end', () => {
      let first = null; try { first = JSON.parse(raw); } catch (_) {}
      if (!first || !first.choices || !first.choices[0]) return fail('unreadable reply');
      const why = dnaDoubt(first);
      if (!why || res.destroyed) return dnaSend(first, [first], 0, why);
      const n = Math.max(1, (+dna.k || 5) - 1), all = [first];
      let left = n;
      for (let i = 1; i <= n; i++) dnaOne(i, (j) => { if (j && j.choices && j.choices[0]) all.push(j); if (--left === 0) { const p = dnaPick(all); dnaSend(all[p], all, p, why); } });
    });
    ur.on('close', () => { if (!ur.complete) fail('aborted'); });
  }
  // her thought goes out live, counted by its deltas; past her doubt, her answer too; past the gate the thought stops and the extras start
  function dnaLive(ur) {
    if (!res.headersSent) res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache, no-transform', connection: 'keep-alive', 'x-accel-buffering': 'no' });
    const gate = +dna.gate || 800, msg = { content: '', reasoning_content: '' }, tc = {}, more = [];
    let buf = '', id = null, created = null, fin = null, usage = null, timings = null, pp = null, nth = 0, ntok = 0;
    let opened = false, pass = false, held = false, stop = false, sent = false, why = '', left = 0, ka = null;
    const out = (d) => { if (!sent && !res.destroyed) res.write('data: ' + JSON.stringify({ id: id, object: 'chat.completion.chunk', created: created, model: target.id, choices: [{ index: 0, delta: d, finish_reason: null }] }) + '\n\n'); };
    const first = () => {
      const calls = Object.keys(tc).sort((a, b) => a - b).map((k) => tc[k]);
      return { id: id, created: created, thought: nth, timings: timings, choices: [{ index: 0, message: calls.length ? Object.assign({ tool_calls: calls }, msg) : msg, finish_reason: fin || 'stop' }],
        usage: usage || Object.assign({ completion_tokens: ntok }, pp && { prompt_tokens: pp.total, prompt_tokens_details: { cached_tokens: pp.cache } }) };
    };
    // she doubts: k-1 more thoughts, whole, with their tails; once all are in, the surest answers
    const think = (w) => {
      why = w; left = Math.max(1, (+dna.k || 5) - 1);
      out({ status: 'Thinking again' });
      ka = setInterval(() => { if (!res.destroyed) res.write(': thinking again\n\n'); }, 5000);
      for (let i = 1, n = left; i <= n; i++) dnaOne(i, (j) => { if (j && j.choices && j.choices[0]) more.push(j); if (--left === 0) { sent = true; clearInterval(ka); const all = [first()].concat(more), p = dnaPick(all); dnaSend(all[p], all, p, why, 1); } });
    };
    res.once('close', () => clearInterval(ka));
    ur.on('data', (d) => {
      if (stop) return;
      buf += d; let i;
      while ((i = buf.indexOf('\n\n')) > -1) {
        const ev = buf.slice(0, i); buf = buf.slice(i + 2);
        const line = ev.split('\n').find((l) => l.startsWith('data: ')); if (!line || line.slice(6).trim() === '[DONE]') continue;
        let j; try { j = JSON.parse(line.slice(6)); } catch (_) { continue; }
        if (j.error) { stop = sent = true; up.destroy(); return fail(String(j.error.message || 'error')); }
        id = id || j.id; created = created || j.created; if (j.usage) usage = j.usage; if (j.timings) timings = j.timings;
        // the engine's reading: the app's meter when it asked, our bill always
        if (j.prompt_progress) { pp = j.prompt_progress; if (dnaPp) { if (!res.destroyed) res.write(ev + '\n\n'); } else if (!opened) out({ role: 'assistant' }); opened = true; continue; }
        const c = (j.choices || [])[0]; if (!c) continue;
        if (c.finish_reason) fin = c.finish_reason;
        const dl = c.delta || {};
        if (dl.role && !opened) { opened = true; out({ role: dl.role }); }
        if (dl.reasoning_content || dl.content || dl.tool_calls) ntok++;
        if (dl.reasoning_content) {
          msg.reasoning_content += dl.reasoning_content; out({ reasoning_content: dl.reasoning_content });
          if (++nth > gate && !pass) { stop = true; up.destroy(); return think('long'); }
        }
        if (dl.content) msg.content += dl.content;
        (dl.tool_calls || []).forEach((t) => { const o = tc[t.index || 0] = tc[t.index || 0] || { id: t.id, type: 'function', function: { name: '', arguments: '' } }; if (t.id) o.id = t.id; if (t.function) { o.function.name += t.function.name || ''; o.function.arguments += t.function.arguments || ''; } });
        if (!(dl.content || dl.tool_calls) || held) continue;
        // she began to answer: past her doubt unless her cap cut her thought (then it is decided at her end)
        if (!pass && (msg.content.trim() || dl.tool_calls)) { if (dnaDoubt(first())) { held = true; continue; } pass = true; if (msg.content) out({ content: msg.content }); if (dl.tool_calls) out({ tool_calls: dl.tool_calls }); continue; }
        if (pass) { if (dl.content) out({ content: dl.content }); if (dl.tool_calls) out({ tool_calls: dl.tool_calls }); }
      }
    });
    ur.on('end', () => {
      if (stop) return;
      if (pass) { sent = true; return dnaSend(first(), [first()], 0, '', 2); }
      const w = dnaDoubt(first());
      if (!w || res.destroyed) { sent = true; return dnaSend(first(), [first()], 0, w, 1); }
      think(w);
    });
    // cut before its end: a client that left is billed what ran (as a cut stream is), anything else is the engine's failure
    ur.on('close', () => { if (ur.complete || stop) return; if (res.destroyed) { sent = true; return dnaSend(first(), [first()], 0, why, 1); } fail('aborted'); });
  }
  // one more thought: the same request, another seed when the client fixed one (the same seed thinks the same thought)
  function dnaOne(i, cb) {
    let done = false; const end = (j) => { if (!done) { done = true; cb(j); } };
    const pl = Buffer.from(JSON.stringify(Object.assign({}, body, { stream: false, stream_options: undefined, return_progress: undefined, logprobs: true, top_logprobs: Math.max(5, +body.top_logprobs || 0) }, body.seed == null ? {} : { seed: (+body.seed || 0) + i })));
    const q = mod.request(u, { method: 'POST', headers: { 'authorization': 'Bearer ' + target.key, 'content-type': 'application/json', 'content-length': pl.length, 'accept': 'application/json' },
      timeout: cfg.timeout_ms, agent: local ? localAgent : mod === https ? cloudAgent : undefined }, (u3) => {
      let raw = ''; u3.on('data', (d) => { if (raw.length < 16777216) raw += d; });
      u3.on('end', () => { let j = null; try { j = u3.statusCode < 400 ? JSON.parse(raw) : null; } catch (_) {} end(j); });
      u3.on('close', () => end(null));
    });
    dnaReqs.push(q);
    q.on('error', () => end(null)); q.on('timeout', () => q.destroy(new Error('timeout')));
    res.once('close', () => { if (!res.writableEnded) q.destroy(); });
    q.end(pl);
  }
  // her doubt: 'empty' (she ended inside her thought), 'cap' (the server's thinking cap cut it), 'long' (thought past the gate)
  function dnaDoubt(j) {
    const c = j.choices[0], m = c.message || {}, said = String(m.content || '').trim(), calls = m.tool_calls && m.tool_calls.length;
    if (!said && !calls) return 'empty';
    if (body.reasoning_budget_message && String(m.reasoning_content || '').indexOf(String(body.reasoning_budget_message).trim()) > -1) return 'cap';
    return dnaThought(j) > (+dna.gate || 800) ? 'long' : '';
  }
  // her thought's tokens: the log-probabilities cover thought and answer, the thought ends at its </think>; a live one was counted
  function dnaThought(j) {
    if (j.thought != null) return j.thought;
    const lp = ((j.choices[0].logprobs || {}).content) || []; let acc = '';
    for (let i = 0; i < lp.length; i++) { acc += lp[i].token || ''; if (acc.indexOf('</think>') > -1) return i; }
    return lp.length;
  }
  // the surest among the thoughts that answered (not cut by length): the mean confidence of its last 64 tokens
  function dnaTail(j) {
    const lp = ((j.choices[0].logprobs || {}).content) || [];
    const w = lp.slice(-64).map((t) => { const tl = t.top_logprobs || []; return tl.length ? -tl.reduce((a, x) => a + (+x.logprob || 0), 0) / tl.length : 0; });
    return w.length ? w.reduce((a, b) => a + b, 0) / w.length : -Infinity;
  }
  function dnaPick(all) {
    let best = 0, bt = -Infinity;
    all.forEach((j, i) => { const c = j.choices[0], m = c.message || {}; if (c.finish_reason === 'length' || (!String(m.content || '').trim() && !(m.tool_calls && m.tool_calls.length))) return; const t = dnaTail(j); if (t > bt) { bt = t; best = i; } });
    return best;
  }
  // every thought is billed, the prompt once; live: what a stream already sent (1 her first thought, 2 her answer too)
  function dnaSend(j, all, pick, why, live) {
    const c0 = j.choices[0], msg = Object.assign({}, c0.message || {}), fin = c0.finish_reason || 'stop';
    const u0 = (all.find((x) => x.usage && x.usage.prompt_tokens != null) || all[0]).usage || {}, hit = cachedOf(u0), tout = all.reduce((a, x) => a + (+((x.usage || {}).completion_tokens) || 0), 0);
    const usage = { prompt_tokens: +u0.prompt_tokens || 0, completion_tokens: tout, total_tokens: (+u0.prompt_tokens || 0) + tout };
    if (hit) usage.prompt_tokens_details = { cached_tokens: hit };
    const meta = { doubt: why || null, thoughts: all.length, pick: pick };
    const report2 = () => report(Object.assign({ tin: (+u0.prompt_tokens || 0) - hit, tout: tout, ms: Date.now() - t0, code: 200, ok: true, fr: fin, cut: res.destroyed || undefined }, perfOf(j.timings)));
    if (res.destroyed) return report2();
    const id = all[0].id || j.id || 'chatcmpl-dna', created = all[0].created || j.created || Math.floor(Date.now() / 1000);
    if (dnaSse) {
      if (!res.headersSent) res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache, no-transform', connection: 'keep-alive', 'x-accel-buffering': 'no' });
      const ch = (delta, f) => 'data: ' + JSON.stringify({ id: id, object: 'chat.completion.chunk', created: created, model: target.id, choices: [{ index: 0, delta: delta, finish_reason: f || null }] }) + '\n\n';
      if (!live) { res.write(ch({ role: 'assistant' })); if (msg.reasoning_content) res.write(ch({ reasoning_content: msg.reasoning_content })); }
      else if (pick && msg.reasoning_content) res.write(ch({ reasoning_content: '\n\n' + msg.reasoning_content }));
      if (live !== 2) {
        if (msg.content) res.write(ch({ content: msg.content }));
        if (msg.tool_calls && msg.tool_calls.length) res.write(ch({ tool_calls: msg.tool_calls.map((x, i) => Object.assign({ index: i }, x)) }));
      }
      res.write(ch({}, fin));
      res.write('data: ' + JSON.stringify({ id: id, object: 'chat.completion.chunk', created: created, model: target.id, choices: [], usage: usage, timings: j.timings, ias_dna: meta }) + '\n\n');
      res.end('data: [DONE]\n\n');
    } else {
      const c1 = Object.assign({}, c0, { message: msg }); if (!dnaLp) delete c1.logprobs;
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify(Object.assign({}, j, { model: target.id, choices: [c1], usage: usage, ias_dna: meta })));
    }
    report2();
  }
  up.write(payload); up.end();
}

// Prompt tokens served from cache (OpenAI and llama-server report them apart): reused, not processed, so not billed.
const cachedOf = (u) => +((u && u.prompt_tokens_details) || {}).cached_tokens || 0;

// The engine's own numbers for this turn (server-common.cpp:84, server_slot_stats::to_json): what the prompt cache
// served, what it read and in how long, how fast it WROTE, and the MTP draft score. draft_* only exist when it drafted at all.
const perfOf = (tm) => !tm ? null : { cn: +tm.cache_n || 0, pn: +tm.prompt_n || 0, pms: +tm.prompt_ms || 0,
  wtps: tm.predicted_per_second != null ? Math.round(tm.predicted_per_second * 10) / 10 : null,
  dn: +tm.draft_n || 0, da: +tm.draft_n_accepted || 0 };

// The last object under key ("usage", "timings") in a response tail: the final JSON or the last chunks of a stream.
function tailObj(txt, key) {
  let best = null, i = -1;
  while ((i = txt.indexOf(key, i + 1)) !== -1) {
    const a = txt.indexOf('{', i + key.length);
    if (a < 0) continue;
    let d = 0;
    for (let j = a; j < txt.length && j < a + 600; j++) {
      if (txt[j] === '{') d++;
      else if (txt[j] === '}') { d--; if (!d) { try { best = JSON.parse(txt.slice(a, j + 1)); } catch (_) {} break; } }
    }
  }
  return best;
}

// Synchronous list for the panel from what is already known; stale providers refresh in the background.
function panelModels(cfg) {
  if (!catalog) return [];
  const out = [];
  Object.keys(cfg.providers).forEach((prov) => {
    const p = cfg.providers[prov], meta = catalog[prov];
    if (!p || p.enabled === false || !p.key || !meta) return;
    const fixed = Array.isArray(p.models) && p.models.length ? p.models : null;
    const live = modelsCache[prov];
    if (!live || Date.now() - live.at > MODELS_TTL) providerModels(cfg, prov, () => {});
    const ids = mergeIds(fixed, live);
    ids.forEach((m) => out.push({
      name: prov + '/' + m, gb: 0, params: '', quant: '', fam: '', caps: [],
      ctx: ctxOf(prov, m, live), ctxSrc: ctxInfo(prov, m, live).src, cloud: true,
      // grouped per provider in the panel, with its health; draws = can pair with a text model,
      // speaks = speech (tts / realtime). The panel groups by these, it does not guess from the name.
      prov: prov, provLabel: meta.label || prov, state: (health[prov] || {}).state || '',
      draws: isImageModel(cfg, prov + '/' + m), speaks: /audio|tts|realtime|speech|voice/i.test(m),
    }));
  });
  return out;
}

// On demand only (owner's keys). Step 1, /models, is free and catches a rejected key; step 2 spends one
// token and catches no balance (zai answers 200 on /models and 429 on generation).
function probe(cfg, prov, cb) {
  const p = cfg.providers[prov] || {}, meta = (catalog && catalog[prov]) || {};
  const baseUrl = String(p.baseUrl || meta.baseUrl || '').replace(/\/+$/, '');
  if (!p.key || !baseUrl) return cb({ at: Date.now(), code: 0, state: 'no_key', msg: '', limit: {} });
  const auth = { accept: 'application/json', authorization: 'Bearer ' + p.key };
  fetchRaw(baseUrl + '/models', auth, null, (e, r, body) => {
    if (e) return cb(note(prov, 0, e.message, null));
    if (r.statusCode >= 400) return cb(note(prov, r.statusCode, errMsg(body), r.headers));
    let first = (Array.isArray(p.models) && p.models[0]) || '';
    if (!first) { try { const j = JSON.parse(body), a = j.data || j.models || []; first = (a[0] && (a[0].id || a[0].name)) || ''; } catch (_) {} }
    if (!first) return cb(note(prov, r.statusCode, '', r.headers));   // valid key, no model to test
    const one = Buffer.from(JSON.stringify({ model: first, messages: [{ role: 'user', content: 'hi' }], max_tokens: 1, stream: false }));
    fetchRaw(baseUrl + '/chat/completions', auth, one, (e2, r2, b2) => {
      if (e2) return cb(note(prov, 0, e2.message, null));
      cb(note(prov, r2.statusCode, r2.statusCode >= 400 ? errMsg(b2) : '', r2.headers));
    });
  });
}

const toolList = () => Object.keys(TOOLS).map((id) => ({ id: id, label: TOOLS[id].label }));

module.exports = {
  options: options, resolve: resolve, listModels: listModels,
  relay: relay, ctxOf: ctxOf, toolList: toolList, providerModels: providerModels, panelModels: panelModels,
  toolCfg: toolCfg, draw: draw, isImageModel: isImageModel,
  catalog: () => catalog, ctxInfo: ctxInfo,
  modelCount: (prov) => (modelsCache[prov] && modelsCache[prov].ids.length) || 0, health: () => health, probe: probe,
};
