'use strict';
// Anthropic's Messages door (/v1/messages, /v1/messages/count_tokens) over the gateway's own OpenAI door, by loopback:
// the same key, quota, thinking brake and pulse as any client. Claude Code: ANTHROPIC_BASE_URL=<this server>,
// ANTHROPIC_AUTH_TOKEN=<key>, ANTHROPIC_MODEL=<a model id of /v1/models>. Server tools (web_search...) do not run here.
const http = require('http'), crypto = require('crypto');

const ERR = { 400: 'invalid_request_error', 401: 'authentication_error', 402: 'billing_error', 403: 'permission_error', 404: 'not_found_error', 413: 'request_too_large', 429: 'rate_limit_error', 503: 'overloaded_error', 504: 'timeout_error', 529: 'overloaded_error' };
const STOP = { stop: 'end_turn', length: 'max_tokens', tool_calls: 'tool_use', function_call: 'tool_use', content_filter: 'refusal' };
const PASS = ['x-should-retry', 'retry-after', 'x-ratelimit-limit-requests', 'x-ratelimit-remaining-requests', 'x-ratelimit-reset-requests'];   // the client's retry decision
const rid = () => crypto.randomBytes(12).toString('hex');

function err(res, code, msg, h) {
  if (res.headersSent) return res.end();
  const b = JSON.stringify({ type: 'error', error: { type: ERR[code] || 'api_error', message: String(msg || 'Error.') } });
  res.writeHead(code, Object.assign({ 'content-type': 'application/json', 'content-length': Buffer.byteLength(b) }, h)); res.end(b);
}
// text of a content (string or blocks): text, text documents, search results; a PDF or a file is named, never dropped silently
function text(c) {
  if (typeof c === 'string') return c;
  if (!Array.isArray(c)) return '';
  return c.map((b) => !b ? '' : b.type === 'text' ? b.text : b.type === 'search_result' ? text(b.content)
    : b.type === 'document' ? (b.source && b.source.type === 'text' ? b.source.data : b.source && b.source.type === 'content' ? text(b.source.content) : '[' + ((b.source || {}).media_type || 'document') + ' not readable by this model]') : '').filter(Boolean).join('\n');
}
const img = (b) => !b || b.type !== 'image' || !b.source ? null : b.source.type === 'base64' ? { type: 'image_url', image_url: { url: 'data:' + b.source.media_type + ';base64,' + b.source.data } }
  : b.source.type === 'url' ? { type: 'image_url', image_url: { url: b.source.url } } : { type: 'text', text: '[image source ' + b.source.type + ' not supported]' };

// request: Anthropic -> OpenAI
function toOpenAI(a) {
  const o = { model: a.model, messages: [] }, sys = text(a.system);
  if (sys) o.messages.push({ role: 'system', content: sys });
  (Array.isArray(a.messages) ? a.messages : []).forEach((m) => {
    const c = typeof m.content === 'string' ? [{ type: 'text', text: m.content }] : Array.isArray(m.content) ? m.content : [];
    if (m.role === 'assistant') {   // its thinking goes back as reasoning_content: the model's template keeps what a tool loop needs
      const calls = c.filter((b) => b.type === 'tool_use').map((b) => ({ id: b.id, type: 'function', function: { name: b.name, arguments: JSON.stringify(b.input || {}) } }));
      const x = { role: 'assistant', content: c.filter((b) => b.type === 'text').map((b) => b.text).join('') }, th = c.filter((b) => b.type === 'thinking').map((b) => b.thinking).join('');
      if (th) x.reasoning_content = th;
      if (calls.length) x.tool_calls = calls;
      return o.messages.push(x);
    }
    const rest = [];   // a user turn: its tool results first (one 'tool' message each), then its text, documents and images, theirs included
    c.forEach((b) => {
      if (b.type === 'tool_result') {
        o.messages.push({ role: 'tool', tool_call_id: b.tool_use_id, content: (b.is_error ? 'Error: ' : '') + text(b.content) });
        if (Array.isArray(b.content)) b.content.forEach((x) => { const p = img(x); if (p) rest.push(p); });
      } else if (b.type === 'image') rest.push(img(b));
      else { const t = text([b]); if (t) rest.push({ type: 'text', text: t }); }
    });
    if (rest.length) o.messages.push({ role: 'user', content: rest.every((p) => p.type === 'text') ? rest.map((p) => p.text).join('\n') : rest });
  });
  const tl = (a.tools || []).filter((t) => t && t.name && (!t.type || t.type === 'custom'));
  if (tl.length) o.tools = tl.map((t) => ({ type: 'function', function: { name: t.name, description: t.description || '', parameters: t.input_schema || { type: 'object', properties: {} } } }));
  const tc = a.tool_choice || {};
  if (tl.length && tc.type) o.tool_choice = tc.type === 'any' ? 'required' : tc.type === 'tool' ? { type: 'function', function: { name: tc.name } } : tc.type;
  if (tc.disable_parallel_tool_use) o.parallel_tool_calls = false;
  ['max_tokens', 'temperature', 'top_p', 'top_k'].forEach((k) => { if (a[k] != null) o[k] = a[k]; });
  if (Array.isArray(a.stop_sequences) && a.stop_sequences.length) o.stop = a.stop_sequences;
  // thinking: enabled = on with the client's budget; disabled = off; absent or adaptive = the model's catalog decides
  const th = a.thinking || {}, ef = (a.output_config || {}).effort;
  if (th.type === 'enabled') { o.reasoning_effort = ef || 'high'; if (+th.budget_tokens > 0) o.reasoning_budget_tokens = +th.budget_tokens; }
  else if (th.type === 'disabled') o.reasoning_effort = 'none';
  else if (ef) o.reasoning_effort = ef;
  if (a.stream) { o.stream = true; o.stream_options = { include_usage: true }; }
  return o;
}
const args = (s) => { try { const v = JSON.parse(s || '{}'); return v && typeof v === 'object' ? v : {}; } catch (_) { return {}; } };
function use(u) {
  u = u || {}; const k = +((u.prompt_tokens_details || {}).cached_tokens) || 0;
  return { input_tokens: Math.max(0, (+u.prompt_tokens || 0) - k), output_tokens: +u.completion_tokens || 0, cache_read_input_tokens: k, cache_creation_input_tokens: 0 };
}
// reply: OpenAI -> Anthropic
function fromOpenAI(j, model) {
  const ch = (j.choices || [])[0] || {}, m = ch.message || {}, c = [], r = m.reasoning_content || m.reasoning;
  if (r) c.push({ type: 'thinking', thinking: r, signature: '' });
  if (m.content) c.push({ type: 'text', text: m.content });
  (m.tool_calls || []).forEach((t) => c.push({ type: 'tool_use', id: t.id || 'toolu_' + rid(), name: (t.function || {}).name, input: args((t.function || {}).arguments) }));
  const tools = c.some((b) => b.type === 'tool_use');
  return { id: 'msg_' + rid(), type: 'message', role: 'assistant', model: j.model || model, content: c, stop_reason: tools && ch.finish_reason !== 'length' ? 'tool_use' : STOP[ch.finish_reason] || 'end_turn', stop_sequence: null, usage: use(j.usage) };
}
// an upstream error in Anthropic's words: a context overflow reads 'prompt is too long' (what Claude Code compacts on)
function upErr(e, a) {
  e = e || {};
  if (e.code === 'context_length_exceeded' || /exceeds? the (available )?context/i.test(e.message || '')) return 'prompt is too long: ' + (e.n_prompt_tokens || count(a)) + ' tokens' + (e.n_ctx ? ' > ' + e.n_ctx + ' maximum' : '');
  return e.message;
}
// stream: OpenAI chunks -> Anthropic events, one open content block at a time; a ping every 15 s while it waits or thinks
function stream(res, model, ms) {
  let i = -1, kind = '', id = '', tools = false, stop = null, usage = null, on = false, done = false, seen = false, tick = null;
  const ev = (t, d) => { if (!done) res.write('event: ' + t + '\ndata: ' + JSON.stringify(Object.assign({ type: t }, d)) + '\n\n'); };
  const start = () => {
    if (on || done) return; on = true;
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    ev('message_start', { message: { id: 'msg_' + rid(), type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } } });
  };
  const close = () => { if (kind) ev('content_block_stop', { index: i }); kind = ''; id = ''; };
  const open = (k, b) => { close(); i++; kind = k; ev('content_block_start', { index: i, content_block: b }); };
  const stopAll = () => { done = true; clearInterval(tick); };
  return {
    start,
    get done() { return done; },
    wait() { tick = setInterval(() => { start(); ev('ping', {}); }, ms || 15000); },   // queue, prefill and long thoughts stay alive
    kill: () => stopAll(),
    chunk(j) {
      if (done) return; start();
      if (j.usage) usage = j.usage;
      const ch = (j.choices || [])[0]; if (!ch) return;
      const d = ch.delta || {}, r = d.reasoning_content || d.reasoning;
      if (r) { if (kind !== 'thinking') open('thinking', { type: 'thinking', thinking: '', signature: '' }); ev('content_block_delta', { index: i, delta: { type: 'thinking_delta', thinking: r } }); }
      if (d.content) { if (kind !== 'text') open('text', { type: 'text', text: '' }); ev('content_block_delta', { index: i, delta: { type: 'text_delta', text: d.content } }); }
      (d.tool_calls || []).forEach((t, n) => {
        const f = t.function || {}, k = 'tool' + (t.index != null ? t.index : n);
        if (kind !== k || (t.id && t.id !== id)) { open(k, { type: 'tool_use', id: t.id || 'toolu_' + rid(), name: f.name || '', input: {} }); id = t.id || ''; tools = true; }
        if (f.arguments) ev('content_block_delta', { index: i, delta: { type: 'input_json_delta', partial_json: f.arguments } });
      });
      if (ch.finish_reason) { seen = true; stop = tools && ch.finish_reason !== 'length' ? 'tool_use' : STOP[ch.finish_reason] || 'end_turn'; }
    },
    fin() { seen = true; },   // [DONE]
    end() {
      if (done) return;
      if (!seen) return this.fail(502, 'upstream stream cut before its end');   // a truncated reply is never passed off as complete
      start(); close(); ev('message_delta', { delta: { stop_reason: stop || (tools ? 'tool_use' : 'end_turn'), stop_sequence: null }, usage: use(usage) }); ev('message_stop', {}); stopAll(); res.end();
    },
    fail(code, msg, h) { if (done) return; if (!on) { stopAll(); return err(res, code, msg, h); } close(); ev('error', { error: { type: ERR[code] || 'api_error', message: String(msg) } }); stopAll(); res.end(); }
  };
}
// count_tokens: an estimate (about 3.5 characters a token; an image or a PDF ~1,500); the engine counts for real in every reply's usage
const count = (a) => Math.ceil(JSON.stringify([a.system || '', a.messages || [], a.tools || []], (k, v) => k === 'data' && typeof v === 'string' && v.length > 2000 ? 'x'.repeat(5250) : v).length / 3.5);

// o = { port, max, ok(key), key, host?, ping? }: the OpenAI door (loopback by default), the body limit, the key check of server.js
function serve(req, res, p, o) {
  if (!o.ok(o.key)) { req.resume(); return err(res, 401, 'Invalid or missing API key.'); }   // before reading a byte, like the OpenAI door
  const c = []; let n = 0, big = false;
  req.on('data', (d) => {
    if (big) return; n += d.length;
    if (n > o.max) { big = true; res.on('finish', () => req.destroy()); return err(res, 413, 'Request body too large. Maximum is ' + (o.max / 1048576) + ' MB.', { connection: 'close' }); }
    c.push(d);
  });
  req.on('end', () => {
    if (big) return;
    let a = null; try { a = JSON.parse(Buffer.concat(c).toString('utf8')); } catch (_) {}
    if (!a || typeof a !== 'object' || Array.isArray(a)) return err(res, 400, 'Request body must be a JSON object.');
    if (/count_tokens/.test(p)) { const b = JSON.stringify({ input_tokens: count(a) }); res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(b) }); return res.end(b); }
    if (typeof a.model !== 'string' || !a.model) return err(res, 400, 'model: Field required');
    const b = Buffer.from(JSON.stringify(toOpenAI(a))), s = a.stream ? stream(res, a.model, o.ping) : null;
    const up = http.request({ host: o.host || '127.0.0.1', port: o.port, path: '/v1/chat/completions', method: 'POST', headers: {
      'content-type': 'application/json', 'content-length': b.length, authorization: 'Bearer ' + o.key, host: req.headers.host || '127.0.0.1',
      'x-forwarded-proto': req.headers['x-forwarded-proto'] || (req.socket.encrypted ? 'https' : 'http'),
      'user-agent': String(req.headers['user-agent'] || '').replace(/[\r\n]/g, '').slice(0, 48) + ' (messages)',
      'x-forwarded-for': String(req.headers['x-forwarded-for'] || req.socket.remoteAddress || '').split(',')[0].replace(/[\r\n]/g, '').trim().replace(/^::ffff:/, '') } }, (r) => {
      r.setEncoding('utf8');
      let buf = '';
      const h = {}; PASS.forEach((k) => { if (r.headers[k] != null) h[k] = r.headers[k]; });
      // an upstream that dies after its headers (the engine died, was cut or unloaded) never ends quietly
      r.on('close', () => { if (!r.complete && !res.writableEnded) (s ? s.fail(502, 'upstream aborted') : err(res, 502, 'upstream aborted')); });
      if (r.statusCode !== 200 || !s || /json/.test(String(r.headers['content-type'] || ''))) {
        r.on('data', (d) => { buf += d; });
        return r.on('end', () => {
          const j = args(buf);
          if (r.statusCode !== 200) { const m = upErr(j.error, a) || buf.slice(0, 300); return s ? s.fail(r.statusCode, m, h) : err(res, r.statusCode, m, h); }
          if (!s) { const x = JSON.stringify(fromOpenAI(j, a.model)); res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(x) }); return res.end(x); }
          const ch = (j.choices || [])[0] || {}, m = ch.message || {};   // a whole reply to a stream request: replayed as one chunk
          s.chunk({ choices: [{ delta: Object.assign({}, m, { tool_calls: (m.tool_calls || []).map((t, n) => Object.assign({ index: n }, t)) }), finish_reason: ch.finish_reason || 'stop' }], usage: j.usage });
          s.end();
        });
      }
      s.start();
      r.on('data', (d) => {
        buf = (buf + d).replace(/\r/g, ''); let k;
        while ((k = buf.indexOf('\n\n')) >= 0 && !s.done) {
          const e = buf.slice(0, k); buf = buf.slice(k + 2);
          e.split('\n').forEach((l) => {
            if (s.done || l.indexOf('data:') !== 0) return;
            const v = l.slice(5).trim(); if (!v) return;
            if (v === '[DONE]') return s.fin();
            const j = args(v);
            if (j.error) { s.fail(500, upErr(j.error, a) || 'upstream error'); up.destroy(); } else s.chunk(j);
          });
        }
        if (buf.length > 4194304) { s.fail(502, 'upstream event too large'); up.destroy(); }   // an event never ends: no unbounded buffer
      });
      r.on('end', () => s.end());
    });
    if (s) s.wait();
    res.on('close', () => { if (s) s.kill(); if (!res.writableEnded) up.destroy(); });
    up.on('error', (e) => { if (!res.writableEnded) (s ? s.fail(502, e.message) : err(res, 502, e.message)); });
    up.end(b);
  });
}
module.exports = { serve, toOpenAI, fromOpenAI, stream, count };
