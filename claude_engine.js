'use strict';
/*
 * claude_engine.js: Claude Code (`claude -p --output-format stream-json`) as a text-only model on the
 * owner's subscription (measured apiKeySource "none", overage "rejected": no pay-per-use by accident).
 * All tools are disallowed. Stateless: the OpenAI history is flattened into one prompt per request
 * (no --resume, which would also count the context twice).
 * Never --bare: it forces ANTHROPIC_API_KEY auth and ignores the subscription.
 * cwd is outside the project, so no CLAUDE.md is auto-loaded.
 * The CLI exposes no temperature/top_p/seed/max_tokens: they are accepted and ignored.
 */

const { spawn } = require('child_process');

// disallowed tools are not even shown to the model, so it never waits for a permission prompt
const NO_TOOLS = [
  'Bash', 'BashOutput', 'KillShell', 'Edit', 'Write', 'NotebookEdit', 'Read', 'Glob', 'Grep',
  'WebFetch', 'WebSearch', 'Task', 'Agent', 'TodoWrite', 'SlashCommand', 'ExitPlanMode',
];

const DEF = {
  enabled: false,
  id: 'claude-code',
  bin: 'claude',
  model: 'sonnet',            // alias (sonnet/opus/haiku) or full id
  context_length: 200000,
  timeout_ms: 300000,
  cwd: '',                    // empty = user home, outside the project
  system: "You are a helpful, direct conversational assistant. Reply in the user's language.",
};

function options(raw) {
  const c = Object.assign({}, DEF, raw || {});
  c.id = String(c.id || DEF.id);
  c.bin = String(c.bin || DEF.bin);
  c.timeout_ms = Math.max(10000, +c.timeout_ms || DEF.timeout_ms);
  c.context_length = Math.max(1, +c.context_length || DEF.context_length);
  if (!c.cwd) c.cwd = process.env.USERPROFILE || process.env.HOME || undefined;
  return c;
}

// OpenAI content -> text. Images cannot go on a command line, so they are marked as omitted.
function textOf(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.map(function (p) {
      if (!p) return '';
      if (typeof p === 'string') return p;
      if (p.type === 'text') return String(p.text || '');
      if (p.type === 'image_url' || p.type === 'image') return '[image omitted: text-only engine]';
      return '';
    }).filter(Boolean).join('\n');
  }
  return content == null ? '' : String(content);
}

function flatten(messages) {
  const list = Array.isArray(messages) ? messages : [];
  const sys = [];
  const turns = [];
  list.forEach(function (m) {
    if (!m || !m.role) return;
    const t = textOf(m.content).trim();
    if (!t) return;
    if (m.role === 'system') { sys.push(t); return; }
    turns.push({ role: m.role, text: t });
  });
  // a single user turn is sent bare, without role labels
  let prompt;
  if (turns.length === 1 && turns[0].role === 'user') {
    prompt = turns[0].text;
  } else {
    prompt = turns.map(function (t) {
      const label = t.role === 'assistant' ? 'Assistant' : (t.role === 'user' ? 'User' : t.role);
      return label + ': ' + t.text;
    }).join('\n\n');
    if (turns.length && turns[turns.length - 1].role === 'user') prompt += '\n\nAssistant:';
  }
  return { system: sys.join('\n\n'), prompt: prompt };
}

// stream-json (NDJSON), buffered to '\n' because a chunk can split a line:
//   {type:"stream_event",event:{type:"content_block_delta"}} -> live token
//   {type:"result",result,usage,total_cost_usd}              -> final text and usage
function run(cfg, input, onToken, ctl) {
  return new Promise(function (resolve, reject) {
    const prompt = String(input.prompt || '');
    if (!prompt.trim()) return reject(new Error('empty prompt'));

    const a = ['-p', prompt, '--output-format', 'stream-json', '--verbose', '--include-partial-messages'];
    a.push('--disallowedTools'); NO_TOOLS.forEach(function (t) { a.push(t); });
    a.push('--permission-mode', 'dontAsk');

    // The system prompt is replaced, not appended. Measured on "say OK": default prompt 1131 output
    // tokens and $0.354; own prompt 4 tokens and $0.044 (the agentic harness makes the model plan).
    const sysPrompt = [cfg.system, input.system].filter(Boolean).join('\n\n') || DEF.system;
    a.push('--system-prompt', sysPrompt);
    a.push('--exclude-dynamic-system-prompt-sections');   // no date/git/environment sections
    const model = input.model || cfg.model;
    if (model) a.push('--model', String(model));

    let settled = false, timer = null, proc = null;
    function done(fn, v) { if (settled) return; settled = true; if (timer) clearTimeout(timer); fn(v); }

    // hard timeout: a client must never wait forever
    timer = setTimeout(function () {
      try { if (proc) proc.kill(); } catch (_) {}
      done(reject, new Error('claude timed out after ' + Math.round(cfg.timeout_ms / 1000) + 's'));
    }, cfg.timeout_ms);

    try { proc = spawn(cfg.bin, a, { env: process.env, cwd: cfg.cwd, windowsHide: true }); }
    catch (e) { return done(reject, new Error('claude spawn: ' + e.message)); }
    // the caller kills the process when the client disconnects, or it keeps spending quota
    if (ctl) ctl.kill = function () { try { proc.kill(); } catch (_) {} };

    let buf = '', text = '', acc = '', usage = null, cost = 0, gotResult = false, err = '';
    proc.stdout.on('data', function (d) {
      buf += d;
      let i;
      while ((i = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 1);
        if (!line.trim()) continue;
        let m; try { m = JSON.parse(line); } catch (_) { continue; }
        if (m.type === 'stream_event' && m.event && m.event.type === 'content_block_delta'
            && m.event.delta && m.event.delta.type === 'text_delta' && m.event.delta.text) {
          acc += m.event.delta.text;
          if (onToken) { try { onToken(m.event.delta.text); } catch (_) {} }
        } else if (m.type === 'result') {
          gotResult = true;
          usage = m.usage || null;
          cost = m.total_cost_usd || 0;
          text = (typeof m.result === 'string' && m.result) ? m.result : acc;
        }
      }
    });
    proc.stderr.on('data', function (d) { if (err.length < 4096) err += d; });
    proc.on('error', function (e) { done(reject, new Error('claude error: ' + e.message)); });
    proc.on('close', function (code) {
      // killed before `result`: usage is estimated from the streamed text (~4 chars per token), never zero
      if (!usage && acc) usage = { output_tokens: Math.ceil(acc.length / 4), estimated: true };
      if (gotResult || acc) done(resolve, { text: text || acc, usage: usage, costUsd: cost });
      else done(reject, new Error('claude exit ' + code + (err ? ': ' + err.replace(/\s+/g, ' ').slice(0, 200) : '')));
    });
  });
}

// Claude usage -> OpenAI usage
function toUsage(u) {
  if (!u) return null;
  const tin = (+u.input_tokens || 0) + (+u.cache_read_input_tokens || 0) + (+u.cache_creation_input_tokens || 0);
  const tout = +u.output_tokens || 0;
  return { prompt_tokens: tin, completion_tokens: tout, total_tokens: tin + tout, prompt_tokens_details: { cached_tokens: +u.cache_read_input_tokens || 0 } };
}
// input billed: cache reads were not processed again
const billed = (u) => u ? u.prompt_tokens - u.prompt_tokens_details.cached_tokens : 0;

// true when the claude CLI is installed
function available(cfg, cb) {
  cb = cb || function () {};
  let ended = false, proc = null, out = '';
  function finish(ok) { if (ended) return; ended = true; try { if (proc) proc.kill(); } catch (_) {} cb(!!ok); }
  const t = setTimeout(function () { finish(false); }, 8000);
  try { proc = spawn(cfg.bin, ['--version'], { env: process.env, windowsHide: true }); }
  catch (_) { clearTimeout(t); return finish(false); }
  proc.stdout.on('data', function (d) { out += d; });
  proc.on('error', function () { clearTimeout(t); finish(false); });   // ENOENT = not installed
  proc.on('close', function (code) { clearTimeout(t); finish(code === 0 || /\d+\.\d+/.test(out)); });
}

// POST /v1/chat/completions. onDone({tin,tout,ms,code,ok,err}) reports usage; quotas and keys live in the server.
function serve(res, cfg, rawBody, onDone) {
  const t0 = Date.now();
  let body = {};
  try { body = JSON.parse(rawBody.toString('utf8') || '{}'); } catch (_) {}

  const flat = flatten(body.messages);
  const stream = body.stream === true;
  const id = 'chatcmpl-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  const created = Math.floor(Date.now() / 1000);
  const modelId = cfg.id;
  const report = function (d) { try { if (onDone) onDone(d); } catch (_) {} };

  if (!flat.prompt.trim()) {
    res.writeHead(400, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: { message: 'messages must contain text.', type: 'invalid_request_error', param: 'messages', code: 'invalid_request' } }));
    return report({ tin: 0, tout: 0, ms: 0, code: 400, ok: false, err: 'empty messages' });
  }

  const chunk = function (delta, finish) {
    return 'data: ' + JSON.stringify({
      id: id, object: 'chat.completion.chunk', created: created, model: modelId,
      choices: [{ index: 0, delta: delta, finish_reason: finish || null }],
    }) + '\n\n';
  };

  if (stream) {
    res.writeHead(200, {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache, no-transform',
      'connection': 'keep-alive',
      'x-accel-buffering': 'no',
    });
    res.write(chunk({ role: 'assistant' }, null));

    let alive = true;
    const ctl = {};
    // client gone -> kill the process (only stopping writes left 6 processes generating 25 s later)
    res.on('close', function () { alive = false; if (ctl.kill) ctl.kill(); });

    run(cfg, flat, function (tok) { if (alive) res.write(chunk({ content: tok }, null)); }, ctl)
      .then(function (r) {
        const u = toUsage(r.usage);
        if (!alive) return report({ tin: billed(u), tout: u ? u.completion_tokens : 0, ms: Date.now() - t0, code: 200, ok: true, cut: true });
        res.write(chunk({}, 'stop'));
        if (u) {
          res.write('data: ' + JSON.stringify({
            id: id, object: 'chat.completion.chunk', created: created, model: modelId,
            choices: [], usage: u,
          }) + '\n\n');
        }
        res.write('data: [DONE]\n\n');
        res.end();
        report({ tin: billed(u), tout: u ? u.completion_tokens : 0, ms: Date.now() - t0, code: 200, ok: true, usd: r.costUsd });
      })
      .catch(function (e) {
        // headers already sent as 200: the error goes in the text
        if (alive) {
          try {
            res.write(chunk({ content: '\n\n[Claude engine error: ' + e.message + ']' }, 'stop'));
            res.write('data: [DONE]\n\n');
            res.end();
          } catch (_) {}
        }
        report({ tin: 0, tout: 0, ms: Date.now() - t0, code: 500, ok: false, err: e.message });
      });
    return;
  }

  run(cfg, flat, null)
    .then(function (r) {
      const u = toUsage(r.usage || {});
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({
        id: id, object: 'chat.completion', created: created, model: modelId,
        choices: [{ index: 0, message: { role: 'assistant', content: r.text }, finish_reason: 'stop' }],
        usage: u,
      }));
      report({ tin: billed(u), tout: u.completion_tokens, ms: Date.now() - t0, code: 200, ok: true, usd: r.costUsd });
    })
    .catch(function (e) {
      res.writeHead(502, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: { message: 'Claude engine error: ' + e.message, type: 'api_error', param: null, code: 'upstream_error' } }));
      report({ tin: 0, tout: 0, ms: Date.now() - t0, code: 502, ok: false, err: e.message });
    });
}

module.exports = { options: options, serve: serve, available: available, flatten: flatten, run: run, toUsage: toUsage, NO_TOOLS: NO_TOOLS };

if (require.main === module) {
  const cfg = options({ enabled: true, model: process.argv[3] || 'sonnet' });
  available(cfg, function (ok) {
    console.log('claude installed: ' + ok + ', model: ' + cfg.model + ', cwd: ' + cfg.cwd);
    if (!ok) return;
    const t0 = Date.now();
    run(cfg, flatten([{ role: 'user', content: process.argv[2] || 'Reply in one line: say OK and what 7x6 is.' }]), function (t) { process.stdout.write(t); })
      .then(function (r) {
        console.log('\n--- usage: ' + JSON.stringify(toUsage(r.usage)) + ', $' + r.costUsd + ', ' + ((Date.now() - t0) / 1000).toFixed(1) + 's');
      })
      .catch(function (e) { console.log('\n[X] ' + e.message); });
  });
}
