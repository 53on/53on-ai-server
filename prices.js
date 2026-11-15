'use strict';
/*
 * prices.js: cost in money, the only unit every provider shares.
 *
 * Tokens do not work across providers: OpenAI bills an image in output tokens (auditable), Gemini
 * too, and Alibaba bills per image and says its own tokens are "not billed". LiteLLM, OpenRouter
 * and Helicone all refuse to convert images into tokens; they price in USD. Same here.
 *
 * Rates come from LiteLLM's public table (3561 models, 2.1 MB). It is NOT kept in memory: on
 * refresh only the models this server actually serves are extracted into prices.json (a few KB),
 * so the 2 MB is downloaded once a day and never carried around.
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

const SRC = 'https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json';
const FILE = path.join(__dirname, 'prices.json');
const TTL = 24 * 3600 * 1000;

let table = null, at = 0;

const load = () => {
  if (table) return table;
  try { const j = JSON.parse(fs.readFileSync(FILE, 'utf8')); table = j.rates || {}; at = j.at || 0; }
  catch (_) { table = {}; at = 0; }
  return table;
};

// Our id is "provider/model"; the table uses bare names or its own prefixes. Try the likely ones.
const aliases = (id) => {
  const i = String(id).indexOf('/');
  const bare = i > 0 ? id.slice(i + 1) : id;
  const prov = i > 0 ? id.slice(0, i) : '';
  const out = [id, bare];
  if (prov === 'alibaba') out.push('dashscope/' + bare);
  if (prov) out.push(prov + '/' + bare);
  out.push('openai/' + bare, 'gemini/' + bare);
  return out;
};

function rateOf(id) {
  const t = load();
  for (const a of aliases(id)) if (t[a]) return t[a];
  return null;
}

// usage -> USD. Images priced per image when the table says so; otherwise per output token.
// Optional cfg: a per-image price written by hand wins over the public table; it is the only one for
// models LiteLLM does not list, and without it the picture is billed as free.
// Startup always comes early: when the server starts, each provider's model list has not arrived yet,
// so asking then asks for nothing (measured: panelModels returns 0). So the table fills ON DEMAND:
// a model used without a price is noted, and all of them are asked together at the next pause.
// It fills with what is really spent, and never too late.
const want = new Set();
let pend = null;
function need(id) {
  if (!id || rateOf(id)) return;
  want.add(id);
  if (pend) return;
  pend = setTimeout(() => {
    pend = null;
    const ids = Array.from(want); want.clear();
    refresh(ids, function () {}, true);
  }, 5000);
  if (pend.unref) pend.unref();
}

function costOf(id, u, cfg) {
  if (!u) return 0;
  const r = rateOf(id) || (need(id), {});
  const tin = +u.prompt_tokens || +u.input_tokens || 0;
  const tout = +u.completion_tokens || +u.output_tokens || 0;
  const imgs = +u.image_count || 0;
  let c = tin * (+r.input_cost_per_token || 0) + tout * (+r.output_cost_per_token || 0);
  if (imgs) c += imgs * (+((cfg || {}).usd_per_image || {})[id] || +r.output_cost_per_image || 0);
  return c;
}

// Refresh keeps ONLY the ids in use: the 2 MB never reaches disk or memory.
function refresh(ids, cb, force) {
  cb = cb || function () {};
  load();
  // An empty table is never sealed: saved with its time, the TTL would trust it for 24 h and every
  // cost would be recorded as zero without anyone noticing (it happened once, for eight days).
  if (!ids || !ids.length) return cb(0);
  if (!force && at && Date.now() - at < TTL) return cb(0);
  let s = '';
  const req = https.get(SRC, { timeout: 60000 }, (r) => {
    if (r.statusCode !== 200) { r.resume(); return cb(0); }
    r.on('data', (d) => { s += d; });
    r.on('end', () => {
      let all; try { all = JSON.parse(s); } catch (_) { return cb(0); }
      const rates = {};
      (ids || []).forEach((id) => {
        for (const a of aliases(id)) if (all[a]) { rates[a] = all[a]; break; }
      });
      const n = Object.keys(rates).length;
      if (!n) return cb(0);   // ni un precio reconocido: se conserva lo que ya hubiera, no se pisa con nada
      // ADD, never replace: two requests with different lists used to erase each other
      table = Object.assign(table || {}, rates); at = Date.now();
      try { fs.writeFileSync(FILE, JSON.stringify({ at: at, src: SRC, rates: table })); } catch (_) {}
      cb(n);
    });
  });
  req.on('error', () => cb(0));
  req.on('timeout', () => { req.destroy(); cb(0); });
}

// THE STANDARD, one for every image AI: chips = pixels / density. Taken from OpenAI's published
// token table, the only one that is auditable (its tokens x its price match its price per image)
// and that turns out to be LINEAR IN PIXELS: 3855 px per chip on low, 993 on medium, 252 on high,
// constant across its three sizes. So it does not depend on the provider: a local Flux and a
// cloud Wan pay the same for the same picture, which is what a quota should do.
const PX_PER_CHIP = { low: 3855, medium: 993, high: 252 };
// "1024*1024", "1024x1024", "1K"/"2K"/"4K" (Alibaba) or {w,h}
function pixelsOf(size) {
  if (!size) return 0;
  if (typeof size === 'object') return (+size.w || 0) * (+size.h || 0);
  const s = String(size).trim().toLowerCase();
  const m = s.match(/^(\d+)\s*[*x×]\s*(\d+)$/);
  if (m) return +m[1] * +m[2];
  const k = s.match(/^(\d+(?:\.\d+)?)k$/);
  if (k) { const side = Math.round(+k[1] * 1024); return side * side; }
  return 0;
}
// size from the provider usage, quality from config ('medium' unless told otherwise)
function imageChips(n, size, quality) {
  const px = pixelsOf(size);
  if (!n || !px) return 0;
  return Math.round(n * px / (PX_PER_CHIP[quality] || PX_PER_CHIP.medium));
}

// An image in CHIPS: its price divided by the price of one output token. Both numbers are
// published (LiteLLM table, or usd_per_image in config for what the table does not carry).
// The reference token is the model itself; if it has no token price (image-only models), the
// fallback is the provider text model given in cfg.chip_ref. Returns 0 when a price is missing:
// counting zero is honest, guessing a rate is not.
function imageTokens(id, imgs, cfg) {
  if (!imgs) return 0;
  cfg = cfg || {};
  const r = rateOf(id) || {};
  // If the model is priced PER TOKEN, its own tokens are already chips (Gemini reports 1290 per
  // image, OpenAI Images reports its own): converting on top would charge the picture twice.
  if (+r.output_cost_per_token > 0) return 0;
  const usdImg = +((cfg.usd_per_image || {})[id]) || +r.output_cost_per_image || 0;
  const ref = rateOf(cfg.chip_ref || '') || {};
  const usdTok = +ref.output_cost_per_token || 0;
  if (!usdImg || !usdTok) return 0;
  return Math.round(imgs * usdImg / usdTok);
}

module.exports = {
  need: need, costOf, rateOf, refresh, imageTokens, imageChips, pixelsOf, table: () => load() };

if (require.main === module) {
  const ids = process.argv.slice(2);
  refresh(ids, (n) => {
    console.log('rates saved: ' + n);
    ids.forEach((id) => {
      const r = rateOf(id);
      console.log('  ' + id.padEnd(34) + (r ? JSON.stringify({ in: r.input_cost_per_token, out: r.output_cost_per_token, img: r.output_cost_per_image, mode: r.mode }) : 'no price'));
    });
  });
}
