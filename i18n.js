'use strict';
// i18n.js — 53ON Harness's i18n: one table per language in lang/<code>.json, keys by surface (ui.),
// a key a language lacks is English's, never blank. Here the caller keeps the language (config.json lang), not userData,
// and a table is read on each use, as admin.html is: a word changed in lang/ shows on the next page load, no restart.
//   __(key, vars) · lang(code?) · langs() → [{code, name}] · bag(prefix) → a surface's strings, for a page with no require
const fs = require('fs'), path = require('path');
const DIR = path.join(__dirname, 'lang'), BASE = 'en';
let cur = BASE;
function load(code) { try { return JSON.parse(fs.readFileSync(path.join(DIR, code + '.json'), 'utf8')); } catch (_) { return {}; } }
function codes() { try { return fs.readdirSync(DIR).filter((f) => /^[a-z]{2}\.json$/.test(f)).map((f) => f.slice(0, 2)); } catch (_) { return [BASE]; } }
function langs() { return codes().map((c) => ({ code: c, name: load(c).name || c })); }
function lang(code) {
  if (code !== undefined) {
    if (!codes().includes(code)) return false;
    cur = code;
  }
  return cur;
}
function __(k, v) {
  let s = load(lang())[k];
  if (s == null) s = load(BASE)[k];
  if (s == null) s = k;
  if (v) for (const n in v) s = s.split('{' + n + '}').join(v[n]);
  return s;
}
function bag(prefix) {
  const o = {}, b = load(BASE), c = load(lang()), p = prefix + '.';
  for (const k in b) if (k.indexOf(p) === 0) o[k.slice(p.length)] = c[k] != null ? c[k] : b[k];
  return o;
}
module.exports = { __: __, lang: lang, langs: langs, bag: bag };
