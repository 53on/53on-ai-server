'use strict';
// GB (decimal, as gpuGb) the GPU can still take: free VRAM + the GTT the system can spare (MemAvailable over a 6 GiB
// floor), at most what the TTM pool has left. A source it cannot read gives NaN (unknown): the gate then refuses nothing.
const fs = require('fs'), D = '/sys/class/drm/';
const num = (p, re) => { try { const s = fs.readFileSync(p, 'utf8'); return re ? re.exec(s)[1] * 1024 : +s; } catch (_) { return NaN; } };
exports.free = () => {
  let c; try { c = fs.readdirSync(D).filter((x) => !x.indexOf('card')).sort().find((x) => fs.existsSync(D + x + '/device/mem_info_vram_total')); } catch (_) {}
  const v = D + c + '/device/mem_info_';
  return +((Math.max(0, num(v + 'vram_total') - num(v + 'vram_used')) + Math.max(0, Math.min(num('/proc/meminfo', /^MemAvailable:\s+(\d+)/m) - 6 * 2 ** 30,
    num('/sys/module/ttm/parameters/pages_limit') * 4096 - num(v + 'gtt_used')))) / 1e9).toFixed(2);
};
