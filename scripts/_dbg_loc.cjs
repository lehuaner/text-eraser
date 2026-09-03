// Localize which stage causes 178_orig to diverge from REF. Toggling each opt and
// comparing to the backend refmask shows which stage is the differentiator.
// (Within one process cv ops are stable, so the IoU *values* reveal the stage.)
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const OPENCV = 'D:/Code/Project/Python/TextPatch/browser/vendor/opencv.js.inline.bak';
const WORK = 'D:/Code/Project/Python/TextPatch/scripts/_cmp_work';
const SRC = 'D:/Code/Project/Python/TextPatch/browser/src';
const NAME = '178_orig';
function readF32(p) { const b = fs.readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.byteLength >> 2); }
function iou(a, b) { let i = 0, u = 0; for (let k = 0; k < a.length; k++) { const x = a[k] ? 1 : 0, y = b[k] ? 1 : 0; if (x & y) i++; if (x | y) u++; } return u ? i / u : 1; }
function start(cv) {
  (async () => {
    const cvb = await import(pathToFileURL(path.join(SRC, 'cv-bridge.js')).href);
    const det = await import(pathToFileURL(path.join(SRC, 'detect-dbnet.js')).href);
    cvb.setCv(cv);
    const d = path.join(WORK, NAME);
    const [H, W] = fs.readFileSync(path.join(d, 'dims.txt'), 'utf8').trim().split(/\s+/).map(Number);
    const [nw, nh, thr] = fs.readFileSync(path.join(d, 'meta.txt'), 'utf8').trim().split(/\s+/).map(Number);
    const rgb = readF32(path.join(d, 'rgb.raw'));
    const pyProb = readF32(path.join(d, 'prob.raw'));
    const refm = new Uint8Array(fs.readFileSync(path.join(d, 'refmask.raw')).buffer, 0, H * W);
    const base = { strength: 1.0, q_off: 55.0, minArea: 30, maxAreaRatio: 0.40, maxBoxRatio: 0.40, fillWhite: true, fillMaxDist: 12, tintFill: true, upscale: true };
    const variants = {
      'base': {},
      'upscale:false': { upscale: false },
      'fillWhite:false': { fillWhite: false },
      'tintFill:false': { tintFill: false },
      'noFillNoTint': { fillWhite: false, tintFill: false },
    };
    for (const [k, ov] of Object.entries(variants)) {
      const opts = Object.assign({}, base, ov);
      const m = det.buildMaskFromProb(rgb, H, W, pyProb, nw, nh, thr, opts).mask;
      let pc = 0; for (let i = 0; i < m.length; i++) if (m[i]) pc++;
      console.log(`${k.padEnd(16)} iou=${iou(m, refm).toFixed(4)} jsPix=${pc}`);
    }
    console.log('DONE');
  })().catch((e) => { console.error('ERR', e); process.exit(1); });
}
const cvmod = require(OPENCV);
(cvmod.then ? cvmod.then(start) : start(cvmod));
