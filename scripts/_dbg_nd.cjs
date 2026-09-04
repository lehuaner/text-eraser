// Confirm non-determinism: run buildMaskFromProb(pyProb) 5x in ONE process for 178_orig
// and report IoU vs ref each time. If it varies -> cv-bridge/detect-dbnet has a
// state/initialization bug (e.g. uninitialized Mat).
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const OPENCV = 'D:/Code/Project/Python/TextPatch/browser/vendor/opencv.js.inline.bak';
const WORK = 'D:/Code/Project/Python/TextPatch/scripts/_cmp_work';
const SRC = 'D:/Code/Project/Python/TextPatch/browser/src';
const NAME = '178_orig';
const OPTS = { strength: 1.0, q_off: 55.0, minArea: 30, maxAreaRatio: 0.40, maxBoxRatio: 0.40, fillWhite: true, fillMaxDist: 12, tintFill: true, upscale: true };
function readF32(p) { const b = fs.readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.byteLength >> 2); }
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
    for (let k = 0; k < 5; k++) {
      const m = det.buildMaskFromProb(rgb, H, W, pyProb, nw, nh, thr, OPTS).mask;
      let i = 0, u = 0, mp = 0;
      for (let j = 0; j < H * W; j++) { const a = m[j] ? 1 : 0, b = refm[j] ? 1 : 0; if (a & b) i++; if (a | b) u++; if (a) mp++; }
      console.log(`run ${k}: iou=${(i / u).toFixed(4)} jsPix=${mp} inter=${i}`);
    }
    console.log('DONE');
  })().catch((e) => { console.error('ERR', e); process.exit(1); });
}
const cvmod = require(OPENCV);
(cvmod.then ? cvmod.then(start) : start(cvmod));
