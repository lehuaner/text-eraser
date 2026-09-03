// Run the JS port (browser/src/detect-dbnet.js buildMaskFromProb) under Node with the
// REAL opencv.js build, feeding the Python-dumped prob map + rgb, and write mask_js.raw.
// This isolates the post-processing (box->Otsu->fill->tint->clean) from DBNet inference.
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');

const OPENCV = 'D:/Code/Project/Python/TextPatch/browser/vendor/opencv.js.inline.bak';
const WORK = 'D:/Code/Project/Python/TextPatch/scripts/_cmp_work';
const SRC = 'D:/Code/Project/Python/TextPatch/browser/src';

function readF32(p) { const b = fs.readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.byteLength >> 2); }
function readU8(p) { return new Uint8Array(fs.readFileSync(p)); }

function start(cv) {
  (async () => {
    const cvb = await import(pathToFileURL(path.join(SRC, 'cv-bridge.js')).href);
    const { buildMaskFromProb } = await import(pathToFileURL(path.join(SRC, 'detect-dbnet.js')).href);
    cvb.setCv(cv);

    const dirs = fs.readdirSync(WORK).filter((d) => fs.existsSync(path.join(WORK, d, 'rgb.raw')));
    for (const name of dirs) {
      const d = path.join(WORK, name);
      const [H, W] = fs.readFileSync(path.join(d, 'dims.txt'), 'utf8').trim().split(/\s+/).map(Number);
      const [nw, nh, thr] = fs.readFileSync(path.join(d, 'meta.txt'), 'utf8').trim().split(/\s+/).map(Number);
      const rgb = readF32(path.join(d, 'rgb.raw'));
      const prob = readF32(path.join(d, 'prob.raw'));
      const { mask } = buildMaskFromProb(rgb, H, W, prob, nw, nh, thr, {
        strength: 1.0, q_off: 55.0, minArea: 30, maxAreaRatio: 0.40, maxBoxRatio: 0.40,
        fillWhite: true, fillMaxDist: 12, tintFill: true, upscale: true,
      });
      fs.writeFileSync(path.join(d, 'mask_js.raw'), Buffer.from(mask.buffer, mask.byteOffset, mask.byteLength));
      let pix = 0; for (let i = 0; i < mask.length; i++) if (mask[i]) pix++;
      console.log(`${name}: js_mask_pix=${pix}`);
    }
    console.log('NODE DONE');
  })().catch((e) => { console.error('NODE ERROR:', e); process.exit(1); });
}

const cvmod = require(OPENCV);
(cvmod.then ? cvmod.then(start) : start(cvmod));
