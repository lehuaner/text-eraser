// Run the BROWSER's actual inferDbnet (real onnxruntime-web + same ONNX model the
// browser ships) under Node, and compare its probability map vs Python's prob.raw.
// This closes the loop on the DBNet inference half (the Node mask harness only
// fed the pre-dumped Python prob, isolating post-processing).
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const ort = require('C:/Users/乐幻/.workbuddy/binaries/node/workspace/node_modules/onnxruntime-web');

const OPENCV = 'D:/Code/Project/Python/TextPatch/browser/vendor/opencv.js.inline.bak';
const WORK = 'D:/Code/Project/Python/TextPatch/scripts/_cmp_work';
const SRC = 'D:/Code/Project/Python/TextPatch/browser/src';
const MODEL = 'D:/Code/Project/Python/TextPatch/browser/vendor/ch_PP-OCRv4_det.onnx';
const ORT_WASM = 'C:/Users/乐幻/.workbuddy/binaries/node/workspace/node_modules/onnxruntime-web/dist/';

function readF32(p) { const b = fs.readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.byteLength >> 2); }

function start(cv) {
  (async () => {
    const cvb = await import(pathToFileURL(path.join(SRC, 'cv-bridge.js')).href);
    const det = await import(pathToFileURL(path.join(SRC, 'detect-dbnet.js')).href);
    cvb.setCv(cv);
    ort.env.wasm.wasmPaths = pathToFileURL(ORT_WASM).href;
    ort.env.wasm.numThreads = 1;
    const modelBuf = fs.readFileSync(MODEL);
    const session = await ort.InferenceSession.create(new Uint8Array(modelBuf), { executionProviders: ['cpu'] });

    const dirs = fs.readdirSync(WORK).filter((d) => fs.existsSync(path.join(WORK, d, 'rgb.raw')));
    const OPTS = { strength: 1.0, q_off: 55.0, minArea: 30, maxAreaRatio: 0.40, maxBoxRatio: 0.40,
      fillWhite: true, fillMaxDist: 12, tintFill: true, upscale: true };
    for (const name of dirs) {
      const d = path.join(WORK, name);
      const [H, W] = fs.readFileSync(path.join(d, 'dims.txt'), 'utf8').trim().split(/\s+/).map(Number);
      const [nw, nh, thr] = fs.readFileSync(path.join(d, 'meta.txt'), 'utf8').trim().split(/\s+/).map(Number);
      const rgb = readF32(path.join(d, 'rgb.raw'));
      const res = await det.inferDbnet(rgb, H, W, { ort, session, strength: 1.0, boxThreshold: 0.3, maxSide: 960 });
      const prob = res.prob;
      const ref = readF32(path.join(d, 'prob.raw'));
      const n = Math.min(prob.length, ref.length);
      let maxd = 0, sumd = 0, disagree = 0;
      for (let i = 0; i < n; i++) {
        const a = prob[i], b = ref[i];
        const dd = Math.abs(a - b);
        if (dd > maxd) maxd = dd;
        sumd += dd;
        if ((a > thr) !== (b > thr)) disagree++;
      }
      // Full browser pipeline mask from the WASM prob, compared to backend refmask.
      const { mask } = det.buildMaskFromProb(rgb, H, W, prob, res.nw, res.nh, res.thr, OPTS);
      const refm = new Uint8Array(fs.readFileSync(path.join(d, 'refmask.raw')).buffer, 0, H * W);
      let inter = 0, uni = 0, mp = 0;
      for (let i = 0; i < H * W; i++) {
        const a = mask[i] ? 1 : 0, b = refm[i] ? 1 : 0;
        if (a & b) inter++; if (a | b) uni++; if (a) mp++;
      }
      const iou = uni ? inter / uni : 1.0;
      fs.writeFileSync(path.join(d, 'mask_js_wasm.raw'), Buffer.from(mask.buffer, mask.byteOffset, mask.byteLength));
      console.log(`${name}: maxAbsDiff=${maxd.toFixed(4)} meanAbsDiff=${(sumd / n).toFixed(5)} thrDisagree=${(100 * disagree / n).toFixed(3)}% | END2END iou=${iou.toFixed(4)} jsMaskPix=${mp} inter=${inter}`);
    }
    console.log('INFER DONE');
  })().catch((e) => { console.error('INFER ERROR:', e); process.exit(1); });
}

const cvmod = require(OPENCV);
(cvmod.then ? cvmod.then(start) : start(cvmod));
