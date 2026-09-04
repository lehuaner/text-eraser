// Debug 178_orig: is the WASM-vs-CPU prob gap a preprocessing bug in JS or irreducible runtime noise?
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const ort = require('onnxruntime-web');

const OPENCV = 'D:/Code/Project/Python/TextPatch/browser/vendor/opencv.js.inline.bak';
const WORK = 'D:/Code/Project/Python/TextPatch/scripts/_cmp_work';
const SRC = 'D:/Code/Project/Python/TextPatch/browser/src';
const MODEL = 'D:/Code/Project/Python/TextPatch/browser/vendor/ch_PP-OCRv4_det.onnx';
const ORT_WASM = 'C:/Users/乐幻/.workbuddy/binaries/node/workspace/node_modules/onnxruntime-web/dist/';
const NAME = '178_orig';

function readF32(p) { const b = fs.readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.byteLength >> 2); }
function start(cv) {
  (async () => {
    const cvb = await import(pathToFileURL(path.join(SRC, 'cv-bridge.js')).href);
    const det = await import(pathToFileURL(path.join(SRC, 'detect-dbnet.js')).href);
    cvb.setCv(cv);
    ort.env.wasm.wasmPaths = pathToFileURL(ORT_WASM).href;
    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create(new Uint8Array(fs.readFileSync(MODEL)), { executionProviders: ['cpu'] });

    const d = path.join(WORK, NAME);
    const [H, W] = fs.readFileSync(path.join(d, 'dims.txt'), 'utf8').trim().split(/\s+/).map(Number);
    const [nw, nh, thr] = fs.readFileSync(path.join(d, 'meta.txt'), 'utf8').trim().split(/\s+/).map(Number);
    const rgb = readF32(path.join(d, 'rgb.raw'));
    const pyProb = readF32(path.join(d, 'prob.raw'));

    const res = await det.inferDbnet(rgb, H, W, { ort, session, strength: 1.0, boxThreshold: 0.3, maxSide: 960 });
    const jsProb = res.prob;

    // Compare preprocessed tensor directly: rebuild via the SAME logic the browser uses,
    // then compare to a Python-replicated tensor is done in the .py side; here just compare probs.
    const wasmBoxes = det.detectBoxesFromProb(rgb, H, W, jsProb, res.nw, res.nh, res.thr, {});
    const pyBoxes = det.detectBoxesFromProb(rgb, H, W, pyProb, nw, nh, thr, {});
    console.log('WASM boxes:', JSON.stringify(wasmBoxes));
    console.log('PY   boxes:', JSON.stringify(pyBoxes));

    const OPTS = { strength: 1.0, q_off: 55.0, minArea: 30, maxAreaRatio: 0.40, maxBoxRatio: 0.40, fillWhite: true, fillMaxDist: 12, tintFill: true, upscale: true };
    const mWasm = det.buildMaskFromProb(rgb, H, W, jsProb, res.nw, res.nh, res.thr, OPTS).mask;
    const mPy = det.buildMaskFromProb(rgb, H, W, pyProb, nw, nh, thr, OPTS).mask;
    const refm = new Uint8Array(fs.readFileSync(path.join(d, 'refmask.raw')).buffer, 0, H * W);

    // Bounding box of where mWasm differs from mPy (the algorithm-internal divergence)
    let minx = 1e9, miny = 1e9, maxx = -1, maxy = -1, diff = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if ((mWasm[i] ? 1 : 0) !== (mPy[i] ? 1 : 0)) { diff++; if (x < minx) minx = x; if (y < miny) miny = y; if (x > maxx) maxx = x; if (y > maxy) maxy = y; }
    }
    console.log(`mWasm-vs-mPy diff=${diff} bbox=(${minx},${miny})-(${maxx},${maxy})`);
    let iw = 0, uw = 0, ip = 0, up = 0;
    for (let i = 0; i < H * W; i++) {
      const a = mWasm[i] ? 1 : 0, b = mPy[i] ? 1 : 0, c = refm[i] ? 1 : 0;
      if (a & b) iw++; if (a | b) uw++; if (a & c) ip++; if (a | c) up++;
    }
    console.log(`WASM-vs-PY  iou=${(iw / uw).toFixed(4)}`);
    console.log(`WASM-vs-REF iou=${(ip / up).toFixed(4)}`);
    let ipy = 0, upy = 0;
    for (let i = 0; i < H * W; i++) { const a = mPy[i] ? 1 : 0, c = refm[i] ? 1 : 0; if (a & c) ipy++; if (a | c) upy++; }
    console.log(`PY-vs-REF   iou=${(ipy / upy).toFixed(4)}`);
    console.log('DONE');
  })().catch((e) => { console.error('ERR', e); process.exit(1); });
}
const cvmod = require(OPENCV);
(cvmod.then ? cvmod.then(start) : start(cvmod));
