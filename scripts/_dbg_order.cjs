// Isolate whether requiring/using onnxruntime-web in the SAME process corrupts
// opencv.js (cv-bridge) computations. This matters because the browser Worker
// loads BOTH opencv.js and onnxruntime-web together.
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const OPENCV = 'D:/Code/Project/Python/TextPatch/browser/vendor/opencv.js.inline.bak';
const WORK = 'D:/Code/Project/Python/TextPatch/scripts/_cmp_work';
const SRC = 'D:/Code/Project/Python/TextPatch/browser/src';
const MODEL = 'D:/Code/Project/Python/TextPatch/browser/vendor/ch_PP-OCRv4_det.onnx';
const ORT_WASM = 'C:/Users/乐幻/.workbuddy/binaries/node/workspace/node_modules/onnxruntime-web/dist/';
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

    // STEP A: compute JS mask from PY prob BEFORE any onnx is even required.
    const mA = det.buildMaskFromProb(rgb, H, W, pyProb, nw, nh, thr, OPTS).mask;
    const iouA = iou(mA, refm);
    console.log(`[A] JS(pyProb) vs REF  BEFORE onnx: iou=${iouA.toFixed(4)}`);

    // Now load onnxruntime-web and create a session.
    const ort = require('onnxruntime-web');
    ort.env.wasm.wasmPaths = pathToFileURL(ORT_WASM).href;
    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create(new Uint8Array(fs.readFileSync(MODEL)), { executionProviders: ['cpu'] });

    // STEP B: recompute JS mask from the SAME pyProb AFTER onnx session exists.
    const mB = det.buildMaskFromProb(rgb, H, W, pyProb, nw, nh, thr, OPTS).mask;
    console.log(`[B] JS(pyProb) vs REF  AFTER  onnx: iou=${iouB(mB, refm).toFixed(4)}`);
    console.log(`[B] mA vs mB (same input, onnx present?): diff=${diffCount(mA, mB)}`);

    // STEP C: WASM prob pipeline.
    const res = await det.inferDbnet(rgb, H, W, { ort, session, strength: 1.0, boxThreshold: 0.3, maxSide: 960 });
    const mC = det.buildMaskFromProb(rgb, H, W, res.prob, res.nw, res.nh, res.thr, OPTS).mask;
    console.log(`[C] JS(wasmProb) vs REF: iou=${iouB(mC, refm).toFixed(4)}`);
    console.log('DONE');
  })().catch((e) => { console.error('ERR', e); process.exit(1); });
}
function iou(a, b) { let i = 0, u = 0; for (let k = 0; k < a.length; k++) { const x = a[k] ? 1 : 0, y = b[k] ? 1 : 0; if (x & y) i++; if (x | y) u++; } return u ? i / u : 1; }
function iouB(a, b) { return iou(a, b); }
function diffCount(a, b) { let c = 0; for (let k = 0; k < a.length; k++) if ((a[k] ? 1 : 0) !== (b[k] ? 1 : 0)) c++; return c; }
const cvmod = require(OPENCV);
(cvmod.then ? cvmod.then(start) : start(cvmod));
