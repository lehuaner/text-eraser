// run_js.cjs — execute the actual browser port (patchmatch.js / deglow.js) under
// Node with the REAL opencv.js (full wasm build), and emit raw RGB outputs for the
// pixel comparator.
//
// Notes:
//  * opencv.js is loaded via `require()` (CommonJS). The wasm runtime initializes
//    through the `cv.then(start)` callback (the `await new Promise` form SIGTERMs
//    in this Node build).
//  * The ES-module browser port is loaded with dynamic import() using file:// URLs
//    (Windows requires a URL scheme for ESM import).
//  * cv ops run through the same opencv.js wasm the browser will use, so they are
//    bit-identical to cv2.
//
// Usage: node run_js.cjs <work_dir>
//   work_dir must contain: input.rgb (float32 H*W*3), input.mask (uint8 H*W), dims.txt
//   writes: out_inpaint.rgb, out_erase.rgb

const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');

const OPENCV = 'C:/Users/乐幻/.workbuddy/binaries/node/workspace/opencv_full.js';

function readF32(p) {
  const b = fs.readFileSync(p);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength >> 2);
}
function readU8(p) { return new Uint8Array(fs.readFileSync(p)); }
function writeF32(p, arr) { fs.writeFileSync(p, Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength)); }
function mask255From01(m01) {
  const a = new Uint8Array(m01.length);
  for (let i = 0; i < m01.length; i++) a[i] = m01[i] ? 255 : 0;
  return a;
}
const asUrl = (p) => pathToFileURL(p).href;

function start(cv) {
  console.log('opencv ready; inpaint:', typeof cv.inpaint, '| TELEA:', cv.INPAINT_TELEA);
  (async () => {
    const cvb = await import(asUrl(path.join(__dirname, '..', 'src', 'cv-bridge.js')));
    const { patchmatchInpaint } = await import(asUrl(path.join(__dirname, '..', 'src', 'patchmatch.js')));
    const { deglowFaintGreen } = await import(asUrl(path.join(__dirname, '..', 'src', 'deglow.js')));
    cvb.setCv(cv);

    const base = path.resolve(process.argv[2] || '.');
    const dims = fs.readFileSync(path.join(base, 'dims.txt'), 'utf8').trim().split(/\s+/).map(Number);
    const H = dims[0], W = dims[1];

    const rgb = readF32(path.join(base, 'input.rgb'));
    const maskU8 = readU8(path.join(base, 'input.mask'));
    const mask255 = new Uint8Array(H * W);
    for (let i = 0; i < H * W; i++) mask255[i] = maskU8[i] ? 255 : 0;

    // ---- inpaint reference (matches gen_reference: patch_fill.inpaint(rgb, mask)) ----
    const t0 = Date.now();
    const outInpaint = patchmatchInpaint(rgb.slice(), H, W, mask255, { flatSpan: 40, flatTex: 15.0 });
    writeF32(path.join(base, 'out_inpaint.rgb'), outInpaint);
    console.log('inpaint done in', Date.now() - t0, 'ms');

    // ---- erase reference (mirrors index.js eraseTextGlyphs: deglow + ellipse-dilate + sample) ----
    const rgb2 = rgb.slice();
    deglowFaintGreen(rgb2, H, W, mask255, { thr: 6, near_r: 24, g_lo: 85, protect: 1, strength: 1.0 });

    const tm01 = new Uint8Array(H * W);
    for (let i = 0; i < H * W; i++) tm01[i] = mask255[i] ? 1 : 0;
    const edge = 1;
    const filled01 = edge > 0 ? cvb.dilateMask(tm01, H, W, edge * 2 + 1) : tm01;
    const sample01 = new Uint8Array(H * W);
    for (let i = 0; i < H * W; i++) sample01[i] = filled01[i] ? 0 : 1;

    const t1 = Date.now();
    const outErase = patchmatchInpaint(rgb2, H, W, mask255From01(filled01), {
      sampleMask: mask255From01(sample01),
      flatSpan: 40,
      flatTex: 15.0,
    });
    writeF32(path.join(base, 'out_erase.rgb'), outErase);
    console.log('erase done in', Date.now() - t1, 'ms');

    console.log('JS outputs written for', H, 'x', W);
  })().catch((e) => { console.error('RUNNER ERROR:', e); process.exit(1); });
}

const cvmod = require(OPENCV);
(cvmod.then ? cvmod.then(start) : start(cvmod));
