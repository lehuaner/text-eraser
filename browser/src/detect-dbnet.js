// detect-dbnet.js — DBNet (PP-OCRv4 det) text detection helper for the browser.
//
// This module mirrors text_eraser.text_select.detect_text_mask(method="ml") so the
// browser's local text mask is pixel-identical to the backend's. The backend pipeline:
//
//   1. DBNet at max_side=960 -> probability map -> threshold(0.15) -> 3x3 RECT dilate
//      -> connected components -> bounding rects scaled to original -> pad=3 -> filters
//      (min_area=30, max_area_ratio=0.40, max_box_ratio=0.40).  (ml_text_select.detect_text_ml)
//   2. For each box: crop the ORIGINAL-res grayscale, run Otsu (minority side = glyph),
//      morphological close, optional upscale for small chars, accumulate.  (_detect_text_mask_classic)
//   3. _fill_nearby_white(rgb, mask, max_dist=12) — complete bright/white pixels near mask.
//   4. _grow_color_tint(rgb, mask) — grow along red/green tinted pixels (halo / red overlay).
//   5. _clean_text_mask(min_area=8, max_area_ratio=0.9) + _mask_to_boxes.
//
// NOTE: the backend uses DBNet ONLY for box localization; the actual glyph pixels come
// from box-constrained Otsu on the original-resolution image — NOT the raw DBNet prob map.

import * as cvb from './cv-bridge.js';

const DEFAULT_MODELS = [
  '/browser/vendor/ch_PP-OCRv4_det.onnx',
  'https://hf-mirror.com/enkylin/onnx-ch-PP-OCRv4-det/resolve/main/ch_PP-OCRv4_det.onnx',
  'https://huggingface.co/enkylin/onnx-ch-PP-OCRv4-det/resolve/main/ch_PP-OCRv4_det.onnx',
];

const PADDLE_MEAN = [0.485, 0.456, 0.406]; // BGR
const PADDLE_STD = [0.229, 0.224, 0.225];

let _ort = null;
function getOrt(ort) {
  if (ort) return ort;
  if (_ort) return _ort;
  throw new Error('detect-dbnet: pass `ort` (an onnxruntime-web instance) or call loadDBNet() first');
}

/** Python-style round (round-half-to-even) — cv2/numpy use this; JS Math.round is half-up. */
function pyRound(x) {
  const f = Math.floor(x);
  const diff = x - f;
  if (diff < 0.5) return f;
  if (diff > 0.5) return f + 1;
  return (f % 2 === 0) ? f : f + 1; // exactly .5 -> even
}

function countOnes(a) { let c = 0; for (let i = 0; i < a.length; i++) if (a[i]) c++; return c; }

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

// ---------------------------------------------------------------------------
// DBNet pre-processing — EXACTLY mirrors ml_text_select._dbnet_infer
// ---------------------------------------------------------------------------
function buildTensor(rgb, H, W, maxSide) {
  const scale = Math.min(maxSide / Math.max(H, W), 1.0);
  const nh = Math.max(32, Math.floor((Math.round(H * scale)) / 32) * 32);
  const nw = Math.max(32, Math.floor((Math.round(W * scale)) / 32) * 32);
  const bgr = new Float32Array(H * W * 3);
  for (let i = 0; i < H * W; i++) {
    const s = i * 3;
    bgr[s] = rgb[s + 2]; bgr[s + 1] = rgb[s + 1]; bgr[s + 2] = rgb[s];
  }
  const rgbr = cvb.resizeRgbArea(bgr, H, W, nh, nw);
  const data = new Float32Array(1 * 3 * nh * nw);
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      const si = (y * nw + x) * 3;
      const r = rgbr[si], g = rgbr[si + 1], b = rgbr[si + 2];
      const di = y * nw + x;
      data[di] = (b / 255 - PADDLE_MEAN[0]) / PADDLE_STD[0];
      data[1 * nh * nw + di] = (g / 255 - PADDLE_MEAN[1]) / PADDLE_STD[1];
      data[2 * nh * nw + di] = (r / 255 - PADDLE_MEAN[2]) / PADDLE_STD[2];
    }
  }
  return { data, Hp: nh, Wp: nw };
}

/** Run DBNet and return the probability map + box-detection threshold. */
export async function inferDbnet(rgb, H, W, opts = {}) {
  const ort = getOrt(opts.ort);
  const session = opts.session;
  if (!session) throw new Error('inferDbnet: pass a loaded `session`');
  const strength = opts.strength != null ? opts.strength : 1.0;
  const boxThreshold = opts.boxThreshold != null ? opts.boxThreshold : 0.3;
  const maxSide = opts.maxSide != null ? opts.maxSide : 960;
  const thr = Math.min(0.9, Math.max(0.1, boxThreshold - 0.15 * strength));
  const { data, Hp, Wp } = buildTensor(rgb, H, W, maxSide);
  const tensor = new ort.Tensor('float32', data, [1, 3, Hp, Wp]);
  const feeds = {}; feeds[session.inputNames[0]] = tensor;
  const out = await session.run(feeds);
  const outName = session.outputNames[0];
  let prob = out[outName].data;
  const dims = out[outName].dims;
  const Ho = dims[dims.length - 2], Wo = dims[dims.length - 1];
  let flat = prob;
  if (dims.length === 4) flat = prob.subarray(0, Ho * Wo);
  // The PP-OCRv4 det ONNX bakes sigmoid into its output node, so the raw output is
  // already a probability map in [0,1]. The backend (_dbnet_infer) takes it directly
  // with NO further activation. To stay pixel-identical we must NOT re-apply sigmoid.
  // We only fall back to sigmoid if the output is unambiguously in logit range
  // (mx > 1.5), so a single float overshoot to ~1.0003 near the top never flips the
  // whole map through sigmoid (which would distort an already-sigmoided map and diverge
  // from the backend — this was exactly the 178_orig collapse).
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < flat.length; i++) { if (flat[i] < mn) mn = flat[i]; if (flat[i] > mx) mx = flat[i]; }
  const isLogit = mx > 1.5;
  const map = new Float32Array(Ho * Wo);
  for (let i = 0; i < Ho * Wo; i++) map[i] = isLogit ? sigmoid(flat[i]) : flat[i];
  return { prob: map, nw: Wo, nh: Ho, thr };
}

// ---------------------------------------------------------------------------
// Step 1 — box localization from the prob map (mirror detect_text_ml)
// ---------------------------------------------------------------------------
export function detectBoxesFromProb(rgb, H, W, prob, nw, nh, thr, opts = {}) {
  const minArea = opts.minArea != null ? opts.minArea : 30;
  const maxAreaRatio = opts.maxAreaRatio != null ? opts.maxAreaRatio : 0.40;
  const maxBoxRatio = opts.maxBoxRatio != null ? opts.maxBoxRatio : 0.40;
  const pad = opts.pad != null ? opts.pad : 3;
  const bin = new Uint8Array(nw * nh);
  for (let i = 0; i < nw * nh; i++) bin[i] = prob[i] > thr ? 1 : 0;
  const dil = cvb.dilateMaskRect(bin, nh, nw, 3); // 3x3 RECT, 1 iter
  const cc = cvb.connectedComponents(dil, nh, nw);
  const invW = W / nw, invH = H / nh;
  const wTotal = nw * nh;
  const total = H * W;
  const boxes = [];
  for (let i = 1; i < cc.n; i++) {
    const a = cc.stats[i].area;
    if (a < minArea) continue;
    if (a > wTotal * maxAreaRatio) continue;
    const x = cc.stats[i].left, y = cc.stats[i].top, w = cc.stats[i].width, h = cc.stats[i].height;
    if (w <= 0 || h <= 0) continue;
    let X0 = Math.round(x * invW), Y0 = Math.round(y * invH);
    let X1 = Math.round((x + w) * invW), Y1 = Math.round((y + h) * invH);
    X0 = Math.max(0, X0 - pad); Y0 = Math.max(0, Y0 - pad);
    X1 = Math.min(W - 1, X1 + pad); Y1 = Math.min(H - 1, Y1 + pad);
    if (X1 - X0 <= 1 || Y1 - Y0 <= 1) continue;
    if ((X1 - X0) * (Y1 - Y0) > total * maxBoxRatio) continue;
    boxes.push({ x0: X0, y0: Y0, x1: X1, y1: Y1 });
  }
  return boxes;
}

// ---------------------------------------------------------------------------
// Step 2 — box-constrained Otsu glyph segmentation (mirror _detect_text_mask_classic)
// ---------------------------------------------------------------------------
/** Otsu on a cropped sub-image. Returns 0/255 glyph mask (hs x ws). */
function segBox(grayU8, hs, ws, qOff) {
  const N = hs * ws;
  const { thr } = cvb.thresholdOtsu(grayU8, hs, ws);
  let t = thr;
  if (t <= 0) t = 1; else if (t >= 255) t = 254;
  const below = new Uint8Array(N), above = new Uint8Array(N);
  let cntB = 0, cntA = 0, mB = 0, mA = 0;
  for (let i = 0; i < N; i++) {
    if (grayU8[i] <= t) { below[i] = 1; cntB++; mB += grayU8[i]; }
    else { above[i] = 1; cntA++; mA += grayU8[i]; }
  }
  if (cntB === 0 || cntA === 0) return new Uint8Array(N);
  mB /= cntB; mA /= cntA;
  if (Math.abs(mB - mA) < 20) return new Uint8Array(N); // low-contrast guard
  const sel = new Uint8Array(N);
  if (cntB < cntA) { for (let i = 0; i < N; i++) if (below[i]) sel[i] = 255; }
  else if (cntA < cntB) { for (let i = 0; i < N; i++) if (above[i]) sel[i] = 255; }
  else {
    const useBelow = Math.abs(mB - 127.5) > Math.abs(mA - 127.5);
    const base = useBelow ? below : above;
    for (let i = 0; i < N; i++) if (base[i]) sel[i] = 255;
  }
  // 1px MORPH_CLOSE (RECT 2x2) heals AA breaks
  let closed = cvb.erodeMaskRect(cvb.dilateMaskRect(sel, hs, ws, 2), hs, ws, 2);
  // tightness: extra dilate 0..2 px driven by q_off
  const extraDilate = Math.max(0, Math.min(2, pyRound((60.0 - qOff) / 10.0)));
  if (extraDilate > 0) closed = cvb.dilateMaskRect(closed, hs, ws, 2);
  // box-internal connected-component cleanup
  const cc = cvb.connectedComponents(closed, hs, ws);
  const boxArea = Math.max(1, N);
  const minKeep = Math.max(4, Math.floor(boxArea * 0.001));
  const maxKeep = Math.floor(boxArea * 0.85);
  const keepLabels = new Set();
  let maxArea = -1, maxW = 0, maxH = 0;
  for (let i = 1; i < cc.n; i++) {
    const a = cc.stats[i].area;
    if (a < minKeep || a > maxKeep) continue;
    keepLabels.add(i);
    if (a > maxArea) { maxArea = a; maxW = cc.stats[i].width; maxH = cc.stats[i].height; }
  }
  const keep = new Uint8Array(N);
  if (keepLabels.size > 0) {
    const lab = cc.labels;
    for (let p = 0; p < N; p++) if (keepLabels.has(lab[p])) keep[p] = 255;
  }
  // tightness guard: a single solid block (>12% box, fill>0.70) -> the whole box is a
  // flat color block (fabric / uniform patch), not text -> drop everything.
  if (keepLabels.size > 0 && cc.n > 1) {
    let mx = -1, mw = 0, mh = 0;
    for (let i = 1; i < cc.n; i++) {
      const a = cc.stats[i].area;
      if (a > mx) { mx = a; mw = cc.stats[i].width; mh = cc.stats[i].height; }
    }
    if (mx > boxArea * 0.12 && mw > 0 && mh > 0 && (mx / (mw * mh)) > 0.70) keep.fill(0);
  }
  return keep;
}

function detectTextMaskClassic(rgb, H, W, boxes, qOff, minArea, upscale) {
  const gray = cvb.rgbToGray(rgb, H, W);
  const grayU8 = new Uint8Array(H * W);
  for (let i = 0; i < H * W; i++) { const v = gray[i]; grayU8[i] = v < 0 ? 0 : (v > 255 ? 255 : v | 0); }
  const INTER_CUBIC = cvb.getCv().INTER_CUBIC;
  const mask = new Uint8Array(H * W); // 0/255
  const pad = 8;
  for (const b of boxes) {
    const x0 = Math.max(0, b.x0 - pad), y0 = Math.max(0, b.y0 - pad);
    const x1 = Math.min(W, b.x1 + pad), y1 = Math.min(H, b.y1 + pad);
    if (x1 - x0 < 3 || y1 - y0 < 3) continue;
    const sw = x1 - x0, sh = y1 - y0;
    const sub = new Uint8Array(sw * sh);
    for (let y = 0; y < sh; y++)
      for (let x = 0; x < sw; x++) sub[y * sw + x] = grayU8[(y0 + y) * W + (x0 + x)];
    let keep = segBox(sub, sh, sw, qOff);
    const hChar = (y1 - y0) - 2 * pad;
    let sFac = 1;
    if (upscale && hChar < 56) sFac = (hChar >= 24) ? 2 : 3;
    if (sFac > 1) {
      const subUp = cvb.resizeGrayU8(sub, sh, sw, sh * sFac, sw * sFac, INTER_CUBIC);
      const keepUp = segBox(subUp, sh * sFac, sw * sFac, qOff);
      const keepUpDown = cvb.resizeFloat(Float32Array.from(keepUp), sh * sFac, sw * sFac, sh, sw);
      for (let i = 0; i < sw * sh; i++) if (keepUpDown[i] > 127) keep[i] = 255; // union (only add)
    }
    for (let y = 0; y < sh; y++)
      for (let x = 0; x < sw; x++) if (keep[y * sw + x]) mask[(y0 + y) * W + (x0 + x)] = 255;
  }
  return mask;
}

// ---------------------------------------------------------------------------
// Step 3 — bright/white completion around the mask (mirror _fill_nearby_white)
// ---------------------------------------------------------------------------
function fillNearbyWhite(rgb, H, W, mask) {
  if (!countOnes(mask)) return mask;
  const lum = cvb.rgbToGray(rgb, H, W);
  const N = H * W;
  let cur = new Uint8Array(N);
  for (let i = 0; i < N; i++) cur[i] = mask[i] ? 1 : 0;
  const PAD = 6, MIN_LUM = 200, ROUNDS = 5, MAX_DIST = 12, AA_LUM = 185, AA_DIST = 3, TAIL_LUM = 145, TAIL_ROUNDS = 6;
  const kEll = PAD * 2 + 1;
  for (let r = 0; r < ROUNDS; r++) {
    const dil = cvb.dilateMask(cur, H, W, kEll);
    const neu = new Uint8Array(N);
    let changed = false;
    for (let i = 0; i < N; i++) { const v = (cur[i] || (dil[i] && lum[i] > MIN_LUM)) ? 1 : 0; neu[i] = v; if (v !== cur[i]) changed = true; }
    if (!changed) break;
    cur = neu;
  }
  // distance-transform based completion (isolated bright segments + AA tail)
  const dist = cvb.distanceFromZeros(cur, H, W, 3);
  const add = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    if ((dist[i] <= MAX_DIST && lum[i] > MIN_LUM) || (dist[i] <= AA_DIST && lum[i] >= AA_LUM)) add[i] = 1;
  }
  { const neu = new Uint8Array(N); for (let i = 0; i < N; i++) neu[i] = (cur[i] || add[i]) ? 1 : 0; cur = neu; }
  // AA tail: continuous 1px growth along >=TAIL_LUM pixels
  if (TAIL_LUM > 0 && TAIL_ROUNDS > 0) {
    for (let r = 0; r < TAIL_ROUNDS; r++) {
      const dil = cvb.dilateMask(cur, H, W, 3);
      const neu = new Uint8Array(N);
      let changed = false;
      for (let i = 0; i < N; i++) { const v = (cur[i] || (dil[i] && lum[i] >= TAIL_LUM)) ? 1 : 0; neu[i] = v; if (v !== cur[i]) changed = true; }
      if (!changed) break;
      cur = neu;
    }
  }
  const out = new Uint8Array(N);
  for (let i = 0; i < N; i++) out[i] = cur[i] ? 255 : 0;
  return out;
}

// ---------------------------------------------------------------------------
// Step 4 — color-tint growth (mirror _grow_color_tint)
// ---------------------------------------------------------------------------
function growColorTint(rgb, H, W, mask) {
  if (!countOnes(mask)) return mask;
  const N = H * W;
  const RED_THR = 30, GREEN_THR = 15, GREEN_G = 100, ROUNDS_MAX = 120, MAX_GROW = 5.0;
  const tint = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    const R = rgb[i * 3], G = rgb[i * 3 + 1], B = rgb[i * 3 + 2];
    const redTint = (R - Math.max(G, B)) > RED_THR;
    const greenTint = ((G - Math.max(R, B)) > GREEN_THR) && (G > GREEN_G);
    tint[i] = (redTint || greenTint) ? 1 : 0;
  }
  let cur = new Uint8Array(N);
  for (let i = 0; i < N; i++) cur[i] = mask[i] ? 1 : 0;
  const cap = Math.max(1, Math.floor(countOnes(cur) * MAX_GROW));
  let prev = cur.slice();
  for (let it = 0; it < ROUNDS_MAX; it++) {
    const dil = cvb.dilateMaskRect(cur, H, W, 3); // ones(3,3)
    const neu = new Uint8Array(N);
    let any = false;
    for (let i = 0; i < N; i++) { const v = (cur[i] || (dil[i] && tint[i])) ? 1 : 0; neu[i] = v; if (v && !cur[i]) any = true; }
    if (!any) break;
    if (countOnes(neu) > cap) { cur = prev; break; }
    prev = neu.slice(); cur = neu;
  }
  const out = new Uint8Array(N);
  for (let i = 0; i < N; i++) out[i] = cur[i] ? 255 : 0;
  return out;
}

// ---------------------------------------------------------------------------
// Connected-component cleanup + box extraction (mirror _clean_text_mask / _mask_to_boxes)
// ---------------------------------------------------------------------------
function cleanTextMask(mask01, H, W, minArea, maxAreaRatio) {
  const total = H * W;
  const cc = cvb.connectedComponents(mask01, H, W);
  const cand = [];
  for (let i = 1; i < cc.n; i++) {
    const a = cc.stats[i].area;
    if (a < Math.max(minArea, 8) || a > total * maxAreaRatio) continue;
    cand.push(i);
  }
  const hs = cand.map((i) => cc.stats[i].height).sort((x, y) => x - y);
  const ws = cand.map((i) => cc.stats[i].width).sort((x, y) => x - y);
  const keep = new Uint8Array(H * W);
  const labels = cc.labels;
  cand.forEach((i, k) => {
    const s = cc.stats[i];
    let tallGate, wideGate;
    if (cand.length > 1) {
      const oh = hs.slice(0, k).concat(hs.slice(k + 1));
      const ow = ws.slice(0, k).concat(ws.slice(k + 1));
      const medH = oh[Math.floor(oh.length / 2)];
      const medW = ow[Math.floor(ow.length / 2)];
      tallGate = s.height > 1.5 * medH;
      wideGate = s.width > 1.5 * medW;
    } else {
      tallGate = wideGate = true;
    }
    if (s.width / Math.max(1, s.height) > 25 && wideGate) return; // long thin -> UI h-line
    if (s.height / Math.max(1, s.width) > 6 && tallGate) return;  // tall thin -> separator
    for (let p = 0; p < H * W; p++) if (labels[p] === i) keep[p] = 1;
  });
  return keep;
}

export function maskToBoxes(mask, H, W) {
  const mask01 = new Uint8Array(H * W);
  for (let i = 0; i < H * W; i++) mask01[i] = mask[i] ? 1 : 0;
  const cc = cvb.connectedComponents(mask01, H, W);
  const boxes = [];
  for (let i = 1; i < cc.n; i++) {
    const s = cc.stats[i];
    boxes.push({ x0: s.left, y0: s.top, x1: s.left + s.width, y1: s.top + s.height });
  }
  return boxes;
}

// ---------------------------------------------------------------------------
// Public: build the mask from a prob map + original rgb (used by Node validation
// AND by detectTextMask below). Reproduces detect_text_mask(method="ml") exactly.
// ---------------------------------------------------------------------------
export function buildMaskFromProb(rgb, H, W, prob, nw, nh, thr, opts = {}) {
  const strength = opts.strength != null ? opts.strength : 1.0;
  const qOff = opts.q_off != null ? opts.q_off : 55.0;
  const minArea = opts.minArea != null ? opts.minArea : 30;
  const maxAreaRatio = opts.maxAreaRatio != null ? opts.maxAreaRatio : 0.40;
  const maxBoxRatio = opts.maxBoxRatio != null ? opts.maxBoxRatio : 0.40;
  const fillWhite = opts.fillWhite != null ? opts.fillWhite : true;
  const fillMaxDist = opts.fillMaxDist != null ? opts.fillMaxDist : 12;
  const tintFill = opts.tintFill != null ? opts.tintFill : true;
  const upscale = opts.upscale != null ? opts.upscale : true;

  // 1) boxes
  const boxes = detectBoxesFromProb(rgb, H, W, prob, nw, nh, thr, { minArea, maxAreaRatio, maxBoxRatio, pad: 3 });
  if (boxes.length === 0) return { mask: new Uint8Array(H * W), boxes: [] };

  // 2) Otsu glyphs
  let mask = detectTextMaskClassic(rgb, H, W, boxes, qOff, minArea, upscale);
  if (!countOnes(mask)) return { mask: new Uint8Array(H * W), boxes: [] };

  // 3) bright/white completion
  if (fillWhite) mask = fillNearbyWhite(rgb, H, W, mask);
  // 4) color-tint growth
  if (tintFill) mask = growColorTint(rgb, H, W, mask);

  // 5) final cleanup
  const mask01 = new Uint8Array(H * W);
  for (let i = 0; i < H * W; i++) mask01[i] = mask[i] ? 1 : 0;
  const cleaned01 = cleanTextMask(mask01, H, W, Math.min(minArea, 8), 0.9);
  const outMask = new Uint8Array(H * W);
  for (let i = 0; i < H * W; i++) outMask[i] = cleaned01[i] ? 255 : 0;
  return { mask: outMask, boxes: maskToBoxes(outMask, H, W) };
}

/**
 * Full browser entry: run DBNet then build the mask. Mirrors the backend so the
 * browser's local mask matches the server's. Returns a Uint8Array (255 = text).
 *
 * @param {Float32Array} rgb H*W*3
 * @param {number} H
 * @param {number} W
 * @param {object} opts { session, ort?, strength=1.0, mlMaxSide=960, q_off=55,
 *                        minArea=30, maxAreaRatio=0.40, maxBoxRatio=0.40,
 *                        fillWhite=true, fillMaxDist=12, tintFill=true, upscale=true }
 */
export async function detectTextMask(rgb, H, W, opts = {}) {
  const ort = getOrt(opts.ort);
  const session = opts.session;
  if (!session) throw new Error('detectTextMask: pass a loaded `session` (see loadDBNet)');
  const strength = opts.strength != null ? opts.strength : 1.0;
  const maxSide = opts.mlMaxSide != null ? opts.mlMaxSide : 960;
  const { prob, nw, nh, thr } = await inferDbnet(rgb, H, W, { ort, session, strength, boxThreshold: 0.3, maxSide });
  const { mask } = buildMaskFromProb(rgb, H, W, prob, nw, nh, thr, opts);
  return mask;
}

export { DEFAULT_MODELS };

// loadDBNet kept for the public surface (browser engine calls it to get `session`).
export async function loadDBNet(opts = {}) {
  const ort = opts.ort || (await import('onnxruntime-web').then((m) => m));
  _ort = ort;
  const insecure = opts.insecureTLS || (typeof process !== 'undefined' && process.env && process.env.ER_INSECURE_TLS === '1');
  if (insecure && typeof process === 'undefined') {
    console.warn('[detect-dbnet] insecureTLS requested but browser fetch cannot disable TLS verification; ignoring.');
  }
  const url = await resolveModelUrl(opts.modelUrl);
  try {
    const session = await ort.InferenceSession.create(url, {
      executionProviders: opts.executionProviders || ['wasm'],
      graphOptimizationLevel: 'all',
    });
    return session;
  } catch (e) {
    throw new Error('detect-dbnet: failed to load ONNX from ' + url + ' — ' + (e && e.message ? e.message : e));
  }
}

async function resolveModelUrl(modelUrl, fetchImpl) {
  if (modelUrl && modelUrl.startsWith('/')) return modelUrl;
  const candidates = modelUrl ? [modelUrl] : DEFAULT_MODELS;
  const fetchFn = fetchImpl || ((u, o) => fetch(u, o));
  let lastErr = null;
  for (const u of candidates) {
    try {
      const head = await fetchFn(u, { method: 'HEAD' });
      if (head.ok) return u;
    } catch (e) { lastErr = e; }
    try {
      const head = await fetchFn(u, { method: 'GET', headers: { Range: 'bytes=0-0' } });
      if (head.ok || head.status === 206) return u;
    } catch (e) { lastErr = e; }
  }
  throw new Error('detect-dbnet: could not resolve a reachable DBNet ONNX URL (' + (lastErr || 'no candidates') + ')');
}
