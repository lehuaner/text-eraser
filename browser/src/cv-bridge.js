// cv-bridge.js — thin wrapper over opencv.js (the `cv2` replacement in the browser).
//
// Dependency map: cv2 -> opencv.js. We only use the subset the algorithms require:
//   inpaint(TELEA), cvtColor(RGB2GRAY), Sobel, distanceTransform, dilate/erode,
//   getStructuringElement, boxFilter(mean), connectedComponentsWithStats,
//   copyMakeBorder. All functions accept/return plain JS TypedArrays so the
//   PatchMatch / deglow cores remain `cv`-free and unit-testable.
//
// The caller must inject a ready `cv` object (opencv.js) via setCv() before use.

// ---- Shared WASM core (textcore.wasm) ---------------------------------------
// Single source of truth shared with the Python backend. When present, the
// text-mask primitives below delegate to it so the browser and backend compute
// bit-identical operators. If the wasm cannot be fetched/instantiated we fall
// back transparently to the pure-JS implementations further down this file.
import * as SharedCore from '../../shared/bindings/textcore.browser.js';

let _sharedReady = false;
let _sharedLoading = null;

export async function ensureSharedCore(url) {
  if (_sharedReady) return true;
  if (_sharedLoading) { try { await _sharedLoading; } catch (_) {} return _sharedReady; }
  _sharedLoading = (async () => {
    try {
      await SharedCore.ensure(url);
      _sharedReady = SharedCore.isReady();
    } catch (e) {
      console.warn('[cv-bridge] shared WASM core unavailable — falling back to pure-JS:', e && e.message);
      _sharedReady = false;
    }
  })();
  await _sharedLoading;
  return _sharedReady;
}

export function usingSharedCore() { return _sharedReady; }

let _cv = null;

export function setCv(cv) {
  if (!cv || typeof cv.Mat !== 'function') {
    throw new Error('cv-bridge: setCv() called with an invalid opencv.js instance');
  }
  _cv = cv;
}

export function getCv() {
  return _cv;
}

export function isCvReady() {
  return _cv !== null;
}

function cv() {
  if (!_cv) throw new Error('cv-bridge: opencv.js not loaded yet (call ensureOpenCV() first)');
  return _cv;
}

function u8FromRgbFloat(rgb) {
  const n = rgb.length;
  const u = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const v = rgb[i];
    u[i] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
  return u;
}

function rgbFloatFromU8(u8) {
  const n = u8.length, f = new Float32Array(n);
  for (let i = 0; i < n; i++) f[i] = u8[i];
  return f;
}

function maskU8From01(mask) {
  const n = mask.length, u = new Uint8Array(n);
  for (let i = 0; i < n; i++) u[i] = mask[i] ? 255 : 0;
  return u;
}

// ---- Mat helpers -----------------------------------------------------------

function matRGB(rgb, H, W) {
  const c = cv();
  const u = u8FromRgbFloat(rgb);
  const m = new c.Mat(H, W, c.CV_8UC3);
  m.data.set(u);
  return m;
}

function matGray8(gray, H, W) {
  const c = cv();
  const u = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i++) {
    const v = gray[i];
    u[i] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
  const m = new c.Mat(H, W, c.CV_8UC1);
  m.data.set(u);
  return m;
}

function matMask(mask01, H, W) {
  return matGray8(maskU8From01(mask01), H, W);
}

// ---- Public ops ------------------------------------------------------------

/** RGB Float32Array (H*W*3) -> grayscale Float32Array (H*W), 0..255.
 *  Pure JS, matching cv2.cvtColor(RGB2GRAY) exact fixed-point: (R*4899+G*9617+B*1868+8192)>>14.
 *  Avoids opencv.js cvtColor (its WASM scratch buffers can be heap-state dependent). */
export function rgbToGray(rgb, H, W) {
  if (_sharedReady) {
    const g = SharedCore.rgbToGray(rgb, H, W); // Uint8Array H*W
    const out = new Float32Array(H * W);
    for (let i = 0; i < H * W; i++) out[i] = g[i];
    return out;
  }
  const out = new Float32Array(H * W);
  for (let i = 0; i < H * W; i++) {
    const r = rgb[i * 3] | 0, g = rgb[i * 3 + 1] | 0, b = rgb[i * 3 + 2] | 0;
    out[i] = (r * 4899 + g * 9617 + b * 1868 + 8192) >> 14;
  }
  return out;
}

/** Sobel gradients on a grayscale Float32Array. Returns {gx, gy, mag} (CV_32F).
 *  Source is a 32F mat (mirrors Python `cv2.Sobel(gray0.astype(float32), ...)`). */
export function sobel(gray, H, W, ksize = 3) {
  const c = cv();
  const src = new c.Mat(H, W, c.CV_32FC1);
  for (let i = 0; i < H * W; i++) src.data32F[i] = gray[i];
  const gx = new c.Mat(H, W, c.CV_32FC1);
  const gy = new c.Mat(H, W, c.CV_32FC1);
  c.Sobel(src, gx, c.CV_32F, 1, 0, ksize);
  c.Sobel(src, gy, c.CV_32F, 0, 1, ksize);
  const gxf = new Float32Array(H * W);
  const gyf = new Float32Array(H * W);
  for (let i = 0; i < H * W; i++) { gxf[i] = gx.data32F[i]; gyf[i] = gy.data32F[i]; }
  const mag = new Float32Array(H * W);
  for (let i = 0; i < H * W; i++) mag[i] = Math.hypot(gxf[i], gyf[i]);
  src.delete(); gx.delete(); gy.delete();
  return { gx: gxf, gy: gyf, mag };
}

/** OpenCV TELEA inpainting. rgb: Float32Array H*W*3, mask01: 0/1 hole. Returns rgb. */
export function inpaintTelea(rgb, mask01, H, W, radius = 3) {
  const c = cv();
  const src = matRGB(rgb, H, W);
  const m = matMask(mask01, H, W);
  const dst = new c.Mat(H, W, c.CV_8UC3);
  c.inpaint(src, m, dst, radius, c.INPAINT_TELEA);
  const out = new Float32Array(H * W * 3);
  for (let i = 0; i < H * W * 3; i++) out[i] = dst.data[i];
  src.delete(); m.delete(); dst.delete();
  return out;
}

// ---- Pure-JS binary morphology (deterministic; avoids opencv.js WASM heap quirks) ----
// Structuring-element offsets mirroring cv.getStructuringElement for RECT and ELLIPSE.
function morphOffsets(shape, ksize) {
  const offs = [];
  const anchor = Math.floor(ksize / 2);   // matches cv getStructuringElement anchor
  const ca = (ksize - 1) / 2;            // float center for ellipse test
  for (let ky = 0; ky < ksize; ky++) {
    for (let kx = 0; kx < ksize; kx++) {
      const dx = kx - anchor, dy = ky - anchor;
      if (shape === 'rect') { offs.push([dx, dy]); }
      else { if (dx * dx + dy * dy <= ca * ca + 1e-6) offs.push([dx, dy]); } // ellipse
    }
  }
  return offs;
}

/** Build the exact opencv.js structuring-element bitmap (0/1 Uint8Array) for the wasm core. */
function kernBitmap(shape, ksize) {
  const c = cv();
  const se = c.getStructuringElement(shape === 'rect' ? c.MORPH_RECT : c.MORPH_ELLIPSE, new c.Size(ksize, ksize));
  const data = se.data; // opencv.js Mat data: 0/255
  const k = new Uint8Array(data.length);
  for (let i = 0; i < data.length; i++) k[i] = data[i] ? 1 : 0;
  se.delete();
  return { k, kh: ksize, kw: ksize };
}
function morphOp(mask01, H, W, offs, isDilate) {
  const out = new Uint8Array(H * W);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (isDilate) {
        let hit = false;
        for (let o = 0; o < offs.length; o++) {
          const nx = x + offs[o][0], ny = y + offs[o][1];
          if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
          if (mask01[ny * W + nx]) { hit = true; break; }
        }
        out[i] = hit ? 1 : 0;
      } else {
        let all = true;
        for (let o = 0; o < offs.length; o++) {
          const nx = x + offs[o][0], ny = y + offs[o][1];
          if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
          if (!mask01[ny * W + nx]) { all = false; break; }
        }
        out[i] = all ? 1 : 0;
      }
    }
  }
  return out;
}

/** Dilate a 0/1 mask with an ellipse kernel of diameter `ksize`. Returns 0/1 Uint8Array. */
export function dilateMask(mask01, H, W, ksize) {
  if (_sharedReady) {
    const { k, kh, kw } = kernBitmap('ellipse', ksize);
    return SharedCore.morphology(mask01, H, W, k, kh, kw, 'dilate');
  }
  return morphOp(mask01, H, W, morphOffsets('ellipse', ksize), true);
}

/** Erode a 0/1 mask. */
export function erodeMask(mask01, H, W, ksize) {
  if (_sharedReady) {
    const { k, kh, kw } = kernBitmap('ellipse', ksize);
    return SharedCore.morphology(mask01, H, W, k, kh, kw, 'erode');
  }
  return morphOp(mask01, H, W, morphOffsets('ellipse', ksize), false);
}

/** Dilate a 0/1 mask with a RECT (square) kernel — matches cv2.dilate(..., np.ones((k,k))). */
export function dilateMaskRect(mask01, H, W, ksize) {
  if (_sharedReady) {
    const { k, kh, kw } = kernBitmap('rect', ksize);
    return SharedCore.morphology(mask01, H, W, k, kh, kw, 'dilate');
  }
  return morphOp(mask01, H, W, morphOffsets('rect', ksize), true);
}

/** Erode a 0/1 mask with a RECT (square) kernel — matches cv2.erode(..., np.ones((k,k))). */
export function erodeMaskRect(mask01, H, W, ksize) {
  if (_sharedReady) {
    const { k, kh, kw } = kernBitmap('rect', ksize);
    return SharedCore.morphology(mask01, H, W, k, kh, kw, 'erode');
  }
  return morphOp(mask01, H, W, morphOffsets('rect', ksize), false);
}

/**
 * Distance transform to the FOREGROUND (mask==1) pixels, returning Float32Array (H*W).
 *
 * Mirrors the backend `_fill_nearby_white`:
 *     cv2.distanceTransform((cur == 0).astype(uint8), cv2.DIST_L2, 3)
 * In `(cur == 0)` the text pixels are 0 and the background is 1, so
 * cv2.distanceTransform (which measures distance to the nearest zero) returns the
 * distance from every pixel to the nearest TEXT pixel. To get the same result from a
 * direct mask we seed every foreground (mask==1) pixel as a distance-0 source and
 * everything else as INF, then run the exact EDT. (The previous opencv.js build did
 * `inv[i] = mask01[i] ? 0 : 255` then `cv.distanceTransform` — equivalent.)
 */
export function distanceFromZeros(mask01, H, W, maskSize = 3) {
  if (_sharedReady) return SharedCore.distanceTransformEdt(mask01, H, W);
  // Exact Euclidean distance transform (Felzenszwalb & Huttenlocher) of the
  // foreground pixels. Deterministic, matching cv2.distanceTransform(DIST_L2).
  const N = H * W;
  const INF = 1e20;
  const f = new Float64Array(N);
  for (let i = 0; i < N; i++) f[i] = mask01[i] ? 0 : INF; // sources are foreground (text)
  const grid = new Float64Array(Math.max(H, W));
  const d = new Float64Array(Math.max(H, W));
  const v = new Int32Array(Math.max(H, W) + 1);
  const z = new Float64Array(Math.max(H, W) + 2);
  // horizontal pass
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) grid[x] = f[y * W + x];
    edt1d(grid, d, v, z, W);
    for (let x = 0; x < W; x++) f[y * W + x] = d[x];
  }
  // vertical pass
  for (let x = 0; x < W; x++) {
    for (let y = 0; y < H; y++) grid[y] = f[y * W + x];
    edt1d(grid, d, v, z, H);
    for (let y = 0; y < H; y++) f[y * W + x] = d[y];
  }
  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) out[i] = Math.sqrt(f[i]);
  return out;
}
function edt1d(f, d, v, z, n) {
  let k = 0;
  v[0] = 0; z[0] = -1e20; z[1] = 1e20;
  for (let q = 1; q < n; q++) {
    let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (s <= z[k]) {
      k--;
      s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    }
    k++; v[k] = q; z[k] = s; z[k + 1] = 1e20;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    const dist = q - v[k];
    d[q] = dist * dist + f[v[k]];
  }
}

/**
 * Mean box filter (normalize=false) over a k×k window, divided by k².
 * src: 0/1 (or 0/255) Uint8Array H*W. Returns Float32Array H*W in [0,1] if src is 0/1.
 */
export function boxFilterMean(src01, H, W, k) {
  const c = cv();
  const src = matMask(src01, H, W);
  const dst = new c.Mat(H, W, c.CV_32FC1);
  c.boxFilter(src, dst, c.CV_32FC1, new c.Size(k, k), new c.Point(-1, -1), false);
  const out = new Float32Array(H * W);
  const inv = 1.0 / (k * k);
  for (let i = 0; i < H * W; i++) out[i] = dst.data32F[i] * inv;
  src.delete(); dst.delete();
  return out;
}

/**
 * Connected components with stats (8-connectivity).
 * mask01: Uint8Array H*W (255/non-zero = foreground).
 * Returns { n, labels: Int32Array H*W, stats: [{area,left,top,width,height}] }.
 *
 * NOTE: implemented in pure JS (deterministic flood-fill) on purpose. The opencv.js
 * `connectedComponentsWithStats` C++ binding leaves label/scratch buffers uninitialized
 * in some WASM builds, so results depend on leftover WASM-heap contents — which vary
 * per process load (and with prior allocations of larger Mats). That made the mask
 * non-deterministic for small images. A JS CC is bit-for-bit reproducible.
 */
export function connectedComponents(mask01, H, W) {
  if (_sharedReady) return SharedCore.connectedComponents(mask01, H, W);
  const N = H * W;
  const labels = new Int32Array(N);
  const queue = new Int32Array(N);
  const DX = [-1, -1, -1, 0, 0, 1, 1, 1];
  const DY = [-1, 0, 1, -1, 1, -1, 0, 1];
  // Index 0 is the background placeholder so that cc.stats[i] aligns with label i,
  // matching opencv.js connectedComponentsWithStats' convention.
  const stats = [{ left: 0, top: 0, width: 0, height: 0, area: 0 }];
  let n = 1; // number of labels including background (cv returns K+1)
  for (let s = 0; s < N; s++) {
    if (!mask01[s] || labels[s] !== 0) continue;
    const comp = n++;
    let minX = W, minY = H, maxX = -1, maxY = -1, area = 0;
    labels[s] = comp;
    let head = 0, tail = 0;
    queue[tail++] = s;
    while (head < tail) {
      const p = queue[head++];
      const py = (p / W) | 0, px = p - py * W;
      if (px < minX) minX = px; if (px > maxX) maxX = px;
      if (py < minY) minY = py; if (py > maxY) maxY = py;
      area++;
      for (let k = 0; k < 8; k++) {
        const nx = px + DX[k], ny = py + DY[k];
        if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
        const np = ny * W + nx;
        if (mask01[np] && labels[np] === 0) { labels[np] = comp; queue[tail++] = np; }
      }
    }
    stats.push({ left: minX, top: minY, width: maxX - minX + 1, height: maxY - minY + 1, area });
  }
  return { n, labels, stats };
}

/** Replicate-border pad of an RGB Float32Array. Returns {rgb, H, W}. */
export function copyMakeBorderReplicate(rgb, H, W, t, b, l, r) {
  const c = cv();
  const src = matRGB(rgb, H, W);
  const dst = new c.Mat();
  c.copyMakeBorder(src, dst, t, b, l, r, c.BORDER_REPLICATE);
  const H2 = dst.rows, W2 = dst.cols;
  const out = new Float32Array(H2 * W2 * 3);
  for (let i = 0; i < H2 * W2 * 3; i++) out[i] = dst.data[i];
  src.delete(); dst.delete();
  return { rgb: out, H: H2, W: W2 };
}

/** Pad a 0/1 mask with constant 0 (no cv needed, but kept here for symmetry). */
export function padMask(mask01, H, W, t, b, l, r) {
  const H2 = H + t + b, W2 = W + l + r;
  const out = new Uint8Array(H2 * W2);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      out[(y + t) * W2 + (x + l)] = mask01[y * W + x] ? 1 : 0;
    }
  }
  return out;
}

/** Bilinear resize of an RGB Float32Array (H*W*3). Returns Float32Array (H2*W2*3). */
export function resizeRgb(rgb, H, W, H2, W2) {
  const c = cv();
  const src = matRGB(rgb, H, W);
  const dst = new c.Mat(H2, W2, c.CV_8UC3);
  const sz = new c.Size(W2, H2);
  c.resize(src, dst, sz, 0, 0, c.INTER_LINEAR);
  const out = new Float32Array(H2 * W2 * 3);
  for (let i = 0; i < H2 * W2 * 3; i++) out[i] = dst.data[i];
  src.delete(); dst.delete();
  return out;
}

/** Resize with INTER_AREA (mirrors cv2.resize(..., INTER_AREA)); channel order is
 *  preserved (resize is per-channel), so an RGB-ordered or BGR-ordered Float32Array
 *  both work. Used for DBNet pre-processing to match the backend exactly. */
export function resizeRgbArea(rgb, H, W, H2, W2) {
  const c = cv();
  const src = matRGB(rgb, H, W);
  const dst = new c.Mat(H2, W2, c.CV_8UC3);
  const sz = new c.Size(W2, H2);
  c.resize(src, dst, sz, 0, 0, c.INTER_AREA);
  const out = new Float32Array(H2 * W2 * 3);
  for (let i = 0; i < H2 * W2 * 3; i++) out[i] = dst.data[i];
  src.delete(); dst.delete();
  return out;
}

/** Otsu threshold of a single-channel U8 array. Returns { thr (0..255), bin (0/255) }.
 *  Mirrors cv2.threshold(gs, 0, 255, THRESH_BINARY | THRESH_OTSU).
 *
 *  NOTE: implemented in pure JS on purpose. The opencv.js WASM `threshold(OTSU)`
 *  computes its histogram in a scratch buffer that can contain leftover WASM-heap
 *  data for near-empty bins; on small crops that shifts the Otsu argmax between
 *  process loads (and away from the cv2-CPU threshold the backend uses), making the
 *  glyph split diverge. A JS Otsu is bit-for-bit reproducible and matches cv2. */
export function thresholdOtsu(u8, H, W) {
  if (_sharedReady) {
    const { thr, bin } = SharedCore.thresholdOtsu(u8, H, W); // bin: Uint8Array 0/255
    return { thr, bin };
  }
  const N = H * W;
  const hist = new Float64Array(256);
  let sum = 0;
  for (let i = 0; i < N; i++) { const v = u8[i]; hist[v]++; sum += v; }
  let sumB = 0, wB = 0, maxBetween = -1, thr = 0;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = N - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > maxBetween) { maxBetween = between; thr = t; }
  }
  const bin = new Uint8Array(N);
  for (let i = 0; i < N; i++) bin[i] = u8[i] > thr ? 255 : 0;
  return { thr, bin };
}

/** Cubic (cv2 INTER_CUBIC, a=-0.75) upscale of a U8 single-channel array. Pure JS,
 *  deterministic, matching cv2.resize(interp=INTER_CUBIC). Used by the small-text
 *  upscale path in detectTextMaskClassic. */
export function resizeGrayU8(u8, H, W, H2, W2, interp) {
  if (_sharedReady) return SharedCore.resizeGrayCubic(u8, H, W, H2, W2);
  const a = -0.75;
  const cubic = (x) => { x = Math.abs(x); if (x <= 1) return (a + 2) * x * x * x - (a + 3) * x * x + 1; if (x < 2) return a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a; return 0; };
  const tmp = new Float32Array(H * W2);
  const sx = W / W2;
  for (let y = 0; y < H; y++) {
    for (let x2 = 0; x2 < W2; x2++) {
      const center = (x2 + 0.5) * sx - 0.5;
      const ix = Math.floor(center), fx = center - ix;
      const w0 = cubic(1 + fx), w1 = cubic(fx), w2 = cubic(1 - fx), w3 = cubic(2 - fx);
      const i0 = ix - 1 < 0 ? 0 : ix - 1 >= W ? W - 1 : ix - 1;
      const i1 = ix < 0 ? 0 : ix >= W ? W - 1 : ix;
      const i2 = ix + 1 < 0 ? 0 : ix + 1 >= W ? W - 1 : ix + 1;
      const i3 = ix + 2 < 0 ? 0 : ix + 2 >= W ? W - 1 : ix + 2;
      tmp[y * W2 + x2] = u8[y * W + i0] * w0 + u8[y * W + i1] * w1 + u8[y * W + i2] * w2 + u8[y * W + i3] * w3;
    }
  }
  const out = new Uint8Array(H2 * W2);
  const sy = H / H2;
  for (let y2 = 0; y2 < H2; y2++) {
    const center = (y2 + 0.5) * sy - 0.5;
    const iy = Math.floor(center), fy = center - iy;
    const w0 = cubic(1 + fy), w1 = cubic(fy), w2 = cubic(1 - fy), w3 = cubic(2 - fy);
    const j0 = iy - 1 < 0 ? 0 : iy - 1 >= H ? H - 1 : iy - 1;
    const j1 = iy < 0 ? 0 : iy >= H ? H - 1 : iy;
    const j2 = iy + 1 < 0 ? 0 : iy + 1 >= H ? H - 1 : iy + 1;
    const j3 = iy + 2 < 0 ? 0 : iy + 2 >= H ? H - 1 : iy + 2;
    for (let x2 = 0; x2 < W2; x2++) {
      const s = tmp[j0 * W2 + x2] * w0 + tmp[j1 * W2 + x2] * w1 + tmp[j2 * W2 + x2] * w2 + tmp[j3 * W2 + x2] * w3;
      out[y2 * W2 + x2] = s < 0 ? 0 : s > 255 ? 255 : Math.round(s);
    }
  }
  return out;
}

/** Bilinear (cv2 INTER_LINEAR) resize of a Float32Array. Pure JS, deterministic,
 *  matching cv2.resize(interp=INTER_LINEAR). Used to downscale the upscaled Otsu mask. */
export function resizeFloat(src, H, W, H2, W2) {
  if (_sharedReady) return SharedCore.resizeFloatLinear(src, H, W, H2, W2);
  const out = new Float32Array(H2 * W2);
  const sx = W / W2, sy = H / H2;
  for (let y2 = 0; y2 < H2; y2++) {
    const cy = (y2 + 0.5) * sy - 0.5;
    const iy = Math.floor(cy), fy = cy - iy;
    const y0 = iy < 0 ? 0 : iy >= H ? H - 1 : iy;
    const y1 = iy + 1 < 0 ? 0 : iy + 1 >= H ? H - 1 : iy + 1;
    for (let x2 = 0; x2 < W2; x2++) {
      const cx = (x2 + 0.5) * sx - 0.5;
      const ix = Math.floor(cx), fx = cx - ix;
      const x0 = ix < 0 ? 0 : ix >= W ? W - 1 : ix;
      const x1 = ix + 1 < 0 ? 0 : ix + 1 >= W ? W - 1 : ix + 1;
      const v00 = src[y0 * W + x0], v01 = src[y0 * W + x1], v10 = src[y1 * W + x0], v11 = src[y1 * W + x1];
      const top = v00 * (1 - fx) + v01 * fx;
      const bot = v10 * (1 - fx) + v11 * fx;
      out[y2 * W2 + x2] = top * (1 - fy) + bot * fy;
    }
  }
  return out;
}

/**
 * Shared PatchMatch fill — THE single source of truth shared with the Python backend.
 * Delegates to textcore.wasm (Rust implementation). Returns the filled ROI as a
 * Float32Array (H*W*3), or null if the wasm core isn't ready (caller falls back to
 * the pure-JS inpaint loop below).
 *
 * @param {Float32Array} rgb  H*W*3 ROI (already padded/ROI-cropped by the caller).
 * @param {Uint8Array} mask   H*W, >0 = hole to fill.
 * @param {Uint8Array|null} sample  H*W, >0 = allowed source region.
 * @param {number} p  patch size (odd).
 * @param {number|null} direction  angle degrees, or null to disable.
 */
export function patchmatchInpaintShared(rgb, H, W, mask, sample, p, direction, seed) {
  if (!_sharedReady) return null;
  const deg = direction == null ? -1.0 : direction;
  try {
    return SharedCore.patchmatchInpaint(rgb, H, W, mask, sample, p, deg, seed || 0);
  } catch (e) {
    console.warn('[cv-bridge] shared PatchMatch failed, falling back to JS:', e && e.message);
    return null;
  }
}
