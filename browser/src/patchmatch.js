// patchmatch.js — browser port of text_eraser.patch_fill.inpaint
//
// Faithful, quality-first port. The PatchMatch core is pure JS over TypedArrays
// (numpy replacement). The few cv2 ops it needs (cvtColor, Sobel, boxFilter,
// inpaint TELEA, ellipse dilate) come from cv-bridge (opencv.js). Binary rect
// morphology and the 3x3 boundary erosion run in pure JS (identical to cv2.rect
// morphologyEx) to avoid per-iteration opencv.js Mat churn in the hot loop.
//
// Behaviour matches core/textpatch_fill.py:
//   * iterative Criminisi-style priority fill (confidence × gradient data-term)
//   * PatchMatch random + neighbourhood-coherence search per target block
//   * local colour self-adaptation (anchor = original-known snapshot, expanding
//     window 7→11→17 to protect boundary blocks)
//   * smooth-gradient TELEA fallback (tex < flat_tex) — see note in inpaint()
//   * direction fill mode (samples only along a line through each target pixel)
//   * residual-hole TELEA cleanup
//
// The forward-traversal batching that upstream uses for speed is intentionally
// NOT reproduced ("不得为提速合并前向遍历"): we process each target block with its
// own best-source search, which is the conceptual basis and keeps output faithful.

import * as cvb from './cv-bridge.js';
import { median, mulberry32, erodeRect, erode3x3, dilateMax3x3 } from './linalg.js';

const P = 7;          // patch size (odd)
const HALF = P >> 1; // 3
const PP = P * P;     // 49
const PP3 = PP * 3;   // 147
const PADM = 4;       // safety border so P×P patches never run off the edge
const MAX_ROI = 1536;

// ---- low-level patch helpers (operate on a sh×sw RGB buffer) ----------------

function gatherPatch(buf, cy, cx, H, W) {
  const out = new Float32Array(PP3);
  let o = 0;
  for (let dy = -HALF; dy <= HALF; dy++) {
    let yy = cy + dy;
    if (yy < 0) yy = 0; else if (yy >= H) yy = H - 1;
    for (let dx = -HALF; dx <= HALF; dx++) {
      let xx = cx + dx;
      if (xx < 0) xx = 0; else if (xx >= W) xx = W - 1;
      const s = (yy * W + xx) * 3;
      out[o++] = buf[s]; out[o++] = buf[s + 1]; out[o++] = buf[s + 2];
    }
  }
  return out;
}

function patchMean(p) {
  let m0 = 0, m1 = 0, m2 = 0;
  for (let i = 0; i < PP; i++) { const o = i * 3; m0 += p[o]; m1 += p[o + 1]; m2 += p[o + 2]; }
  return [m0 / PP, m1 / PP, m2 / PP];
}

function cropRGB(buf, pH, pW, padm, OH, OW) {
  const out = new Float32Array(OH * OW * 3);
  for (let y = 0; y < OH; y++) {
    for (let x = 0; x < OW; x++) {
      const s = ((y + padm) * pW + (x + padm)) * 3;
      const d = (y * OW + x) * 3;
      out[d] = buf[s]; out[d + 1] = buf[s + 1]; out[d + 2] = buf[s + 2];
    }
  }
  return out;
}

/**
 * Core inpaint. Mutates/returns an RGB Float32Array (H*W*3, 0..255).
 *
 * @param {Float32Array} rgb    H*W*3 RGB (working copy).
 * @param {number} H
 * @param {number} W
 * @param {Uint8Array} mask255  H*W, >0 = hole to fill.
 * @param {object} opts { sampleMask:Uint8Array|null, direction:number|null,
 *                         flatSpan:number, flatTex:number, shouldCancel:fn|null }
 */
export function patchmatchInpaint(rgb, H, W, mask255, opts = {}) {
  const {
    sampleMask: sampleMask255 = null,
    direction = null,
    flatSpan = 40,
    flatTex = 20.0,
    shouldCancel = null,
  } = opts;

  const n = H * W;
  const m = new Uint8Array(n);
  for (let i = 0; i < n; i++) m[i] = mask255[i] ? 1 : 0;
  let anyHole = false;
  for (let i = 0; i < n; i++) { if (m[i]) { anyHole = true; break; } }
  if (!anyHole) return rgb;

  const OH = H, OW = W;

  // ---- smooth-gradient TELEA fallback (mirrors textpatch_fill.py) ----
  // NOTE on fidelity: upstream (patch_fill.inpaint) fires TELEA when the ring
  // texture median `tex < flat_tex` ONLY (the span>=flatSpan requirement was
  // removed; `flatSpan` is kept in the signature for parity but the live gate
  // is `tex < flatTex`). flatTex MUST equal Python's flat_tex default (20.0):
  // a background with tex in [15, 20) would take TELEA on the backend but
  // PatchMatch here. Flip FLAT_USE_SPAN=true to also require span>=flatSpan
  // (the stricter spec prose variant).
  const FLAT_USE_SPAN = false;
  let spanCheck = true;
  if (direction === null) {
    const gray0 = cvb.rgbToGray(rgb, H, W);
    let y0 = H, y1 = -1, x0 = W, x1 = -1;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      if (m[y * W + x]) { if (y < y0) y0 = y; if (y > y1) y1 = y; if (x < x0) x0 = x; if (x > x1) x1 = x; }
    }
    const band = 12;
    const medians = [];
    const bandCollect = (r0, r1, c0, c1) => {
      const vals = [];
      for (let y = r0; y <= r1; y++) for (let x = c0; x <= c1; x++) {
        const idx = y * W + x; if (!m[idx]) vals.push(gray0[idx]);
      }
      return vals;
    };
    if (y1 >= y0 && x1 >= x0) {
      medians.push(...bandCollect(Math.max(0, y0 - band), y0, x0, x1));
      medians.push(...bandCollect(y1, Math.min(H - 1, y1 + band), x0, x1));
      medians.push(...bandCollect(y0, y1, Math.max(0, x0 - band), x0));
      medians.push(...bandCollect(y0, y1, x1, Math.min(W - 1, x1 + band)));
    }
    if (medians.length >= 2) {
      const svals = medians.slice().sort((a, b) => a - b);
      const span = svals[svals.length - 1] - svals[0];
      spanCheck = span >= flatSpan;
      const { gx, gy } = cvb.sobel(gray0, H, W, 3);
      const grad0 = new Float32Array(n);
      for (let i = 0; i < n; i++) grad0[i] = Math.hypot(gx[i], gy[i]);
      const dil = cvb.dilateMaskRect(m, H, W, 41);
      const ring0 = new Uint8Array(n);
      for (let i = 0; i < n; i++) ring0[i] = (dil[i] && !m[i]) ? 1 : 0;
      let tex = 0;
      if (ring0.some((v) => v)) {
        const rv = [];
        for (let i = 0; i < n; i++) if (ring0[i]) rv.push(grad0[i]);
        tex = median(rv);
      }
      const fire = FLAT_USE_SPAN ? (spanCheck && tex < flatTex) : (tex < flatTex);
      if (fire) {
        const { rgb: prgb, H: pH, W: pW } = cvb.copyMakeBorderReplicate(rgb, OH, OW, PADM, PADM, PADM, PADM);
        const pmp = new Uint8Array(pH * pW);
        for (let y = 0; y < OH; y++) for (let x = 0; x < OW; x++) pmp[(y + PADM) * pW + (x + PADM)] = m[y * W + x] ? 1 : 0;
        const out = cvb.inpaintTelea(prgb, pmp, pH, pW, 3);
        return cropRGB(out, pH, pW, PADM, OH, OW);
      }
    }
  }

  // ---- pad with replicate border ----
  const { rgb: work, H: curH, W: curW } = cvb.copyMakeBorderReplicate(rgb, OH, OW, PADM, PADM, PADM, PADM);
  const curMask = new Uint8Array(curH * curW);
  for (let y = 0; y < OH; y++) for (let x = 0; x < OW; x++) curMask[(y + PADM) * curW + (x + PADM)] = m[y * W + x] ? 1 : 0;
  let psm = null;
  if (sampleMask255) {
    psm = new Uint8Array(curH * curW);
    for (let y = 0; y < OH; y++) for (let x = 0; x < OW; x++) {
      psm[(y + PADM) * curW + (x + PADM)] = sampleMask255[y * W + x] ? 1 : 0;
    }
  }

  // ---- local ROI ----
  let hy0 = curH, hy1 = -1, hx0 = curW, hx1 = -1;
  for (let i = 0; i < curH * curW; i++) if (curMask[i]) {
    const y = (i / curW) | 0, x = i % curW;
    if (y < hy0) hy0 = y; if (y > hy1) hy1 = y; if (x < hx0) hx0 = x; if (x > hx1) hx1 = x;
  }
  let margin = Math.max(32, Math.floor(0.6 * Math.max(hy1 - hy0, hx1 - hx0)));
  if (psm) margin = Math.max(margin, Math.floor(0.9 * Math.max(hy1 - hy0, hx1 - hx0)), 80);
  let y0r = Math.max(0, hy0 - margin), y1r = Math.min(curH, hy1 + margin + 1);
  let x0r = Math.max(0, hx0 - margin), x1r = Math.min(curW, hx1 + margin + 1);
  while (Math.max(y1r - y0r, x1r - x0r) > MAX_ROI && margin > 24) {
    margin = Math.floor(margin * 0.85);
    y0r = Math.max(0, hy0 - margin); y1r = Math.min(curH, hy1 + margin + 1);
    x0r = Math.max(0, hx0 - margin); x1r = Math.min(curW, hx1 + margin + 1);
  }
  const sw = x1r - x0r, sh = y1r - y0r;
  const sub = new Float32Array(sh * sw * 3);
  const subm = new Uint8Array(sh * sw);
  const subsm = psm ? new Uint8Array(sh * sw) : null;
    for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) {
      const si = (y * sw + x) * 3;
      const wi = ((y + y0r) * curW + (x + x0r)) * 3;
      sub[si] = work[wi]; sub[si + 1] = work[wi + 1]; sub[si + 2] = work[wi + 2];
      subm[y * sw + x] = curMask[(y + y0r) * curW + (x + x0r)] ? 1 : 0;
      if (subsm) subsm[y * sw + x] = psm[(y + y0r) * curW + (x + x0r)] ? 1 : 0;
    }

    // ---- shared-core fill (single source of truth, identical to the backend) ----
    // Replaces the JS PatchMatch loop below with the SAME Rust implementation the
    // Python backend calls. The pre-check (TELEA) and any residual cleanup still use
    // opencv.js, which is byte-identical across platforms. The pure-JS loop remains
    // as the fallback when the wasm core is unavailable.
    const _sharedFill = cvb.patchmatchInpaintShared(sub, sh, sw, subm, subsm, P, direction, 0);
    if (_sharedFill) {
      for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) {
        const si = (y * sw + x) * 3;
        const wi = ((y + y0r) * curW + (x + x0r)) * 3;
        work[wi] = _sharedFill[si];
        work[wi + 1] = _sharedFill[si + 1];
        work[wi + 2] = _sharedFill[si + 2];
      }
      return cropRGB(work, curH, curW, PADM, OH, OW);
    }

    const known = new Uint8Array(sh * sw);
  for (let i = 0; i < sh * sw; i++) known[i] = subm[i] ? 0 : 1;
  const origKnown = known.slice();
  const hole = subm.slice();
  const filled = sub; // modified in place

  // candidate source centres: erode(known, P×P) then drop half-px border
  let candMask = erodeRect(known, sh, sw, P);
  for (let y = 0; y < HALF; y++) for (let x = 0; x < sw; x++) { candMask[y * sw + x] = 0; candMask[(sh - 1 - y) * sw + x] = 0; }
  for (let x = 0; x < HALF; x++) for (let y = 0; y < sh; y++) { candMask[y * sw + x] = 0; candMask[y * sw + (sw - 1 - x)] = 0; }
  if (subsm) for (let i = 0; i < sh * sw; i++) if (!subsm[i]) candMask[i] = 0;
  const candY = [], candX = [];
  for (let i = 0; i < sh * sw; i++) if (candMask[i]) { candY.push((i / sw) | 0); candX.push(i % sw); }
  const Nc = candY.length;

  if (Nc === 0) {
    const out = cvb.inpaintTelea(work, curMask, curH, curW, 3);
    return cropRGB(out, curH, curW, PADM, OH, OW);
  }

  // direction mode precompute
  let dirVec = null;
  if (direction !== null) {
    const rad = (direction * Math.PI) / 180.0;
    const ux = Math.cos(rad), uy = Math.sin(rad);
    const maxd = Math.floor(Math.hypot(sh, sw)) + 1;
    dirVec = { ux, uy, maxd, step: 2 };
  }

  // data term: gradient magnitude of known region, dilated 3x3
  const gray = cvb.rgbToGray(filled, sh, sw);
  const s2 = cvb.sobel(gray, sh, sw, 3);
  const grad = s2.mag;
  const gk = new Float32Array(sh * sw);
  for (let i = 0; i < sh * sw; i++) gk[i] = grad[i] * known[i];
  const Dmap = dilateMax3x3(gk, sh, sw);

  // NNF
  const nnfY = new Int32Array(sh * sw);
  const nnfX = new Int32Array(sh * sw);
  const nnfSet = new Uint8Array(sh * sw);

  // precompute candidate source patches + means (sources live in known region,
  // which is never overwritten, so this stays valid for the whole loop)
  const srcPatches = new Float32Array(Nc * PP3);
  const srcMean = new Float32Array(Nc * 3);
  for (let ci = 0; ci < Nc; ci++) {
    const p = gatherPatch(filled, candY[ci], candX[ci], sh, sw);
    srcPatches.set(p, ci * PP3);
    const mm = patchMean(p);
    srcMean[ci * 3] = mm[0]; srcMean[ci * 3 + 1] = mm[1]; srcMean[ci * 3 + 2] = mm[2];
  }

  const rng = mulberry32(0);
  const K = Math.min(256, Math.max(32, (Nc / 4) | 0));
  const dy = [], dx = [];
  for (let d = -HALF; d <= HALF; d++) { dy.push(d); dx.push(d); }
  const dy4 = [-1, 1, 0, 0], dx4 = [0, 0, -1, 1];

  // reusable scratch for the pool
  const poolCY = new Int32Array(K + 4);
  const poolCX = new Int32Array(K + 4);
  const poolPatches = new Float32Array((K + 4) * PP3);
  const poolMean = new Float32Array((K + 4) * 3);

  function bestSource(ty, tx) {
    const tPatch = gatherPatch(filled, ty, tx, sh, sw);
    // known positions in target window
    const tkIdx = [];
    for (let j = 0; j < PP; j++) {
      const wy = ty + dy[j % P], wx = tx + dx[(j / P) | 0];
      if (wy >= 0 && wy < sh && wx >= 0 && wx < sw && known[wy * sw + wx]) tkIdx.push(j);
    }
    let nPool = 0;
    if (dirVec) {
      const { ux, uy, maxd, step } = dirVec;
      const lineY = [], lineX = [];
      for (const sign of [1, -1]) {
        for (let d = step; d <= maxd; d += step) {
          const cy = Math.round(ty + sign * d * uy);
          const cx = Math.round(tx + sign * d * ux);
          if (cy < HALF || cy >= sh - HALF || cx < HALF || cx >= sw - HALF) continue;
          const ok = subsm ? subsm[cy * sw + cx] : known[cy * sw + cx];
          if (ok) { lineY.push(cy); lineX.push(cx); }
        }
      }
      if (lineY.length === 0) {
        for (let r = 0; r < K; r++) { const ci = (rng() * Nc) | 0; poolCY[nPool] = candY[ci]; poolCX[nPool] = candX[ci]; poolPatches.set(srcPatches.subarray(ci * PP3, ci * PP3 + PP3), nPool * PP3); poolMean[nPool * 3] = srcMean[ci * 3]; poolMean[nPool * 3 + 1] = srcMean[ci * 3 + 1]; poolMean[nPool * 3 + 2] = srcMean[ci * 3 + 2]; nPool++; }
      } else {
        for (let i = 0; i < lineY.length; i++) {
          const cy = lineY[i], cx = lineX[i];
          const p = gatherPatch(filled, cy, cx, sh, sw);
          poolPatches.set(p, nPool * PP3);
          const mm = patchMean(p); poolMean[nPool * 3] = mm[0]; poolMean[nPool * 3 + 1] = mm[1]; poolMean[nPool * 3 + 2] = mm[2];
          poolCY[nPool] = cy; poolCX[nPool] = cx; nPool++;
        }
      }
    } else {
      for (let r = 0; r < K; r++) { const ci = (rng() * Nc) | 0; poolCY[nPool] = candY[ci]; poolCX[nPool] = candX[ci]; poolPatches.set(srcPatches.subarray(ci * PP3, ci * PP3 + PP3), nPool * PP3); poolMean[nPool * 3] = srcMean[ci * 3]; poolMean[nPool * 3 + 1] = srcMean[ci * 3 + 1]; poolMean[nPool * 3 + 2] = srcMean[ci * 3 + 2]; nPool++; }
    }
    // neighbourhood coherence
    for (let k = 0; k < 4; k++) {
      const ny = ty + dy4[k], nx = tx + dx4[k];
      if (ny < 0 || ny >= sh || nx < 0 || nx >= sw) continue;
      if (!nnfSet[ny * sw + nx]) continue;
      const cy = nnfY[ny * sw + nx], cx = nnfX[ny * sw + nx];
      const p = gatherPatch(filled, cy, cx, sh, sw);
      poolPatches.set(p, nPool * PP3);
      const mm = patchMean(p); poolMean[nPool * 3] = mm[0]; poolMean[nPool * 3 + 1] = mm[1]; poolMean[nPool * 3 + 2] = mm[2];
      poolCY[nPool] = cy; poolCX[nPool] = cx; nPool++;
    }
    // target mean over known positions (for mean-compat penalty)
    let tm0 = 0, tm1 = 0, tm2 = 0;
    for (const j of tkIdx) { const o = j * 3; tm0 += tPatch[o]; tm1 += tPatch[o + 1]; tm2 += tPatch[o + 2]; }
    const ts = tkIdx.length || 1;
    tm0 /= ts; tm1 /= ts; tm2 /= ts;

    let bestI = 0, bestS = Infinity;
    for (let i = 0; i < nPool; i++) {
      let ssd = 0;
      const base = i * PP3;
      for (const j of tkIdx) { const o = base + j * 3; const d0 = poolPatches[o] - tPatch[o]; const d1 = poolPatches[o + 1] - tPatch[o + 1]; const d2 = poolPatches[o + 2] - tPatch[o + 2]; ssd += d0 * d0 + d1 * d1 + d2 * d2; }
      if (dirVec === null) {
        const sm0 = poolMean[i * 3], sm1 = poolMean[i * 3 + 1], sm2 = poolMean[i * 3 + 2];
        const dm0 = sm0 - tm0, dm1 = sm1 - tm1, dm2 = sm2 - tm2;
        ssd += 4.0 * ts * (dm0 * dm0 + dm1 * dm1 + dm2 * dm2);
      }
      if (ssd < bestS) { bestS = ssd; bestI = i; }
    }
    return [poolCY[bestI], poolCX[bestI]];
  }

  function copyPatch(ty, tx, sy, sx) {
    const wy0 = ty - HALF, wx0 = tx - HALF;
    const src = gatherPatch(filled, sy, sx, sh, sw); // P×P×3 float
    // colour self-adaptation anchored to origKnown
    let taWin = [];
    for (let j = 0; j < PP; j++) {
      const wy = wy0 + (j / P | 0), wx = wx0 + (j % P);
      if (wy >= 0 && wy < sh && wx >= 0 && wx < sw && origKnown[wy * sw + wx]) taWin.push(j);
    }
    if (taWin.length < 8) {
      for (const r of [5, 8]) {
        const by0 = Math.max(0, ty - r), by1 = Math.min(sh, ty + r + 1);
        const bx0 = Math.max(0, tx - r), bx1 = Math.min(sw, tx + r + 1);
        const tv = [];
        for (let y = by0; y < by1; y++) for (let x = bx0; x < bx1; x++) if (origKnown[y * sw + x]) tv.push(y * sw + x);
        if (tv.length >= 8) { taWin = tv; break; }
      }
    }
    if (taWin.length >= 8) {
      let tmean0 = 0, tmean1 = 0, tmean2 = 0;
      for (const idx of taWin) { const o = idx * 3; tmean0 += filled[o]; tmean1 += filled[o + 1]; tmean2 += filled[o + 2]; }
      const tl = taWin.length; tmean0 /= tl; tmean1 /= tl; tmean2 /= tl;
      let tvar0 = 0, tvar1 = 0, tvar2 = 0;
      for (const idx of taWin) { const o = idx * 3; const a = filled[o] - tmean0, b = filled[o + 1] - tmean1, c = filled[o + 2] - tmean2; tvar0 += a * a; tvar1 += b * b; tvar2 += c * c; }
      const tstd0 = Math.sqrt(tvar0 / tl) + 1e-3, tstd1 = Math.sqrt(tvar1 / tl) + 1e-3, tstd2 = Math.sqrt(tvar2 / tl) + 1e-3;
      let sm0 = 0, sm1 = 0, sm2 = 0;
      for (let i = 0; i < PP; i++) { const o = i * 3; sm0 += src[o]; sm1 += src[o + 1]; sm2 += src[o + 2]; }
      sm0 /= PP; sm1 /= PP; sm2 /= PP;
      let sv0 = 0, sv1 = 0, sv2 = 0;
      for (let i = 0; i < PP; i++) { const o = i * 3; const a = src[o] - sm0, b = src[o + 1] - sm1, c = src[o + 2] - sm2; sv0 += a * a; sv1 += b * b; sv2 += c * c; }
      const sstd0 = Math.sqrt(sv0 / PP) + 1e-3, sstd1 = Math.sqrt(sv1 / PP) + 1e-3, sstd2 = Math.sqrt(sv2 / PP) + 1e-3;
      for (let i = 0; i < PP3; i += 3) {
        src[i] = (src[i] - sm0) * (tstd0 / sstd0) + tmean0;
        src[i + 1] = (src[i + 1] - sm1) * (tstd1 / sstd1) + tmean1;
        src[i + 2] = (src[i + 2] - sm2) * (tstd2 / sstd2) + tmean2;
      }
    }
    for (let j = 0; j < PP; j++) {
      const wy = wy0 + (j / P | 0), wx = wx0 + (j % P);
      if (wy < 0 || wy >= sh || wx < 0 || wx >= sw) continue;
      if (!hole[wy * sw + wx]) continue;
      const o = j * 3, d = (wy * sw + wx) * 3;
      filled[d] = src[o]; filled[d + 1] = src[o + 1]; filled[d + 2] = src[o + 2];
      known[wy * sw + wx] = 1; hole[wy * sw + wx] = 0;
    }
  }

  // ---- main fill loop (per-pixel priority, quality-first) ----
  while (true) {
    if (shouldCancel && shouldCancel()) break;
    const eroded = erode3x3(hole, sh, sw);
    const boundary = new Uint8Array(sh * sw);
    let anyBoundary = false;
    for (let i = 0; i < sh * sw; i++) { if (hole[i] && !eroded[i]) { boundary[i] = 1; anyBoundary = true; } }
    if (!anyBoundary) break;
    const Cmap = cvb.boxFilterMean(known, sh, sw, P); // fraction known in P×P
    let bestS = -Infinity, bestI = -1;
    for (let i = 0; i < sh * sw; i++) {
      if (!boundary[i]) continue;
      const pr = Cmap[i] * Dmap[i];
      if (pr > bestS) { bestS = pr; bestI = i; }
    }
    if (bestI < 0) break;
    const ty = (bestI / sw) | 0, tx = bestI % sw;
    const [sy, sx] = bestSource(ty, tx);
    copyPatch(ty, tx, sy, sx);
    nnfY[bestI] = sy; nnfX[bestI] = sx; nnfSet[bestI] = 1;
  }

  // residual-hole TELEA cleanup
  let anyHoleLeft = false;
  for (let i = 0; i < sh * sw; i++) if (hole[i]) { anyHoleLeft = true; break; }
  if (anyHoleLeft) {
    const out = cvb.inpaintTelea(filled, hole, sh, sw, 3);
    for (let i = 0; i < sh * sw * 3; i++) filled[i] = out[i];
  }

  // write ROI back and crop
  for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) {
    const si = (y * sw + x) * 3;
    const wi = ((y + y0r) * curW + (x + x0r)) * 3;
    work[wi] = filled[si]; work[wi + 1] = filled[si + 1]; work[wi + 2] = filled[si + 2];
  }
  return cropRGB(work, curH, curW, PADM, OH, OW);
}
