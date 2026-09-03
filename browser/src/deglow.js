// deglow.js — browser port of text_eraser.text_select._deglow_faint_green
//
// Channel-method de-glow for the faint green halo around text. Pure JS + cv-bridge
// (grayscale + ellipse dilate). Returns the cleaned RGB (Float32Array H*W*3) and the
// weak-region mask (255=de-glowed, debug only).
//
// Signature aligns with the spec: deglowFaintGreen(rgb, H, W, textMask,
//   { thr=6, near_r=24, g_lo=85, protect=1, strength=1.0 })

import * as cvb from './cv-bridge.js';
import { percentile } from './linalg.js';

function textLikeThresholds(protect) {
  const p = Math.max(0, Math.min(1, protect));
  return [Math.round(150 + 106 * (1 - p)), Math.round(170 + 85 * (1 - p))];
}

/**
 * @returns {[Float32Array, Uint8Array]} [cleanedRgb, weakMask255]
 */
export function deglowFaintGreen(rgb, H, W, textMask255, opts = {}) {
  const {
    thr = 6,
    near_r = 24,
    g_lo = 85,
    protect = 1.0,
    strength = 1.0,
  } = opts;

  const n = H * W;
  const s = Math.max(0, Math.min(1, strength));
  if (s <= 0) return [rgb, new Uint8Array(n)];

  // channels
  const r = new Float32Array(n), g = new Float32Array(n), b = new Float32Array(n);
  for (let i = 0; i < n; i++) { const o = i * 3; r[i] = rgb[o]; g[i] = rgb[o + 1]; b[i] = rgb[o + 2]; }
  const out = rgb; // modify in place

  const thr_strong = 15, g_strong = 100, min_strong = 50;
  const strongGreen = new Uint8Array(n);
  let strongCount = 0;
  for (let i = 0; i < n; i++) {
    if ((g[i] - Math.max(r[i], b[i]) > thr_strong) && g[i] > g_strong) { strongGreen[i] = 1; strongCount++; }
  }
  // no strong-glow signal → ordinary image, zero change
  if (strongCount < min_strong) return [out, new Uint8Array(n)];

  const greenWeak = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if ((g[i] - Math.max(r[i], b[i]) > thr) && g[i] > g_lo) greenWeak[i] = 1;
  }

  const m01 = new Uint8Array(n);
  for (let i = 0; i < n; i++) m01[i] = textMask255[i] ? 1 : 0;
  const strong = new Uint8Array(n);
  for (let i = 0; i < n; i++) strong[i] = (m01[i] || strongGreen[i]) ? 1 : 0;

  const near = cvb.dilateMask(m01, H, W, near_r * 2 + 1);

  const [tlRb, tlMin] = textLikeThresholds(protect);
  const textLike = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const mn = Math.min(r[i], g[i], b[i]);
    if (r[i] > tlRb || b[i] > tlRb || mn > tlMin) textLike[i] = 1;
  }

  const weak = new Uint8Array(n);
  let anyWeak = false;
  for (let i = 0; i < n; i++) {
    if (greenWeak[i] && !strong[i] && near[i] && !textLike[i]) { weak[i] = 1; anyWeak = true; }
  }
  if (!anyWeak) return [out, new Uint8Array(n)];

  // background colour: non-green, non-bright region median
  const gray = cvb.rgbToGray(rgb, H, W);
  const bgMsk = new Uint8Array(n);
  for (let i = 0; i < n; i++) if (!greenWeak[i] && gray[i] < 160) bgMsk[i] = 1;
  let bgR = 0, bgG = 0, bgB = 0, bgC = 0;
  for (let i = 0; i < n; i++) if (bgMsk[i]) { bgR += r[i]; bgG += g[i]; bgB += b[i]; bgC++; }
  const bg = bgC ? [bgR / bgC, bgG / bgC, bgB / bgC] : [89, 81, 72];

  // glow colour: top 30% (G-B) pixels inside strong region
  const gbList = [];
  for (let i = 0; i < n; i++) if (strong[i]) gbList.push(g[i] - b[i]);
  let glow = [160, 220, 140];
  if (gbList.length) {
    const thr70 = percentile(Float32Array.from(gbList), 0.7);
    let gR = 0, gG = 0, gB = 0, gC = 0;
    for (let i = 0; i < n; i++) if (strong[i] && (g[i] - b[i]) >= thr70) { gR += r[i]; gG += g[i]; gB += b[i]; gC++; }
    if (gC) glow = [gR / gC, gG / gC, gB / gC];
  }

  // alpha from G channel, then recover_bg (lerp toward bg by strength)
  for (let i = 0; i < n; i++) {
    if (!weak[i]) continue;
    let a = (g[i] - bg[1]) / (glow[1] - bg[1] + 1e-6);
    if (a < 0) a = 0; else if (a > 1) a = 1;
    out[i * 3] = rgb[i * 3] + (bg[0] - rgb[i * 3]) * s;
    out[i * 3 + 1] = rgb[i * 3 + 1] + (bg[1] - rgb[i * 3 + 1]) * s;
    out[i * 3 + 2] = rgb[i * 3 + 2] + (bg[2] - rgb[i * 3 + 2]) * s;
  }

  const weak255 = new Uint8Array(n);
  for (let i = 0; i < n; i++) weak255[i] = weak[i] ? 255 : 0;
  // clamp
  for (let i = 0; i < n * 3; i++) { if (out[i] < 0) out[i] = 0; else if (out[i] > 255) out[i] = 255; }
  return [out, weak255];
}

/**
 * Strong-green (glow) zone mask for display: 255 where (g - max(r,b) > 15) && g > 100.
 * Used by the high-level `erase()` to build the "发光蒙版" overlay. Pure JS, no cv.
 * @returns {Uint8Array} H*W, 255 = strong green
 */
export function greenStrongMask(rgb, H, W) {
  const n = H * W;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const s = i * 3;
    const r = rgb[s], g = rgb[s + 1], b = rgb[s + 2];
    if ((g - Math.max(r, b) > 15) && g > 100) out[i] = 255;
  }
  return out;
}
