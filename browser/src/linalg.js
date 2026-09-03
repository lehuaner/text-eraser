// linalg.js — minimal self-written linear-algebra / stats helpers (numpy replacement).
//
// Per the dependency map we replace numpy with JS TypedArrays + a small set of
// routines. No Pyodide is used (chosen for size + startup speed in the browser).
// Everything here is pure JS and runs in a Web Worker or the main thread.

/** Clamp x to [lo, hi]. */
export function clamp(x, lo, hi) {
  return x < lo ? lo : x > hi ? hi : x;
}

/**
 * Iterative quickselect-based percentile over a Float32Array.
 * q in [0,1]; returns the q-quantile WITHOUT modifying the input (copies first).
 * O(n) average; used instead of a full sort so large (whole-image) arrays stay cheap.
 */
export function percentile(arr, q) {
  const n = arr.length;
  if (n === 0) return 0;
  if (n === 1) return arr[0];
  const a = arr.slice(); // copy
  const k = Math.max(0, Math.min(n - 1, Math.round(q * (n - 1))));
  return quickselect(a, k);
}

/** Median of a Float32Array (q = 0.5). */
export function median(arr) {
  return percentile(arr, 0.5);
}

function quickselect(a, k) {
  let lo = 0, hi = a.length - 1;
  while (lo < hi) {
    const pivot = a[(lo + hi) >> 1];
    let i = lo, j = hi;
    while (i <= j) {
      while (a[i] < pivot) i++;
      while (a[j] > pivot) j--;
      if (i <= j) {
        const t = a[i]; a[i] = a[j]; a[j] = t;
        i++; j--;
      }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else return a[k];
  }
  return a[k];
}

/**
 * Histogram median for integer-valued arrays in [min, max].
 * Faster than quickselect when values are quantized (e.g. 0..255 grayscale).
 */
export function histMedian(values, min = 0, max = 255) {
  const n = values.length;
  if (n === 0) return 0;
  const bins = max - min + 1;
  const hist = new Int32Array(bins);
  for (let i = 0; i < n; i++) {
    let v = values[i] | 0;
    if (v < min) v = min; else if (v > max) v = max;
    hist[v - min]++;
  }
  const half = n >> 1;
  let acc = 0;
  for (let b = 0; b < bins; b++) {
    acc += hist[b];
    if (acc > half) return b + min;
  }
  return min;
}

/** Mean of a Float32Array. */
export function mean(arr) {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return arr.length ? s / arr.length : 0;
}

/** Standard deviation (population, +1e-3 guard handled by caller). */
export function std(arr) {
  const m = mean(arr);
  let s = 0;
  for (let i = 0; i < arr.length; i++) { const d = arr[i] - m; s += d * d; }
  return arr.length ? Math.sqrt(s / arr.length) : 0;
}

/** argmax over a flat Float32Array, returns index. */
export function argmax(arr) {
  let bi = 0, bv = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > bv) { bv = arr[i]; bi = i; }
  }
  return bi;
}

/**
 * Convert a browser ImageData (RGBA, 0..255) to a flat RGB Float32Array (H*W*3).
 * `out` optional preallocated Float32Array(H*W*3).
 */
export function imageDataToRgb(data, H, W, out) {
  const n = H * W;
  const rgb = out && out.length === n * 3 ? out : new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const s = i * 4, d = i * 3;
    rgb[d] = data[s];
    rgb[d + 1] = data[s + 1];
    rgb[d + 2] = data[s + 2];
  }
  return rgb;
}

/**
 * Convert a flat RGB Float32Array (H*W*3, any range) to ImageData (RGBA, 0..255).
 * Values are clamped to [0,255] and rounded.
 */
export function rgbToImageData(rgb, H, W) {
  const n = H * W;
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    const s = i * 3, d = i * 4;
    out[d] = rgb[s];
    out[d + 1] = rgb[s + 1];
    out[d + 2] = rgb[s + 2];
    out[d + 3] = 255;
  }
  return new ImageData(out, W, H);
}

/** Build a single-channel Uint8Array mask (0/255) from any 0..255-ish input. */
export function toMask255(src, n) {
  const m = new Uint8Array(n);
  for (let i = 0; i < n; i++) m[i] = src[i] > 0 ? 255 : 0;
  return m;
}

/** XOR-ish equality count helper (debug). */
export function countDiff(a, b) {
  let c = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) c++;
  return c;
}

/** Maximum absolute difference between two equal-length arrays (acceptance metric). */
export function maxAbsDiff(a, b) {
  let m = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

/**
 * Deterministic seeded PRNG (mulberry32). The upstream Python uses
 * numpy's PCG64 (seed 0); we cannot reproduce PCG64 bit-for-bit in JS, and
 * PatchMatch results are stable to candidate ordering for texture regions, so a
 * deterministic seed is sufficient and keeps browser runs reproducible.
 * Returns a function () => float in [0,1).
 */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Binary erosion with a square (rect) kernel of diameter k, mask is 0/1. */
export function erodeRect(mask, H, W, k) {
  const r = (k / 2) | 0;
  const out = new Uint8Array(H * W);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let ok = 1;
      for (let dy = -r; dy <= r && ok; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= H) { ok = 0; break; }
        for (let dx = -r; dx <= r; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= W || !mask[yy * W + xx]) { ok = 0; break; }
        }
      }
      out[y * W + x] = ok;
    }
  }
  return out;
}

/** Binary dilation with a square (rect) kernel of diameter k, mask is 0/1. */
export function dilateRect(mask, H, W, k) {
  const r = (k / 2) | 0;
  const out = new Uint8Array(H * W);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (!mask[y * W + x]) continue;
      for (let dy = -r; dy <= r; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= H) continue;
        for (let dx = -r; dx <= r; dx++) {
          const xx = x + dx;
          if (xx >= 0 && xx < W) out[yy * W + xx] = 1;
        }
      }
    }
  }
  return out;
}

/** 3x3 binary erosion (hot-loop connectivity test). */
export function erode3x3(mask, H, W) {
  const out = new Uint8Array(H * W);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let ok = 1;
      for (let dy = -1; dy <= 1 && ok; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= H) { ok = 0; break; }
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= W || !mask[yy * W + xx]) { ok = 0; break; }
        }
      }
      out[y * W + x] = ok;
    }
  }
  return out;
}

/** Morphological dilation of a FLOAT image (max over 3x3) — used for Dmap. */
export function dilateMax3x3(src, H, W) {
  const out = new Float32Array(H * W);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let m = src[y * W + x];
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= H) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= W) continue;
          const v = src[yy * W + xx];
          if (v > m) m = v;
        }
      }
      out[y * W + x] = m;
    }
  }
  return out;
}
