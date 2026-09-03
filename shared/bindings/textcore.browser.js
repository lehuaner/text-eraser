// Browser (Web Worker) binding for the shared WASM core (textcore.wasm).
//
// Loads textcore.wasm via fetch + WebAssembly.instantiate (no fs/Node deps), so it
// can be bundled by esbuild into the browser pipeline and run in a Web Worker.
// Loads the SAME .wasm the Python backend loads via wasmtime — single source of truth.
//
// IMPORTANT: wasm memory may GROW during alloc / during a kernel call, invalidating
// the `memory.buffer` reference. Re-fetch `ex.memory.buffer` before every read/write.

// Served by the webapp: the StaticFiles mount `/shared` -> <repo>/shared,
// so the built artifact at <repo>/shared/build/textcore.wasm is reachable here.
const WASM_DEFAULT_URL = "/shared/build/textcore.wasm";

let _instance = null;
let _ready = null;

/** Load + instantiate the wasm. Idempotent. `url` overrides the default fetch path. */
export async function ensure(url) {
  if (_ready) return _ready;
  _ready = (async () => {
    const wasmUrl = url || WASM_DEFAULT_URL;
    const resp = await fetch(wasmUrl);
    if (!resp.ok) throw new Error("textcore.wasm fetch failed: " + resp.status + " " + wasmUrl);
    const bytes = await resp.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes, {});
    _instance = instance;
  })();
  return _ready;
}

export function isReady() {
  return _instance != null;
}

function ex() {
  if (!_instance) throw new Error("textcore not ready (call ensure() first)");
  return _instance.exports;
}

export function distanceTransformEdt(maskU8, h, w) {
  const e = ex();
  const n = h * w;
  const pMask = e.alloc(n);
  const pOut = e.alloc(n * 4);
  try {
    let m = e.memory.buffer;
    new Uint8Array(m, pMask, n).set(maskU8);
    e.distance_transform_edt(pMask, h, w, pOut);
    m = e.memory.buffer;
    const out = new Float32Array(n);
    new Uint8Array(out.buffer).set(new Uint8Array(m, pOut, n * 4));
    return out;
  } finally {
    e.dealloc(pMask, n);
    e.dealloc(pOut, n * 4);
  }
}

export function rgbToGray(rgbF32, h, w) {
  const e = ex();
  const n = h * w;
  const pIn = e.alloc(n * 3 * 4);
  const pOut = e.alloc(n);
  try {
    let m = e.memory.buffer;
    new Float32Array(m, pIn, n * 3).set(rgbF32);
    e.rgb_to_gray(pIn, h, w, pOut);
    m = e.memory.buffer;
    const out = new Uint8Array(n);
    out.set(new Uint8Array(m, pOut, n));
    return out;
  } finally {
    e.dealloc(pIn, n * 3 * 4);
    e.dealloc(pOut, n);
  }
}

export function thresholdOtsu(u8, h, w) {
  const e = ex();
  const n = h * w;
  const pIn = e.alloc(n);
  const pOut = e.alloc(n);
  try {
    let m = e.memory.buffer;
    new Uint8Array(m, pIn, n).set(u8);
    const thr = e.threshold_otsu(pIn, pOut, n);
    m = e.memory.buffer;
    const bin = new Uint8Array(n);
    bin.set(new Uint8Array(m, pOut, n));
    return { thr, bin };
  } finally {
    e.dealloc(pIn, n);
    e.dealloc(pOut, n);
  }
}

export function morphology(maskU8, h, w, kernU8, kh, kw, op) {
  const e = ex();
  const n = h * w;
  const nk = kh * kw;
  const opCode = op === "erode" ? 0 : 1;
  const pIn = e.alloc(n);
  const pOut = e.alloc(n);
  const pK = e.alloc(nk);
  try {
    let m = e.memory.buffer;
    new Uint8Array(m, pIn, n).set(maskU8);
    new Uint8Array(m, pK, nk).set(kernU8);
    e.morphology(pIn, pOut, h, w, pK, kh, kw, opCode);
    m = e.memory.buffer;
    const out = new Uint8Array(n);
    out.set(new Uint8Array(m, pOut, n));
    return out;
  } finally {
    e.dealloc(pIn, n);
    e.dealloc(pOut, n);
    e.dealloc(pK, nk);
  }
}

export function connectedComponents(maskU8, h, w) {
  const e = ex();
  const n = h * w;
  const pIn = e.alloc(n);
  const pLabels = e.alloc(n * 4);
  try {
    let m = e.memory.buffer;
    new Uint8Array(m, pIn, n).set(maskU8);
    const ncomp = e.connected_components(pIn, pLabels, h, w);
    m = e.memory.buffer;
    const labels = new Int32Array(n);
    new Uint8Array(labels.buffer).set(new Uint8Array(m, pLabels, n * 4));
    const pStats = e.alloc(ncomp * 5 * 4);
    try {
      e.connected_components_stats(pLabels, pStats, h, w, ncomp);
      m = e.memory.buffer;
      const raw = new Int32Array(ncomp * 5);
      new Uint8Array(raw.buffer).set(new Uint8Array(m, pStats, ncomp * 5 * 4));
      const stats = [];
      for (let i = 0; i < ncomp; i++) {
        const b = i * 5;
        stats.push({ left: raw[b], top: raw[b + 1], width: raw[b + 2], height: raw[b + 3], area: raw[b + 4] });
      }
      return { n: ncomp, labels, stats };
    } finally {
      e.dealloc(pStats, ncomp * 5 * 4);
      e.dealloc(pLabels, n * 4);
    }
  } finally {
    e.dealloc(pIn, n);
  }
}

export function resizeGrayCubic(u8, h, w, h2, w2) {
  const e = ex();
  const n = h * w;
  const n2 = h2 * w2;
  const pIn = e.alloc(n);
  const pOut = e.alloc(n2);
  try {
    let m = e.memory.buffer;
    new Uint8Array(m, pIn, n).set(u8);
    e.resize_gray_cubic(pIn, pOut, h, w, h2, w2);
    m = e.memory.buffer;
    const out = new Uint8Array(n2);
    out.set(new Uint8Array(m, pOut, n2));
    return out;
  } finally {
    e.dealloc(pIn, n);
    e.dealloc(pOut, n2);
  }
}

export function resizeFloatLinear(f32, h, w, h2, w2) {
  const e = ex();
  const n = h * w;
  const n2 = h2 * w2;
  const pIn = e.alloc(n * 4);
  const pOut = e.alloc(n2 * 4);
  try {
    let m = e.memory.buffer;
    new Float32Array(m, pIn, n).set(f32);
    e.resize_float_linear(pIn, pOut, h, w, h2, w2);
    m = e.memory.buffer;
    const out = new Float32Array(n2);
    new Uint8Array(out.buffer).set(new Uint8Array(m, pOut, n2 * 4));
    return out;
  } finally {
    e.dealloc(pIn, n * 4);
    e.dealloc(pOut, n2 * 4);
  }
}

// THE shared text-mask synthesis. textMaskU8: Uint8Array H*W (255=text).
// edge: >0 dilate / <0 erode / 0 identity (ellipse diameter = abs(edge)*2+1).
// limitU8: optional Uint8Array H*W (255=allowed fill). Returns [fillU8, sampleU8] 0/255.
export function synthesizeMasks(textMaskU8, h, w, edge, limitU8) {
  const e = ex();
  const n = h * w;
  const pIn = e.alloc(n);
  const pFill = e.alloc(n);
  const pSmpl = e.alloc(n);
  let pLim = 0;
  const hasLimit = limitU8 ? 1 : 0;
  if (limitU8) pLim = e.alloc(n);
  try {
    let m = e.memory.buffer;
    new Uint8Array(m, pIn, n).set(textMaskU8);
    if (limitU8) new Uint8Array(m, pLim, n).set(limitU8);
    e.synthesize_masks(pIn, h, w, edge == null ? 1 : edge, pLim, hasLimit, pFill, pSmpl);
    m = e.memory.buffer;
    const fill = new Uint8Array(n);
    const sample = new Uint8Array(n);
    fill.set(new Uint8Array(m, pFill, n));
    sample.set(new Uint8Array(m, pSmpl, n));
    return [fill, sample];
  } finally {
    e.dealloc(pIn, n);
    e.dealloc(pFill, n);
    e.dealloc(pSmpl, n);
    if (pLim) e.dealloc(pLim, n);
  }
}

// THE shared PatchMatch fill. rgbF32: Float32Array H*W*3 (0..255, mutated to result).
export function patchmatchInpaint(rgbF32, h, w, maskU8, sampleU8, p, directionDeg, seed) {
  const e = ex();
  const n = h * w;
  const pIn = e.alloc(n * 3 * 4);
  const pOut = e.alloc(n * 3 * 4);
  const pMask = e.alloc(n);
  let pSample = 0;
  const hasSample = sampleU8 ? 1 : 0;
  if (sampleU8) pSample = e.alloc(n);
  try {
    let m = e.memory.buffer;
    new Float32Array(m, pIn, n * 3).set(rgbF32);
    new Uint8Array(m, pMask, n).set(maskU8);
    if (sampleU8) new Uint8Array(m, pSample, n).set(sampleU8);
    e.patchmatch_inpaint(pIn, h, w, pMask, pSample, hasSample,
      p || 7, directionDeg == null ? -1.0 : directionDeg, (seed | 0) >>> 0, pOut);
    m = e.memory.buffer;
    const out = new Float32Array(n * 3);
    new Uint8Array(out.buffer).set(new Uint8Array(m, pOut, n * 3 * 4));
    return out;
  } finally {
    e.dealloc(pIn, n * 3 * 4);
    e.dealloc(pOut, n * 3 * 4);
    e.dealloc(pMask, n);
    if (pSample) e.dealloc(pSample, n);
  }
}

// THE shared de-glow (full green v2). rgbF32: Float32Array H*W*3 (0..255).
// tmaskU8: Uint8Array H*W (255=text). Returns [cleanU8 H*W*3, coreU8 H*W, zoneU8 H*W].
export function deglowFullGreenV2(rgbF32, h, w, tmaskU8, strength, zoneRatio, zoneExpand, protectPx, chromaKeep) {
  const e = ex();
  const n = h * w;
  const pIn = e.alloc(n * 3 * 4);
  const pTm = e.alloc(n);
  const pClean = e.alloc(n * 3);
  const pCore = e.alloc(n);
  const pZone = e.alloc(n);
  try {
    let m = e.memory.buffer;
    new Float32Array(m, pIn, n * 3).set(rgbF32);
    new Uint8Array(m, pTm, n).set(tmaskU8);
    e.deglow_full_green_v2(pIn, h, w, pTm,
      strength == null ? 1.0 : strength,
      zoneRatio == null ? 0.6 : zoneRatio,
      zoneExpand == null ? 0 : zoneExpand,
      protectPx == null ? 0 : protectPx,
      chromaKeep == null ? 0 : chromaKeep,
      pClean, pCore, pZone);
    m = e.memory.buffer;
    const clean = new Uint8Array(n * 3);
    const core = new Uint8Array(n);
    const zone = new Uint8Array(n);
    clean.set(new Uint8Array(m, pClean, n * 3));
    core.set(new Uint8Array(m, pCore, n));
    zone.set(new Uint8Array(m, pZone, n));
    return [clean, core, zone];
  } finally {
    e.dealloc(pIn, n * 3 * 4);
    e.dealloc(pTm, n);
    e.dealloc(pClean, n * 3);
    e.dealloc(pCore, n);
    e.dealloc(pZone, n);
  }
}

// THE single shared pipeline entry. Runs the FULL de-glow + mask-surgery +
// PatchMatch fill (browser + backend call this identically).
// rgbF32: Float32Array H*W*3 (0..255). tmaskU8: Uint8Array H*W (255=text).
// tmask2U8: optional Uint8Array H*W second detect to union in.
// Returns [resultU8 H*W*3, fillU8 H*W, cleanU8 H*W*3, zoneU8 H*W].
export function eraseTextGlyphs(rgbF32, h, w, tmaskU8, tmask2U8,
  strength, zoneRatio, zoneExpand, protectPx, chromaKeep, edge, directionDeg, seed) {
  const e = ex();
  const n = h * w;
  const pIn = e.alloc(n * 3 * 4);
  const pTm = e.alloc(n);
  const pTm2 = e.alloc(n);
  const pResult = e.alloc(n * 3);
  const pFill = e.alloc(n);
  const pClean = e.alloc(n * 3);
  const pZone = e.alloc(n);
  try {
    let m = e.memory.buffer;
    new Float32Array(m, pIn, n * 3).set(rgbF32);
    new Uint8Array(m, pTm, n).set(tmaskU8);
    if (tmask2U8) new Uint8Array(m, pTm2, n).set(tmask2U8);
    e.erase_text_glyphs(pIn, h, w, pTm, pTm2,
      strength == null ? 1.0 : strength,
      zoneRatio == null ? 0.6 : zoneRatio,
      zoneExpand == null ? 0 : zoneExpand,
      protectPx == null ? 0 : protectPx,
      chromaKeep == null ? 0 : chromaKeep,
      edge == null ? 0 : edge,
      directionDeg == null ? -1.0 : directionDeg,
      (seed == null ? 0 : seed) >>> 0,
      pResult, pFill, pClean, pZone);
    m = e.memory.buffer;
    const result = new Uint8Array(n * 3);
    const fill = new Uint8Array(n);
    const clean = new Uint8Array(n * 3);
    const zone = new Uint8Array(n);
    result.set(new Uint8Array(m, pResult, n * 3));
    fill.set(new Uint8Array(m, pFill, n));
    clean.set(new Uint8Array(m, pClean, n * 3));
    zone.set(new Uint8Array(m, pZone, n));
    return [result, fill, clean, zone];
  } finally {
    e.dealloc(pIn, n * 3 * 4);
    e.dealloc(pTm, n);
    e.dealloc(pTm2, n);
    e.dealloc(pResult, n * 3);
    e.dealloc(pFill, n);
    e.dealloc(pClean, n * 3);
    e.dealloc(pZone, n);
  }
}
