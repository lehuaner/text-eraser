// Browser-side / Node binding for the shared WASM core (textcore.wasm).
//
// Single integration point the browser uses to call shared algorithms. Loads the
// SAME .wasm the Python backend loads (via wasmtime), so any operator added here
// is automatically available — and identical — on both ends.
//
// IMPORTANT: wasm memory may GROW during alloc / during a kernel call, which
// invalidates the `memory.buffer` reference. We re-fetch `ex.memory.buffer`
// right before every read/write. The browser worker must do the same.
const fs = require("fs");
const path = require("path");

const WASM_PATH = path.join(__dirname, "..", "build", "textcore.wasm");

let _instance = null;
let _ready = null;

function ready() {
  if (_ready) return _ready;
  _ready = WebAssembly.instantiate(fs.readFileSync(WASM_PATH), {}).then(
    ({ instance }) => {
      _instance = instance;
    }
  );
  return _ready;
}

async function distanceTransformEdt(maskU8, h, w) {
  await ready();
  const ex = _instance.exports;
  const n = h * w;
  const pMask = ex.alloc(n);
  const pOut = ex.alloc(n * 4);
  try {
    let m = ex.memory.buffer;
    new Uint8Array(m, pMask, n).set(maskU8);
    ex.distance_transform_edt(pMask, h, w, pOut);
    m = ex.memory.buffer; // re-fetch: kernel may have grown memory
    const out = new Float32Array(n);
    new Uint8Array(out.buffer).set(new Uint8Array(m, pOut, n * 4));
    ex.dealloc(pMask, n);
    ex.dealloc(pOut, n * 4);
    return out; // Float32Array length n
  } catch (e) {
    ex.dealloc(pMask, n);
    ex.dealloc(pOut, n * 4);
    throw e;
  }
}

async function rgbToGray(rgbF32, h, w) {
  await ready();
  const ex = _instance.exports;
  const n = h * w;
  const pIn = ex.alloc(n * 3 * 4);
  const pOut = ex.alloc(n);
  try {
    let m = ex.memory.buffer;
    new Float32Array(m, pIn, n * 3).set(rgbF32);
    ex.rgb_to_gray(pIn, h, w, pOut);
    m = ex.memory.buffer;
    const out = new Uint8Array(n);
    out.set(new Uint8Array(m, pOut, n));
    return out;
  } finally {
    ex.dealloc(pIn, n * 3 * 4);
    ex.dealloc(pOut, n);
  }
}

async function thresholdOtsu(u8, h, w) {
  await ready();
  const ex = _instance.exports;
  const n = h * w;
  const pIn = ex.alloc(n);
  const pOut = ex.alloc(n);
  try {
    let m = ex.memory.buffer;
    new Uint8Array(m, pIn, n).set(u8);
    const thr = ex.threshold_otsu(pIn, pOut, n);
    m = ex.memory.buffer;
    const bin = new Uint8Array(n);
    bin.set(new Uint8Array(m, pOut, n));
    return { thr, bin };
  } finally {
    ex.dealloc(pIn, n);
    ex.dealloc(pOut, n);
  }
}

async function morphology(maskU8, h, w, kernU8, kh, kw, op) {
  await ready();
  const ex = _instance.exports;
  const n = h * w;
  const nk = kh * kw;
  const opCode = op === "erode" ? 0 : 1;
  const pIn = ex.alloc(n);
  const pOut = ex.alloc(n);
  const pK = ex.alloc(nk);
  try {
    let m = ex.memory.buffer;
    new Uint8Array(m, pIn, n).set(maskU8);
    new Uint8Array(m, pK, nk).set(kernU8);
    ex.morphology(pIn, pOut, h, w, pK, kh, kw, opCode);
    m = ex.memory.buffer;
    const out = new Uint8Array(n);
    out.set(new Uint8Array(m, pOut, n));
    return out;
  } finally {
    ex.dealloc(pIn, n);
    ex.dealloc(pOut, n);
    ex.dealloc(pK, nk);
  }
}

async function connectedComponents(maskU8, h, w) {
  await ready();
  const ex = _instance.exports;
  const n = h * w;
  const pIn = ex.alloc(n);
  const pLabels = ex.alloc(n * 4);
  try {
    let m = ex.memory.buffer;
    new Uint8Array(m, pIn, n).set(maskU8);
    const ncomp = ex.connected_components(pIn, pLabels, h, w);
    m = ex.memory.buffer;
    const labels = new Int32Array(n);
    new Uint8Array(labels.buffer).set(new Uint8Array(m, pLabels, n * 4));
    const pStats = ex.alloc(ncomp * 5 * 4);
    try {
      ex.connected_components_stats(pLabels, pStats, h, w, ncomp);
      m = ex.memory.buffer;
      const raw = new Int32Array(ncomp * 5);
      new Uint8Array(raw.buffer).set(new Uint8Array(m, pStats, ncomp * 5 * 4));
      const stats = [];
      for (let i = 0; i < ncomp; i++) {
        const b = i * 5;
        stats.push({ left: raw[b], top: raw[b + 1], width: raw[b + 2], height: raw[b + 3], area: raw[b + 4] });
      }
      return { n: ncomp, labels, stats };
    } finally {
      ex.dealloc(pStats, ncomp * 5 * 4);
      ex.dealloc(pLabels, n * 4);
    }
  } finally {
    ex.dealloc(pIn, n);
  }
}

async function resizeGrayCubic(u8, h, w, h2, w2) {
  await ready();
  const ex = _instance.exports;
  const n = h * w;
  const n2 = h2 * w2;
  const pIn = ex.alloc(n);
  const pOut = ex.alloc(n2);
  try {
    let m = ex.memory.buffer;
    new Uint8Array(m, pIn, n).set(u8);
    ex.resize_gray_cubic(pIn, pOut, h, w, h2, w2);
    m = ex.memory.buffer;
    const out = new Uint8Array(n2);
    out.set(new Uint8Array(m, pOut, n2));
    return out;
  } finally {
    ex.dealloc(pIn, n);
    ex.dealloc(pOut, n2);
  }
}

async function resizeFloatLinear(f32, h, w, h2, w2) {
  await ready();
  const ex = _instance.exports;
  const n = h * w;
  const n2 = h2 * w2;
  const pIn = ex.alloc(n * 4);
  const pOut = ex.alloc(n2 * 4);
  try {
    let m = ex.memory.buffer;
    new Float32Array(m, pIn, n).set(f32);
    ex.resize_float_linear(pIn, pOut, h, w, h2, w2);
    m = ex.memory.buffer;
    const out = new Float32Array(n2);
    new Uint8Array(out.buffer).set(new Uint8Array(m, pOut, n2 * 4));
    return out;
  } finally {
    ex.dealloc(pIn, n * 4);
    ex.dealloc(pOut, n2 * 4);
  }
}

// THE shared PatchMatch fill. rgbF32: Float32Array H*W*3 (0..255, mutated to result).
async function patchmatchInpaint(rgbF32, h, w, maskU8, sampleU8, p, directionDeg, seed) {
  await ready();
  const ex = _instance.exports;
  const n = h * w;
  const pIn = ex.alloc(n * 3 * 4);
  const pOut = ex.alloc(n * 3 * 4);
  const pMask = ex.alloc(n);
  let pSample = 0;
  const hasSample = sampleU8 ? 1 : 0;
  if (sampleU8) pSample = ex.alloc(n);
  try {
    let m = ex.memory.buffer;
    new Float32Array(m, pIn, n * 3).set(rgbF32);
    new Uint8Array(m, pMask, n).set(maskU8);
    if (sampleU8) new Uint8Array(m, pSample, n).set(sampleU8);
    ex.patchmatch_inpaint(pIn, h, w, pMask, pSample, hasSample,
      p || 7, directionDeg == null ? -1.0 : directionDeg, (seed | 0) >>> 0, pOut);
    m = ex.memory.buffer;
    const out = new Float32Array(n * 3);
    new Uint8Array(out.buffer).set(new Uint8Array(m, pOut, n * 3 * 4));
    return out;
  } finally {
    ex.dealloc(pIn, n * 3 * 4);
    ex.dealloc(pOut, n * 3 * 4);
    ex.dealloc(pMask, n);
    if (pSample) ex.dealloc(pSample, n);
  }
}

// THE shared text-mask synthesis. textMaskU8: Uint8Array H*W (255=text).
// edge: >0 dilate / <0 erode / 0 identity (ellipse diameter = abs(edge)*2+1).
// limitU8: optional Uint8Array H*W (255=allowed fill). Returns [fillU8, sampleU8] 0/255.
async function synthesizeMasks(textMaskU8, h, w, edge, limitU8) {
  await ready();
  const ex = _instance.exports;
  const n = h * w;
  const pIn = ex.alloc(n);
  const pFill = ex.alloc(n);
  const pSmpl = ex.alloc(n);
  let pLim = 0;
  const hasLimit = limitU8 ? 1 : 0;
  if (limitU8) pLim = ex.alloc(n);
  try {
    let m = ex.memory.buffer;
    new Uint8Array(m, pIn, n).set(textMaskU8);
    if (limitU8) new Uint8Array(m, pLim, n).set(limitU8);
    ex.synthesize_masks(pIn, h, w, edge == null ? 1 : edge, pLim, hasLimit, pFill, pSmpl);
    m = ex.memory.buffer;
    const fill = new Uint8Array(n);
    const sample = new Uint8Array(n);
    fill.set(new Uint8Array(m, pFill, n));
    sample.set(new Uint8Array(m, pSmpl, n));
    return [fill, sample];
  } finally {
    ex.dealloc(pIn, n);
    ex.dealloc(pFill, n);
    ex.dealloc(pSmpl, n);
    if (pLim) ex.dealloc(pLim, n);
  }
}

// THE shared de-glow (full green v2). rgbF32: Float32Array H*W*3 (0..255).
// tmaskU8: Uint8Array H*W (255=text). Returns [cleanU8 H*W*3, coreU8 H*W] (0..255/0|255).
async function deglowFullGreenV2(rgbF32, h, w, tmaskU8, strength, zoneRatio, zoneExpand, protectPx, chromaKeep) {
  await ready();
  const ex = _instance.exports;
  const n = h * w;
  const pIn = ex.alloc(n * 3 * 4);
  const pTm = ex.alloc(n);
  const pClean = ex.alloc(n * 3);
  const pCore = ex.alloc(n);
  const pZone = ex.alloc(n);
  try {
    let m = ex.memory.buffer;
    new Float32Array(m, pIn, n * 3).set(rgbF32);
    new Uint8Array(m, pTm, n).set(tmaskU8);
    ex.deglow_full_green_v2(pIn, h, w, pTm,
      strength == null ? 1.0 : strength,
      zoneRatio == null ? 0.6 : zoneRatio,
      zoneExpand == null ? 0 : zoneExpand,
      protectPx == null ? 0 : protectPx,
      chromaKeep == null ? 0 : chromaKeep,
      pClean, pCore, pZone);
    m = ex.memory.buffer;
    const clean = new Uint8Array(n * 3);
    const core = new Uint8Array(n);
    const zone = new Uint8Array(n);
    clean.set(new Uint8Array(m, pClean, n * 3));
    core.set(new Uint8Array(m, pCore, n));
    zone.set(new Uint8Array(m, pZone, n));
    return [clean, core, zone];
  } finally {
    ex.dealloc(pIn, n * 3 * 4);
    ex.dealloc(pTm, n);
    ex.dealloc(pClean, n * 3);
    ex.dealloc(pCore, n);
    ex.dealloc(pZone, n);
  }
}

// THE single shared pipeline entry. Runs the FULL de-glow + mask-surgery +
// PatchMatch fill (browser + backend call this identically).
// rgbF32: Float32Array H*W*3 (0..255). tmaskU8: Uint8Array H*W (255=text).
// tmask2U8: optional Uint8Array H*W second detect to union in.
// Returns [resultU8 H*W*3, fillU8 H*W, cleanU8 H*W*3, zoneU8 H*W].
async function eraseTextGlyphs(rgbF32, h, w, tmaskU8, tmask2U8,
  strength, zoneRatio, zoneExpand, protectPx, chromaKeep, edge, directionDeg, seed,
  edgeAware, softExpand) {
  await ready();
  const ex = _instance.exports;
  const n = h * w;
  const pIn = ex.alloc(n * 3 * 4);
  const pTm = ex.alloc(n);
  const pTm2 = ex.alloc(n);
  const pResult = ex.alloc(n * 3);
  const pFill = ex.alloc(n);
  const pClean = ex.alloc(n * 3);
  const pZone = ex.alloc(n);
  try {
    let m = ex.memory.buffer;
    new Float32Array(m, pIn, n * 3).set(rgbF32);
    new Uint8Array(m, pTm, n).set(tmaskU8);
    if (tmask2U8) new Uint8Array(m, pTm2, n).set(tmask2U8);
    ex.erase_text_glyphs(pIn, h, w, pTm, pTm2,
      strength == null ? 1.0 : strength,
      zoneRatio == null ? 0.6 : zoneRatio,
      zoneExpand == null ? 0 : zoneExpand,
      protectPx == null ? 0 : protectPx,
      chromaKeep == null ? 0 : chromaKeep,
      edge == null ? 0 : edge,
      directionDeg == null ? -1.0 : directionDeg,
      (seed == null ? 0 : seed) >>> 0,
      edgeAware == null ? 0 : edgeAware,
      softExpand == null ? 0.0 : softExpand,
      pResult, pFill, pClean, pZone);
    m = ex.memory.buffer;
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
    ex.dealloc(pIn, n * 3 * 4);
    ex.dealloc(pTm, n);
    ex.dealloc(pTm2, n);
    ex.dealloc(pResult, n * 3);
    ex.dealloc(pFill, n);
    ex.dealloc(pClean, n * 3);
    ex.dealloc(pZone, n);
  }
}

module.exports = {
  ready,
  distanceTransformEdt,
  rgbToGray,
  thresholdOtsu,
  morphology,
  connectedComponents,
  resizeGrayCubic,
  resizeFloatLinear,
  patchmatchInpaint,
  synthesizeMasks,
  deglowFullGreenV2,
  eraseTextGlyphs,
  WASM_PATH,
};
