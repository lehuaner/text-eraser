// index.js — public ES Module entry for the pure-browser text-eraser port.
//
//   import { inpaint, eraseTextGlyphs } from 'text-eraser-browser';
//
// Runtime target: browser (no Python interpreter, no Node service).
// Dependencies are loaded lazily at runtime:
//   * opencv.js  — via ensureOpenCV() (loads the wasm build, then cv-bridge uses it)
//   * onnxruntime-web — only needed for detectTextMask() (DBNet), optional
//
// Data contract (matches core/textpatch_fill.py):
//   imageData : browser ImageData (RGBA)
//   mask / sampleMask / textMask : same-size single-channel Uint8Array (255 = target)
//   returns   : same-size ImageData
//
// Cancellation: `shouldCancel` may be a zero-arg function returning bool OR an
// AbortSignal (we read `.aborted`). The Worker wrapper (eraser.worker.js) additionally
// accepts a `{ signal }`-style or postMessage('cancel') flow.

import * as cvb from './cv-bridge.js';
import { patchmatchInpaint } from './patchmatch.js';
import { deglowFaintGreen, greenStrongMask } from './deglow.js';
import { imageDataToRgb, rgbToImageData } from './linalg.js';
import { loadDBNet as _loadDBNet, detectTextMask as _detectTextMask, maskToBoxes, DEFAULT_MODELS } from './detect-dbnet.js';
// Shared WASM algorithm core (textcore.wasm) — the SAME bytes the Python backend
// loads. When ready, the pipeline below dispatches through it so the browser and
// backend run the identical glow pipeline (de-glow + mask surgery + PatchMatch).
import * as SharedCore from '../../shared/bindings/textcore.browser.js';

export { _loadDBNet as loadDBNet, _detectTextMask as detectTextMask, DEFAULT_MODELS };
export { setCv, getCv, isCvReady, ensureSharedCore, usingSharedCore } from './cv-bridge.js';

// ---------------------------------------------------------------------------
// OpenCV.js loader
// ---------------------------------------------------------------------------
let _cvPromise = null;

// Vendored locally so the browser engine works fully offline (no CDN dependency).
const DEFAULT_OPENCV_URL = '/browser/vendor/opencv.js';

/**
 * Load opencv.js (wasm) into the current global scope and wire it into cv-bridge.
 * Safe to call multiple times — returns the same promise. Works on the main thread
 * (injects a <script>) and inside a module Web Worker (dynamic import(); opencv.js
 * assigns self.cv after the wasm runtime initializes).
 * @param {object} opts { opencvUrl?, opencvWasmPath? }
 */
export function ensureOpenCV(opts = {}) {
  if (_cvPromise) return _cvPromise;
  // already loaded (e.g. set via setCv in tests / pre-warmed) — return as-is
  if (cvb.isCvReady()) return Promise.resolve();
  const url = opts.opencvUrl || DEFAULT_OPENCV_URL;
  _cvPromise = (async () => {
    const inWorker = (typeof self !== 'undefined' && (typeof document === 'undefined' || !document.createElement));
    // Classic Web Worker is the only runtime where opencv.js's UMD wrapper reliably
    // attaches `cv`: its `typeof importScripts === 'function'` branch assigns
    // `root.cv = factory()` (root === worker global). A module worker can't load the
    // UMD build at all (top-level `this` is undefined → root is undefined → crash).
    const inClassicWorker = inWorker && typeof importScripts === 'function';
    let cv;
    if (inClassicWorker) {
      if (!self.cv) importScripts(url);
      cv = self.cv;
    } else if (inWorker) {
      await import(/* @vite-ignore */ url);
      cv = self.cv || globalThis.cv;
    } else if (typeof document !== 'undefined' && document.createElement) {
      cv = await new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = url;
        s.onerror = () => reject(new Error('ensureOpenCV: failed to load opencv.js from ' + url));
        s.onload = () => resolve(self.cv || globalThis.cv);
        document.head.appendChild(s);
      });
    } else {
      throw new Error('ensureOpenCV: unsupported runtime (no document and no worker self)');
    }
    if (!cv) throw new Error('ensureOpenCV: opencv.js loaded but global `cv` not found');
    if (opts.opencvWasmPath && typeof cv.locateFile === 'function') {
      cv.locateFile = (p) => opts.opencvWasmPath + p;
    }
    // opencv.js assigns `cv` immediately, but `cv.Mat` only exists after the wasm
    // runtime initializes (fires cv.onRuntimeInitialized).
    if (typeof cv.Mat === 'function' && typeof cv.getBuildInformation === 'function') {
      cvb.setCv(cv);
      return { cv };
    }
    // Wait for the wasm runtime to finish initializing. This is racy in workers: the
    // inline opencv.js build may fire `onRuntimeInitialized` *before* we attach our
    // handler, leaving the callback never called. Guard with a polling fallback so we
    // always resolve (and a hard cap so we never hang forever).
    await new Promise((resolve) => {
      let done = false;
      const finish = () => { if (done) return; done = true; clearInterval(poll); clearTimeout(cap); resolve(); };
      const prev = cv.onRuntimeInitialized;
      if (typeof prev === 'function') cv.onRuntimeInitialized = () => { try { prev(); } catch (_) {} finish(); };
      else cv.onRuntimeInitialized = finish;
      const poll = setInterval(() => {
        if (typeof cv.Mat === 'function' && typeof cv.getBuildInformation === 'function') finish();
      }, 50);
      const cap = setTimeout(finish, 20000);
    });
    cvb.setCv(cv);
    // opencv's `cv` is itself a thenable (await cv resolves once the wasm runtime is
    // ready). Returning it directly from this async function makes our result promise
    // *adopt* that thenable; opencv may have already consumed it internally, leaving
    // the adoption to never settle and every caller's `await` to hang forever. Return
    // a plain (non-thenable) wrapper instead; callers read the live instance via
    // cvb.getCv() / self.cv.
    return { cv };
  })();
  // reset on failure so a later retry can reload
  _cvPromise.catch(() => { _cvPromise = null; });
  return _cvPromise;
}

/**
 * Preload opencv.js (and optionally a DBNet session). Useful to call once up-front
 * so the first inpaint() doesn't pay the load latency.
 */
export async function init(opts = {}) {
  await ensureOpenCV(opts);
  if (opts.dbnet) {
    const session = await _loadDBNet(opts.dbnet === true ? {} : (opts.dbnet || {}));
    return { cv: cvb.getCv(), dbnetSession: session };
  }
  return { cv: cvb.getCv(), dbnetSession: null };
}

// ---------------------------------------------------------------------------
// Argument helpers
// ---------------------------------------------------------------------------

function maskTo255(m, n) {
  if (m == null) return null;
  if (m instanceof Uint8Array || m instanceof Uint8ClampedArray) {
    const a = new Uint8Array(n);
    for (let i = 0; i < n; i++) a[i] = m[i] ? 255 : 0;
    return a;
  }
  if (m.data) return maskTo255(m.data, n); // e.g. {data:Uint8Array}
  throw new Error('mask must be a Uint8Array of length H*W with 255=target');
}

function normalizeCancel(shouldCancel) {
  if (!shouldCancel) return null;
  if (typeof shouldCancel === 'function') return shouldCancel;
  if (typeof shouldCancel === 'object' && 'aborted' in shouldCancel) {
    return () => shouldCancel.aborted === true;
  }
  return null;
}

function imageDataLike(img) {
  if (img && img.width && img.height && img.data) return img;
  throw new Error('imageData must be a browser ImageData (or {width,height,data})');
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Content-aware fill (PatchMatch + TELEA fallback).
 * @param {ImageData} imageData
 * @param {object} opts { mask, sampleMask=null, direction=null,
 *                        flatSpan=40, flatTex=15.0, shouldCancel=null }
 * @returns {Promise<ImageData>}
 */
export async function inpaint(imageData, opts = {}) {
  await ensureOpenCV();
  await cvb.ensureSharedCore();
  const img = imageDataLike(imageData);
  const W = img.width, H = img.height, n = H * W;
  if (!opts.mask) throw new Error('inpaint: `mask` (Uint8Array H*W, 255=target) is required');
  const rgb = imageDataToRgb(img.data, H, W);
  const mask255 = maskTo255(opts.mask, n);
  const sample255 = opts.sampleMask ? maskTo255(opts.sampleMask, n) : null;
  const out = patchmatchInpaint(rgb, H, W, mask255, {
    sampleMask: sample255,
    direction: opts.direction ?? null,
    flatSpan: opts.flatSpan ?? 40,
    flatTex: opts.flatTex ?? 15.0,
    shouldCancel: normalizeCancel(opts.shouldCancel),
  });
  return rgbToImageData(out, H, W);
}

/**
 * Erase text glyphs given a text mask.
 * @param {ImageData} imageData
 * @param {object} opts { textMask, edge=1, deglow=true, deglowStrength=1.0,
 *                        limit=null, direction=null, flatSpan=40, flatTex=15.0,
 *                        shouldCancel=null }
 * @returns {Promise<ImageData>}
 */
export async function eraseTextGlyphs(imageData, opts = {}) {
  await ensureOpenCV();
  await cvb.ensureSharedCore();
  const img = imageDataLike(imageData);
  const W = img.width, H = img.height, n = H * W;
  if (!opts.textMask) throw new Error('eraseTextGlyphs: `textMask` (Uint8Array H*W, 255=text) is required');
  const rgb = imageDataToRgb(img.data, H, W);
  const tm = maskTo255(opts.textMask, n);
  // Optional second-stage re-detect mask (tm_clean) — when supplied, the browser
  // feeds the SAME mask union into the shared wasm as the Python backend does.
  // Without it the wasm unions only the raw detect, which is the backend!=frontend
  // divergence. Callers that re-detect on the de-glowed image should pass it here.
  const tm2 = opts.textMask2 ? maskTo255(opts.textMask2, n) : null;

  // WASM-FIRST: run the FULL shared glow pipeline (de-glow + mask surgery + fill)
  // through textcore.wasm — the identical bytes the Python backend loads, so the
  // browser and backend produce byte-identical results. Falls back to the pure-JS
  // pipeline below when the shared core is unavailable or deglow is disabled.
  if (cvb.usingSharedCore() && typeof SharedCore.eraseTextGlyphs === 'function' && opts.deglow !== false) {
    const rgbF32 = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) rgbF32[i] = rgb[i];
    const [resultU8, _fillU8, _cleanU8, _zoneU8] = SharedCore.eraseTextGlyphs(
      rgbF32, H, W, tm, tm2,
      opts.deglowStrength ?? 1.0,
      opts.deglowZoneRatio ?? 0.6,
      opts.deglowZoneExpand ?? 10,
      opts.deglowProtectPx ?? 1,
      (opts.deglowChromaKeep ?? true) ? 1 : 0,
      opts.edge ?? 1,
      opts.direction ?? -1.0,
      0,
    );
    return u8rgbToImageData(resultU8, H, W);
  }

  // optional channel-method de-glow on the text neighbourhood
  if (opts.deglow !== false) {
    deglowFaintGreen(rgb, H, W, tm, {
      thr: 6, near_r: 24, g_lo: 85, protect: 1, strength: opts.deglowStrength ?? 1.0,
    });
  }

  // fill region = ellipse-dilate(textMask, edge)
  const tm01 = new Uint8Array(n);
  for (let i = 0; i < n; i++) tm01[i] = tm[i] ? 1 : 0;
  let filled01 = tm01;
  if (opts.edge > 0) filled01 = cvb.dilateMask(tm01, H, W, opts.edge * 2 + 1);
  else if (opts.edge < 0) filled01 = cvb.erodeMask(tm01, H, W, -opts.edge * 2 + 1);

  // sample region = whole image - ellipse-dilate(textMask, edge)  (per spec #4)
  const sample01 = new Uint8Array(n);
  for (let i = 0; i < n; i++) sample01[i] = filled01[i] ? 0 : 1;

  // limit restricts fill range but NOT sampling
  if (opts.limit) {
    const lm = maskTo255(opts.limit, n);
    for (let i = 0; i < n; i++) if (!lm[i]) filled01[i] = 0;
  }

  const out = patchmatchInpaint(rgb, H, W, mask255From01(filled01), {
    sampleMask: mask255From01(sample01),
    direction: opts.direction ?? null,
    flatSpan: opts.flatSpan ?? 40,
    flatTex: opts.flatTex ?? 15.0,
    shouldCancel: normalizeCancel(opts.shouldCancel),
  });
  return rgbToImageData(out, H, W);
}

// local helper to convert 0/1 -> 255 mask
function mask255From01(m01) {
  const a = new Uint8Array(m01.length);
  for (let i = 0; i < m01.length; i++) a[i] = m01[i] ? 255 : 0;
  return a;
}

// Convert a flat RGB Uint8Array (H*W*3) to ImageData (clamped).
function u8rgbToImageData(u8, H, W) {
  const n = H * W;
  const f = new Float32Array(n * 3);
  for (let i = 0; i < n * 3; i++) f[i] = u8[i];
  return rgbToImageData(f, H, W);
}

// ---------------------------------------------------------------------------
// High-level `erase()` — full pipeline (detect → deglow → fill → overlays)
// ---------------------------------------------------------------------------

function to01(mask255, n) {
  const a = new Uint8Array(n);
  for (let i = 0; i < n; i++) a[i] = mask255[i] ? 1 : 0;
  return a;
}

function edgeDilate01(m01, H, W, edge) {
  if (edge > 0) return cvb.dilateMask(m01, H, W, edge * 2 + 1);
  if (edge < 0) return cvb.erodeMask(m01, H, W, -edge * 2 + 1);
  return m01;
}

/** Red-blend overlay of a 255 mask over a base RGB (Float32Array). */
function maskOverlayImageData(baseRgb, mask255, H, W) {
  const n = H * W;
  const rgb = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const s = i * 3;
    if (mask255[i]) {
      rgb[s] = baseRgb[s] * 0.35 + 255 * 0.65;
      rgb[s + 1] = baseRgb[s + 1] * 0.35 + 60 * 0.65;
      rgb[s + 2] = baseRgb[s + 2] * 0.35 + 60 * 0.65;
    } else {
      rgb[s] = baseRgb[s]; rgb[s + 1] = baseRgb[s + 1]; rgb[s + 2] = baseRgb[s + 2];
    }
  }
  return rgbToImageData(rgb, H, W);
}

/** RGBA text-layer: keep only mask pixels from base, rest transparent. */
function textLayerImageData(baseRgb, mask255, H, W) {
  const n = H * W;
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    if (mask255[i]) {
      const s = i * 3, d = i * 4;
      out[d] = baseRgb[s]; out[d + 1] = baseRgb[s + 1]; out[d + 2] = baseRgb[s + 2]; out[d + 3] = 255;
    }
  }
  return new ImageData(out, W, H);
}

/** RGBA mask (red, semi-transparent) for the fill region. */
function maskTransparentImageData(mask255, H, W) {
  const n = H * W;
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    if (mask255[i]) { out[i * 4] = 255; out[i * 4 + 1] = 60; out[i * 4 + 2] = 60; out[i * 4 + 3] = 150; }
  }
  return new ImageData(out, W, H);
}

/**
 * Full text-erasing pipeline that mirrors the backend `/api/erase` contract so the
 * web frontend can display results identically whether compute ran on the server
 * or in the browser. Reuses the current browser-port primitives:
 *   detectTextMask (DBNet) OR caller-supplied textMask → deglowFaintGreen →
 *   patchmatchInpaint → overlays.
 *
 * @param {ImageData} imageData
 * @param {object} opts {
 *   textMask?, edge=1, direction?, deglow=true, deglowStrength=1,
 *   deglowGreenThr=6, deglowRange=24, deglowGlo=85, deglowProtect=1,
 *   dbnetSession? (required if textMask omitted), mlMaxSide=960, shouldCancel?
 * }
 * @returns {Promise<object>} raw intermediates (ImageData / masks / scalars) — the
 *   frontend worker/engine turns these into the base64 `data` object.
 */
export async function erase(imageData, opts = {}) {
  await ensureOpenCV();
  await cvb.ensureSharedCore();
  const img = imageDataLike(imageData);
  const W = img.width, H = img.height, n = H * W;
  const rgb = imageDataToRgb(img.data, H, W);

  // 1) text mask
  let textMask255;
  if (opts.textMask) {
    textMask255 = maskTo255(opts.textMask, n);
  } else {
    const session = opts.dbnetSession;
    if (!session) throw new Error('erase: provide opts.textMask or a loaded DBNet session (opts.dbnetSession)');
    textMask255 = await _detectTextMask(rgb, H, W, {
      session,
      strength: opts.strength ?? 1.0,
      maskThreshold: opts.maskThreshold ?? 0.4,
      maskMaxSide: opts.maskMaxSide || 1600,
      // Match the Python backend's v2 pipeline exactly: the RAW text mask is
      // detected with tint_fill=False (no color-tint growth, so the glow halo is
      // NOT pulled into the mask). The re-detect on the de-glowed image below
      // uses tintFill=True. This keeps browser and backend inputs identical.
      tintFill: false,
      fillWhite: opts.fillWhite ?? true,
      fillMaxDist: opts.fillMaxDist ?? 12,
    });
  }
  const textMask01 = to01(textMask255, n);

  // WASM-FIRST: run the FULL shared glow pipeline through textcore.wasm (the identical
  // bytes the Python backend loads) so browser + backend are byte-consistent. Falls
  // back to the pure-JS pipeline below when the core is unavailable or deglow disabled.
  if (cvb.usingSharedCore() && typeof SharedCore.eraseTextGlyphs === 'function' && opts.deglow !== false) {
    const _edge = opts.edge ?? 1;
    const rgbF32 = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) rgbF32[i] = rgb[i];

    // Mirror the Python backend's v2 pipeline inputs EXACTLY:
    //   tmask    = raw detect on the original image (tintFill=false)  -> textMask255
    //   clean0   = de-glow(rgb, tmask)                                -> deglowFullGreenV2
    //   tm_clean = re-detect on clean0 (tintFill=true)                -> second detect
    // Both masks are passed to the shared wasm so the browser and backend feed the
    // identical mask union into PatchMatch (previously the browser passed null here,
    // which is the backend!=frontend divergence).
    const tmask = textMask255;
    const [cleanU8_t0] = SharedCore.deglowFullGreenV2(
      rgbF32, H, W, tmask,
      opts.deglowStrength ?? 1.0,
      opts.deglowZoneRatio ?? 0.6,
      opts.deglowZoneExpand ?? 10,
      opts.deglowProtectPx ?? 1,
      (opts.deglowChromaKeep ?? true) ? 1 : 0,
    );
    let tmClean = tmask; // fallback if no DBNet session to re-detect
    if (opts.dbnetSession) {
      const cleanF32 = new Float32Array(n * 3);
      for (let i = 0; i < n * 3; i++) cleanF32[i] = cleanU8_t0[i];
      tmClean = await _detectTextMask(cleanF32, H, W, {
        session: opts.dbnetSession,
        strength: opts.strength ?? 1.0,
        maskThreshold: opts.maskThreshold ?? 0.4,
        maskMaxSide: opts.maskMaxSide || 1600,
        tintFill: true,
        fillWhite: opts.fillWhite ?? true,
        fillMaxDist: opts.fillMaxDist ?? 12,
      });
    }
    const [resultU8, fillU8, cleanU8, zoneU8] = SharedCore.eraseTextGlyphs(
      rgbF32, H, W, tmask, tmClean,
      opts.deglowStrength ?? 1.0,
      opts.deglowZoneRatio ?? 0.6,
      opts.deglowZoneExpand ?? 10,
      opts.deglowProtectPx ?? 1,
      (opts.deglowChromaKeep ?? true) ? 1 : 0,
      _edge,
      opts.direction ?? -1.0,
      0,
    );
    const cleanFloat = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) cleanFloat[i] = cleanU8[i];
    const result = u8rgbToImageData(resultU8, H, W);
    const overlay = maskOverlayImageData(cleanFloat, fillU8, H, W);
    const overlayPre = maskOverlayImageData(cleanFloat, textMask255, H, W);
    const textLayer = textLayerImageData(cleanFloat, textMask255, H, W);
    const maskTransparent = maskTransparentImageData(fillU8, H, W);
    let boxes = [];
    try { boxes = maskToBoxes(textMask255, H, W); } catch (_e) { /* best-effort */ }
    let maskPix = 0;
    for (let i = 0; i < n; i++) if (textMask255[i]) maskPix++;
    const hasGlow = zoneU8.some((v) => v !== 0);
    const _cfg = {
      glow_mode: "auto",
      deglow_scheme: "v2",
      deglow_strength: opts.deglowStrength ?? 1.0,
      deglow_green_thr: opts.deglowGreenThr ?? 6.0,
      deglow_range: opts.deglowRange ?? 24,
      deglow_glo: opts.deglowGlo ?? 85.0,
      deglow_protect: opts.deglowProtect ?? 1.0,
      deglow_chroma_keep: opts.deglowChromaKeep ?? true,
      fill_white: opts.fillWhite ?? true,
      fill_max_dist: opts.fillMaxDist ?? 12,
      auto_edge: opts.autoEdge ?? false,
      auto_max_edge: opts.autoMaxEdge ?? 2,
    };
    return {
      result,
      overlay,
      maskOverlay: overlay,
      overlayPre,
      textLayer,
      maskTransparent,
      deglow: u8rgbToImageData(cleanU8, H, W),
      glowZone: hasGlow ? maskOverlayImageData(rgb, zoneU8, H, W) : null,
      boxes,
      cfg: _cfg,
      maskPix,
      edgeUsed: _edge,
      autoEdge: false,
      hasGlow,
    };
  }

  // 2) de-glow (browser port's channel method)
  let work = rgb, deglowImg = null, glowZone255 = null, hasGlow = false;
  if (opts.deglow !== false) {
    const [cl, _weak] = deglowFaintGreen(rgb, H, W, textMask255, {
      thr: opts.deglowGreenThr ?? 6, near_r: opts.deglowRange ?? 24,
      g_lo: opts.deglowGlo ?? 85, protect: opts.deglowProtect ?? 1,
      strength: opts.deglowStrength ?? 1.0,
    });
    let changed = false;
    for (let i = 0; i < n * 3; i++) { if (Math.abs(cl[i] - rgb[i]) > 0.5) { changed = true; break; } }
    work = cl;
    if (changed) {
      hasGlow = true;
      deglowImg = cl;
      glowZone255 = greenStrongMask(rgb, H, W);
    }
  }
  const base = hasGlow ? work : rgb; // overlay base matches backend (de-glowed when glow present)

  // 3) edge-dilate → fill region; sample = whole image minus fill
  const edge = opts.edge ?? 1;
  const filled01 = edgeDilate01(textMask01, H, W, edge);
  const filled255 = mask255From01(filled01);
  const sample01 = new Uint8Array(n);
  for (let i = 0; i < n; i++) sample01[i] = filled01[i] ? 0 : 1;

  // 4) patch-fill
  const out = patchmatchInpaint(work, H, W, filled255, {
    sampleMask: mask255From01(sample01),
    direction: opts.direction ?? null,
    flatSpan: opts.flatSpan ?? 40,
    flatTex: opts.flatTex ?? 15.0,
    shouldCancel: normalizeCancel(opts.shouldCancel),
  });

  // 5) build display intermediates
  const result = rgbToImageData(out, H, W);
  const overlay = maskOverlayImageData(base, filled255, H, W);     // moved-edge mask over base
  const overlayPre = maskOverlayImageData(base, textMask255, H, W); // pre-edge mask over base
  const textLayer = textLayerImageData(base, textMask255, H, W);
  const maskTransparent = maskTransparentImageData(filled255, H, W);

  let boxes = [];
  try {
    boxes = maskToBoxes(textMask255, H, W);
  } catch (_e) { /* boxes are best-effort for the status count */ }

  const cfg = {
    glow_mode: "auto",
    deglow_scheme: (opts.deglow === false) ? "off" : "v2",
    deglow_strength: opts.deglowStrength ?? 1.0,
    deglow_green_thr: opts.deglowGreenThr ?? 6.0,
    deglow_range: opts.deglowRange ?? 24,
    deglow_glo: opts.deglowGlo ?? 85.0,
    deglow_protect: opts.deglowProtect ?? 1.0,
    deglow_chroma_keep: opts.deglowChromaKeep ?? true,
    fill_white: opts.fillWhite ?? true,
    fill_max_dist: opts.fillMaxDist ?? 12,
    auto_edge: opts.autoEdge ?? false,
    auto_max_edge: opts.autoMaxEdge ?? 2,
  };
  let maskPix = 0;
  for (let i = 0; i < n; i++) if (textMask255[i]) maskPix++;

  return {
    result,
    overlay,            // = overlay_b64 source
    maskOverlay: overlay, // = mask_b64 source
    overlayPre,         // = overlay_pre_b64 source
    textLayer,          // = text_layer_b64 source
    maskTransparent,    // = mask_transparent_b64 source
    deglow: hasGlow ? rgbToImageData(deglowImg, H, W) : null,
    glowZone: hasGlow ? maskOverlayImageData(rgb, glowZone255, H, W) : null,
    boxes,
    cfg,
    maskPix,
    edgeUsed: edge,
    autoEdge: false,
    hasGlow,
  };
}

