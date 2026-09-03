// eraser.worker.js — classic Web Worker that runs the full text-eraser pipeline
// (opencv.js + DBNet + patchmatch + deglow) OFF the main thread.
//
// Why a *classic* worker: opencv.js is a UMD build that only attaches `cv` when loaded
// via `importScripts` (its `typeof importScripts === 'function'` branch). A module worker
// cannot load it. So this file is a classic worker and pulls the bundled pipeline
// (`/browser/dist/te-bundle.js`, a classic IIFE exposing `self.TE`) the same way.
//
// The main thread (browser-engine.mjs) only forwards messages and never blocks, so the
// page stays responsive while the (heavy, synchronous wasm) compute runs here.

/* global importScripts, self, OffscreenCanvas, btoa, WorkerGlobalScope */

// Tell opencv.js where to fetch its wasm. opencv.js reads `self.Module.locateFile`
// at script-import time (the wasm fetch starts synchronously during importScripts),
// so this MUST be set BEFORE opencv.js is imported below / by the bundle. Without it
// opencv resolves `opencv_js.wasm` relative to the *worker* script URL (/static/)
// and 404s → onRuntimeInitialized never fires → init hangs forever.
// Also disable pthreads: the vendored opencv.js is a pthreads-enabled build, and on
// init it tries to fetch a `opencv_js.worker.js` pthread worker that we don't ship
// → 404 → hang. We run single-threaded, so pthreadPoolSize=0 stops that fetch.
self.Module = self.Module || {};
if (typeof self.Module.locateFile !== 'function') {
  self.Module.locateFile = (p) => '/browser/vendor/' + p + '?v=20260903';
}
self.Module.pthreadPoolSize = 0;

// ?v= cache-buster: forces the browser to re-fetch these assets even if an older
// build was cached (without it, a stale opencv.js could keep 404-ing on its wasm).
const OPENCV_URL = '/browser/vendor/opencv.js?v=20260903';
// onnxruntime-web in a *classic* worker MUST be loaded via importScripts of the UMD
// build (ort.min.js). The ESM build (ort.min.mjs) may NOT be dynamically imported
// inside a classic worker — it silently never resolves, hanging init forever.
const ORT_URL = '/browser/vendor/ort/ort.min.js?v=20260903';
const DBNET_URL = '/browser/vendor/ch_PP-OCRv4_det.onnx?v=20260903';

// Pull in the pipeline bundle (defines self.TE with ensureOpenCV / erase / loadDBNet …).
importScripts('/browser/dist/te-bundle.js?v=20260905');

let _session = null;     // DBNet onnxruntime session
let _cvReady = false;
let _hasDbnet = false;
let _dbnetError = null;

function progress(stage, extra) {
  self.postMessage(Object.assign({ type: 'progress', stage }, extra || {}));
}

// onnxruntime-web 1.27.0 only ships the *threaded* wasm (ort-wasm-simd-threaded.wasm);
// there is no single-thread build. So numThreads MUST be > 1, which in turn requires
// `crossOriginIsolated` (COOP/COEP). If the page isn't isolated we bail with a clear
// error instead of silently hanging on a missing wasm file.
function pickNumThreads() {
  const isolated = (typeof self.crossOriginIsolated !== 'undefined') && self.crossOriginIsolated;
  if (!isolated) return 0;                       // signal: cannot run onnx here
  const hw = (self.navigator && self.navigator.hardwareConcurrency) || 4;
  return Math.max(1, Math.min(hw, 4));
}

async function initEngine(dbnet) {
  progress('opencv');
  if (!_cvReady) {
    // diagnostic probe: what globals exist right now?
    progress('opencv:probe', { hasTE: typeof self.TE, hasCv: typeof self.cv });
    try {
      await self.TE.ensureOpenCV({ opencvUrl: OPENCV_URL });
      progress('opencv:done', {
        hasCv: typeof self.cv,
        mat: self.cv && typeof self.cv.Mat,
        build: self.cv && typeof self.cv.getBuildInformation,
      });
      _cvReady = true;
      // preload the shared WASM core (textcore.wasm); on failure the binding
      // falls back to pure-JS and usingSharedCore() reports false.
      try { await self.TE.ensureSharedCore(); } catch (_) {}
    } catch (e) {
      self.postMessage({ type: 'error', message: 'ensureOpenCV threw: ' + (e && e.message ? e.message : String(e)) });
      return;
    }
  }

  if (dbnet && !_session) {
    const nThreads = pickNumThreads();
    if (nThreads === 0) {
      _hasDbnet = false;
      _dbnetError = '当前页面未启用跨源隔离（crossOriginIsolated=false），无法加载 onnxruntime 线程版 wasm。' +
        '请确认服务端对 / 与 /static/eraser.worker.js 返回 COOP/COEP 响应头，或改用后端计算。';
      self.postMessage({ type: 'ready', hasDbnet: false, dbnetError: _dbnetError });
      return;
    }

    progress('ort', { threads: nThreads });
    let ort;
    try {
      // Classic worker: load the UMD build synchronously via importScripts (the only
      // supported way). It attaches `ort` to the worker global. Must happen before any
      // ort API call; numThreads>1 then drives the threaded wasm we vendored.
      importScripts(ORT_URL);
      ort = self.ort;
      if (!ort || !ort.env) throw new Error('onnxruntime-web UMD 未正确挂载到 self.ort');
    } catch (e) {
      _hasDbnet = false;
      _dbnetError = '加载 onnxruntime-web 失败：' + (e && e.message ? e.message : String(e));
      self.postMessage({ type: 'ready', hasDbnet: false, dbnetError: _dbnetError });
      return;
    }
    if (ort.env && ort.env.wasm) {
      ort.env.wasm.numThreads = nThreads;        // >1 → uses the threaded wasm we vendored
      ort.env.wasm.wasmPaths = '/browser/vendor/ort/';
      ort.env.wasm.simd = true;
    }

    progress('dbnet');
    try {
      _session = await self.TE.loadDBNet({ ort, modelUrl: DBNET_URL });
      _hasDbnet = true;
    } catch (e) {
      // Never hang: report the failure but still become "ready" so the main thread
      // gets a precise error instead of a 120s timeout.
      _hasDbnet = false;
      _dbnetError = '加载 DBNet 文字检测模型失败：' + (e && e.message ? e.message : String(e));
      self.postMessage({ type: 'ready', hasDbnet: false, dbnetError: _dbnetError });
      return;
    }
  }

  self.postMessage({ type: 'ready', hasDbnet: !!_session });
}

async function encodePng(imageData) {
  const c = new OffscreenCanvas(imageData.width, imageData.height);
  const ctx = c.getContext('2d');
  ctx.putImageData(imageData, 0, 0);
  const blob = await c.convertToBlob({ type: 'image/png' });
  const buf = new Uint8Array(await blob.arrayBuffer());
  // btoa over a large buffer in chunks to avoid call-stack limits.
  let bin = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < buf.length; i += CHUNK) {
    bin += String.fromCharCode.apply(null, buf.subarray(i, i + CHUNK));
  }
  return btoa(bin);
}

async function runErase(imageData, params) {
  const r = await self.TE.erase(imageData, {
    edge: params.edge != null ? params.edge : 1,
    direction: params.direction || null,
    deglow: params.deglow !== false,
    deglowStrength: params.deglowStrength ?? 1.0,
    deglowChromaKeep: params.deglowChromaKeep ?? true,
    dbnetSession: _session,
    strength: params.strength ?? 1.0,
    maskThreshold: params.maskThreshold ?? 0.4,
    maskMaxSide: params.maskMaxSide || 1600,
  });

  const data = {
    result_b64: await encodePng(r.result),
    overlay_b64: await encodePng(r.overlay),
    mask_b64: await encodePng(r.maskOverlay),
    overlay_pre_b64: await encodePng(r.overlayPre),
    text_layer_b64: await encodePng(r.textLayer),
    mask_transparent_b64: await encodePng(r.maskTransparent),
    boxes: r.boxes,
    cfg: r.cfg,
    mask_pix: r.maskPix,
    edge_used: r.edgeUsed,
    auto_edge: r.autoEdge,
    compute_source: '浏览器',
    shared_core: !!(self.TE && self.TE.usingSharedCore && self.TE.usingSharedCore()),
  };
  if (r.deglow) data.deglow_b64 = await encodePng(r.deglow);
  if (r.glowZone) data.glow_zone_b64 = await encodePng(r.glowZone);
  return data;
}

self.onmessage = async (e) => {
  const msg = e.data || {};
  try {
    if (msg.type === 'init') {
      await initEngine(msg.dbnet !== false);
    } else if (msg.type === 'erase') {
      progress('erase');
      const data = await runErase(msg.imageData, msg.params || {});
      self.postMessage({ type: 'result', data });
    } else {
      self.postMessage({ type: 'error', message: '未知消息类型: ' + msg.type });
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: (err && err.message) ? err.message : String(err) });
  }
};

// Surface early load failures (e.g. bundle 404) instead of hanging on the main thread.
self.postMessage({ type: 'worker-loaded' });
