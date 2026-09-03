// browser-engine.mjs — main-thread bridge to the pure-browser text-eraser port.
//
// Loaded as <script type="module"> from index.html. Exposes window.TextEraserBrowser so
// the classic app.js can run the full algorithm in the browser (the "本地浏览器计算"
// toggle). ALL heavy work (opencv.js wasm compile + DBNet + patchmatch) now runs inside a
// classic Web Worker (eraser.worker.js); this module only forwards messages, so the page
// never freezes. The returned `data` shape is identical to the backend `/api/erase`.
//
// Why a classic worker: opencv.js is a UMD build that only attaches `cv` via importScripts
// (see eraser.worker.js). A module worker cannot load it.

const WORKER_URL = '/static/eraser.worker.js?v=20260903';
const INIT_TIMEOUT_MS = 15000;   // 初始化应在数秒内完成，超时过快失败便于用户定位问题

let _worker = null;
let _ready = null;
let _hasDbnet = false;
let _dbnetError = null;

function getWorker() {
  if (_worker) return _worker;
  _worker = new Worker(WORKER_URL);   // classic worker (no {type:'module'})
  return _worker;
}

/**
 * Talk to the worker: send a request message, resolve on `respType`, reject on 'error'
 * or worker failure or timeout. Optionally transfer `transfer` buffers and report
 * intermediate `progress` messages via `onProgress`.
 */
function callWorker(message, { respType, timeoutMs = INIT_TIMEOUT_MS, transfer = [], onProgress } = {}) {
  const w = getWorker();
  return new Promise((resolve, reject) => {
    let done = false;
    const timer = timeoutMs && timeoutMs > 0
      ? setTimeout(() => {
          if (done) return;
          done = true;
          cleanup();
          reject(new Error('浏览器引擎初始化超时（' + (timeoutMs / 1000) + 's）。' +
            'opencv.js 与文字检测模型较大，请检查网络/控制台，或改用后端计算。'));
        }, timeoutMs)
      : null;

    function onMessage(e) {
      const d = e.data || {};
      if (d.type === 'progress') {
        if (typeof onProgress === 'function') onProgress(d);
        return;   // not a terminal message
      }
      if (d.type === 'error') {
        finish(new Error(d.message || '浏览器引擎错误'));
        return;
      }
      if (d.type === respType) {
        finish(null, d);
      }
    }
    function onError(err) {
      finish(err instanceof Error ? err : new Error(String(err && err.message || err)));
    }
    function cleanup() {
      clearTimeout(timer);
      w.removeEventListener('message', onMessage);
      w.removeEventListener('error', onError);
    }
    function finish(err, data) {
      if (done) return;
      done = true;
      cleanup();
      if (err) reject(err);
      else resolve(data);
    }
    w.addEventListener('message', onMessage);
    w.addEventListener('error', onError);
    w.postMessage(message, transfer);
  });
}

/**
 * Load opencv.js (wasm) and optionally a DBNet session, inside the worker. Idempotent.
 * @param {{dbnet?: boolean, onProgress?: function}} opts
 */
export async function initEngine(opts = {}) {
  if (_ready) return _ready;
  _ready = (async () => {
    const d = await callWorker(
      { type: 'init', dbnet: opts.dbnet !== false },
      { respType: 'ready', onProgress: opts.onProgress },
    );
    _hasDbnet = !!d.hasDbnet;
    _dbnetError = d.dbnetError || null;
  })();
  // If init fails, allow a later retry (reset the cached promise).
  _ready.catch(() => { _ready = null; });
  return _ready;
}

export function hasDbnet() { return _hasDbnet; }
export function dbnetError() { return _dbnetError; }

/**
 * Run the full erase pipeline in the browser worker and return the same `data` shape as
 * the backend `/api/erase` (base64 PNG fields), so app.js::displayErase() is reused as-is.
 * @param {ImageData} imageData
 * @param {object} params  frontend param object (edge, direction, deglow, ...)
 * @returns {Promise<object>} data
 */
export async function eraseWith(imageData, params = {}) {
  await initEngine({ dbnet: true });
  if (!_hasDbnet) {
    throw new Error('浏览器文字检测模型不可用：' + (_dbnetError || '未知原因') +
      '。请改用后端计算，或检查 /browser/vendor/ort/ 是否完整、服务端是否返回 COOP/COEP 头。');
  }
  // Transfer the pixel buffer to the worker (it owns it from here on).
  const transfer = imageData.data && imageData.data.buffer ? [imageData.data.buffer] : [];
  const resp = await callWorker(
    { type: 'erase', imageData, params },
    // Defensive cap: the worker keeps the UI alive, but never let a stuck run
    // hang the user silently. With the threaded onnxruntime wasm this
    // completes in seconds; this only catches pathological stalls.
    { respType: 'result', timeoutMs: 180000, transfer },
  );
  return resp.data;
}

window.TextEraserBrowser = { initEngine, eraseWith, hasDbnet, dbnetError };
