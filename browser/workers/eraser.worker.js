// eraser.worker.js — Web Worker wrapper for the text-eraser browser module.
//
// Usage (main thread):
//   const w = new Worker(new URL('./eraser.worker.js', import.meta.url), { type: 'module' });
//   w.postMessage({ type: 'init', id: 1, opts: { dbnet: { /* modelUrl? */ } } });
//   w.postMessage({ type: 'eraseTextGlyphs', id: 2, imageData, opts: { textMask, edge: 1 } });
//   w.postMessage({ type: 'cancel', id: 2 });   // abort an in-flight job
//   w.onmessage = (e) => { ... e.data.{id,type,imageData,mask,error} ... };
//
// The worker owns opencv.js (loaded via dynamic import) and optionally a DBNet session,
// so the UI thread never blocks on wasm init or inpaint.

import {
  init, ensureOpenCV, inpaint, eraseTextGlyphs,
  loadDBNet, detectTextMask,
} from '../src/index.js';
import { imageDataToRgb } from '../src/linalg.js';

const cancelFlags = new Map(); // id -> bool

function makeCancel(id) {
  return () => cancelFlags.get(id) === true;
}

async function handle(msg) {
  const { id, type } = msg;
  try {
    switch (type) {
      case 'init': {
        const res = await init(msg.opts || {});
        return { id, type: 'init:done', hasDbnet: !!res.dbnetSession };
      }
      case 'inpaint': {
        const out = await inpaint(msg.imageData, {
          ...(msg.opts || {}), shouldCancel: makeCancel(id),
        });
        return { id, type: 'result', imageData: out };
      }
      case 'eraseTextGlyphs': {
        const out = await eraseTextGlyphs(msg.imageData, {
          ...(msg.opts || {}), shouldCancel: makeCancel(id),
        });
        return { id, type: 'result', imageData: out };
      }
      case 'detectTextMask': {
        await ensureOpenCV();
        const { width, height, data } = msg.imageData;
        const rgb = imageDataToRgb(data, height, width);
        const session = msg.session || (msg.opts && msg.opts.session);
        if (!session) throw new Error('detectTextMask: no DBNet session; call init({dbnet:{}}) first');
        const mask = await detectTextMask(rgb, height, width, { session, ...(msg.opts || {}) });
        return { id, type: 'mask', mask };
      }
      case 'cancel': {
        cancelFlags.set(id, true);
        return null;
      }
      default:
        throw new Error('eraser.worker: unknown message type ' + type);
    }
  } catch (e) {
    return { id, type: 'error', error: e && e.message ? e.message : String(e) };
  } finally {
    cancelFlags.delete(id);
  }
}

self.onmessage = async (e) => {
  const result = await handle(e.data);
  if (result) {
    if (result.type === 'result' || result.type === 'mask') {
      const transfer = [];
      if (result.imageData) transfer.push(result.imageData.data.buffer);
      if (result.mask) transfer.push(result.mask.buffer);
      self.postMessage(result, transfer);
    } else {
      self.postMessage(result);
    }
  }
};
