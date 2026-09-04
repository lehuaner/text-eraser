// smoke.js — browser-side acceptance harness for text-eraser-browser.
//
// Loads an image + mask, runs inpaint()/eraseTextGlyphs(), and (when a reference
// PNG is supplied) computes the per-channel maxAbsDiff — the same pixel-level
// gate (≈3) used by scripts/smoke_universal.py on the Python side.

import { inpaint, eraseTextGlyphs } from '../src/index.js';
import { maxAbsDiff } from '../src/linalg.js';

const $ = (id) => document.getElementById(id);
function log(msg, cls) {
  const el = $('log');
  const line = document.createElement('div');
  if (cls) line.className = cls;
  line.textContent = msg;
  el.appendChild(line);
}
function clearLog() { $('log').innerHTML = ''; }

async function fileToImageData(file) {
  const bmp = await createImageBitmap(file);
  const c = document.createElement('canvas');
  c.width = bmp.width; c.height = bmp.height;
  const ctx = c.getContext('2d');
  ctx.drawImage(bmp, 0, 0);
  const data = ctx.getImageData(0, 0, bmp.width, bmp.height);
  bmp.close();
  return data;
}

// grayscale mask PNG -> Uint8Array(H*W), 255 where pixel > 127 (uses red channel)
async function fileToMask(file) {
  const img = await fileToImageData(file);
  const n = img.width * img.height;
  const m = new Uint8Array(n);
  for (let i = 0; i < n; i++) m[i] = img.data[i * 4] > 127 ? 255 : 0;
  return m;
}

function draw(canvas, imageData) {
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  canvas.getContext('2d').putImageData(imageData, 0, 0);
}

function maskPreview(imageData, mask) {
  const out = new ImageData(imageData.width, imageData.height);
  for (let i = 0; i < imageData.width * imageData.height; i++) {
    const v = mask[i] ? 255 : 0;
    out.data[i * 4] = v; out.data[i * 4 + 1] = v; out.data[i * 4 + 2] = v; out.data[i * 4 + 3] = 255;
  }
  return out;
}

function reportDiff(label, result, reference) {
  const n = result.width * result.height;
  if (result.width !== reference.width || result.height !== reference.height) {
    log(`${label}: size mismatch vs reference — cannot compare`, 'bad');
    return;
  }
  let maxR = 0, maxG = 0, maxB = 0;
  for (let i = 0; i < n; i++) {
    maxR = Math.max(maxR, Math.abs(result.data[i * 4] - reference.data[i * 4]));
    maxG = Math.max(maxG, Math.abs(result.data[i * 4 + 1] - reference.data[i * 4 + 1]));
    maxB = Math.max(maxB, Math.abs(result.data[i * 4 + 2] - reference.data[i * 4 + 2]));
  }
  const m = Math.max(maxR, maxG, maxB);
  const ok = m <= 3;
  log(`${label}: maxDiff = ${m.toFixed(2)}  (R ${maxR.toFixed(2)} / G ${maxG.toFixed(2)} / B ${maxB.toFixed(2)})  → ${ok ? 'PASS (≤3)' : 'CHECK (>3)'}`, ok ? 'ok' : 'bad');
}

$('runInpaint').onclick = async () => {
  clearLog();
  try {
    if (!$('imgFile').files[0] || !$('maskFile').files[0]) { log('Need image + fill mask.', 'bad'); return; }
    log('Loading image + mask…');
    const img = await fileToImageData($('imgFile').files[0]);
    const mask = await fileToMask($('maskFile').files[0]);
    draw($('cIn'), img); draw($('cMask'), maskPreview(img, mask));
    log('Running inpaint()…');
    const t0 = performance.now();
    const out = await inpaint(img, { mask, flatTex: parseFloat($('flatTex').value) });
    log(`inpaint() done in ${(performance.now() - t0).toFixed(0)} ms`, 'ok');
    draw($('cOut'), out);
    if ($('refFile').files[0]) {
      const ref = await fileToImageData($('refFile').files[0]);
      draw($('cRef'), ref);
      reportDiff('inpaint', out, ref);
    }
  } catch (e) {
    log('ERROR: ' + (e && e.message ? e.message : e), 'bad');
  }
};

$('runErase').onclick = async () => {
  clearLog();
  try {
    if (!$('imgFile').files[0] || !$('txtFile').files[0]) { log('Need image + text mask.', 'bad'); return; }
    log('Loading image + text mask…');
    const img = await fileToImageData($('imgFile').files[0]);
    const textMask = await fileToMask($('txtFile').files[0]);
    draw($('cIn'), img); draw($('cMask'), maskPreview(img, textMask));
    log('Running eraseTextGlyphs()…');
    const t0 = performance.now();
    const out = await eraseTextGlyphs(img, {
      textMask,
      edge: parseInt($('edge').value, 10) || 1,
      deglow: $('deglow').checked,
      deglowStrength: 1.0,
      flatTex: parseFloat($('flatTex').value),
    });
    log(`eraseTextGlyphs() done in ${(performance.now() - t0).toFixed(0)} ms`, 'ok');
    draw($('cOut'), out);
    if ($('refFile').files[0]) {
      const ref = await fileToImageData($('refFile').files[0]);
      draw($('cRef'), ref);
      reportDiff('eraseTextGlyphs', out, ref);
    }
  } catch (e) {
    log('ERROR: ' + (e && e.message ? e.message : e), 'bad');
  }
};

log('Ready. Load an image + mask, then run.');
