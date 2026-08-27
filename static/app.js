/* Text Eraser frontend — vanilla JS, no framework. */

(function () {
  const $ = (id) => document.getElementById(id);
  const fileInput = $("fileInput");
  const exampleBtn = $("exampleBtn");
  const runBtn = $("runBtn");
  const downloadBtn = $("downloadBtn");
  const downloadTextBtn = $("downloadTextBtn");
  const downloadMaskBtn = $("downloadMaskBtn");

  const boxOrig = $("boxOrig");
  const boxMask = $("boxMask");
  const boxText = $("boxText");
  const boxResult = $("boxResult");
  const boxDeglow = $("boxDeglow");
  const panelDeglow = $("panelDeglow");

  const status = $("status");
  const maskPadEl = $("maskPad");
  const qOffEl = $("qOff");
  const maxAreaRatioEl = $("maxAreaRatio");
  const maxBoxRatioEl = $("maxBoxRatio");
  const directionEl = $("direction");
  const edgeAwareEl = $("edgeAware");
  const edgeExtendEl = $("edgeExtend");
  const glowModeEl = $("glowMode");
  const deglowStrengthEl = $("deglowStrength");
  const deglowGreenThrEl = $("deglowGreenThr");
  const deglowRangeEl = $("deglowRange");
  const deglowGloEl = $("deglowGlo");
  const deglowProtectEl = $("deglowProtect");
  const deglowMaskSoftEl = $("deglowMaskSoft");
  const deglowSchemeEl = $("deglowScheme");

  let currentFile = null;
  let resultB64 = null;
  let resultBlobUrl = null;
  let textTb64 = null;
  let maskTb64 = null;

  // ---- 批量(轮询)结果翻页状态 ----
  const batchPager = $("batchPager");
  const batchPrev = $("batchPrev");
  const batchNext = $("batchNext");
  const batchCounter = $("batchCounter");
  const batchName = $("batchName");
  let batchResults = [];   // [{name, d}] 轮询过程中每张图的完整结果
  let batchIdx = -1;

  function setStatus(msg, cls = "") {
    status.textContent = msg;
    status.className = "status" + (cls ? " " + cls : "");
  }

  function setImg(box, src, alt) {
    if (box.querySelector("img")) box.querySelector("img").remove();
    if (box.querySelector(".hint")) box.querySelector(".hint").remove();
    const img = document.createElement("img");
    img.src = src;
    img.alt = alt;
    box.appendChild(img);
  }

  function resetPreview(msg = "请上传图片") {
    boxOrig.innerHTML = `<span class="hint">${msg}</span>`;
    boxMask.innerHTML = `<span class="hint">蒙版会在擦除完成后显示</span>`;
    boxText.innerHTML = `<span class="hint">文字图层会在擦除完成后显示</span>`;
    boxResult.innerHTML = `<span class="hint">结果会在擦除完成后显示</span>`;
    boxDeglow.innerHTML = `<span class="hint">开启「先去发光再去字」后显示</span>`;
    panelDeglow.hidden = true;
    runBtn.disabled = true;
    downloadBtn.hidden = true;
    downloadTextBtn.hidden = true;
    downloadMaskBtn.hidden = true;
    if (resultBlobUrl) URL.revokeObjectURL(resultBlobUrl);
    resultBlobUrl = null;
    resultB64 = null;
    textTb64 = null;
    maskTb64 = null;
    clearBatch();
  }

  fileInput.addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (!f) return;
    currentFile = f;
    clearBatch();
    boxOrig.innerHTML = "";
    const url = URL.createObjectURL(f);
    setImg(boxOrig, url, "原图");
    setStatus(`已选择 ${f.name} (${(f.size / 1024).toFixed(1)} KB)`);
    runBtn.disabled = false;
    boxMask.innerHTML = `<span class="hint">蒙版会在擦除完成后显示</span>`;
    boxText.innerHTML = `<span class="hint">文字图层会在擦除完成后显示</span>`;
    boxResult.innerHTML = `<span class="hint">点击"擦除"开始</span>`;
    downloadBtn.hidden = true;
    downloadTextBtn.hidden = true;
    downloadMaskBtn.hidden = true;
  });

  exampleBtn.addEventListener("click", async () => {
    setStatus("加载示例图…", "working");
    try {
      const r = await fetch("/api/example.png");
      if (!r.ok) throw new Error(r.statusText);
      const blob = await r.blob();
      const f = new File([blob], "example.png", { type: "image/png" });
      const dt = new DataTransfer();
      dt.items.add(f);
      fileInput.files = dt.files;
      fileInput.dispatchEvent(new Event("change"));
    } catch (e) {
      setStatus("示例加载失败: " + e.message, "error");
    }
  });

  function buildForm(file) {
    const form = new FormData();
    form.append("image", file);
    form.append("mask_pad", maskPadEl.value);
    form.append("q_off", qOffEl.value);
    form.append("max_area_ratio", maxAreaRatioEl.value);
    form.append("max_box_ratio", maxBoxRatioEl.value);
    const dirVal = directionEl.value.trim();
    if (dirVal !== "") form.append("direction", dirVal);
    form.append("edge_aware", edgeAwareEl.checked ? "true" : "false");
    form.append("edge_extend", edgeExtendEl.value);
    form.append("glow_mode", glowModeEl.value);
    form.append("deglow_strength", deglowStrengthEl.value);
    form.append("deglow_green_thr", deglowGreenThrEl.value);
    form.append("deglow_range", deglowRangeEl.value);
    form.append("deglow_glo", deglowGloEl.value);
    form.append("deglow_protect", deglowProtectEl.value);
    form.append("deglow_mask_soft", deglowMaskSoftEl.value);
    form.append("deglow_scheme", deglowSchemeEl.value);
    form.append("return_overlay", "true");
    return form;
  }

  function downloadB64(b64, mime, name) {
    const url = URL.createObjectURL(base64ToBlob(b64, mime));
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 3000);
  }

  async function submitErase(file, nameHint) {
    const t0 = performance.now();
    const r = await fetch("/api/erase", { method: "POST", body: buildForm(file) });
    const j = await r.json();
    const elapsedMs = (performance.now() - t0).toFixed(0);
    if (!j.ok) throw new Error(j.msg || "后端失败");
    const d = j.data;
    displayErase(d, nameHint, elapsedMs);
    return d;
  }

  /* 把后端擦除结果渲染到四个预览面板 + 下载按钮 */
  function displayErase(d, nameHint, elapsedMs) {
    setImg(boxMask, "data:image/png;base64," + d.overlay_b64, "红蒙版(叠原图)");
    if (d.text_layer_b64) {
      setImg(boxText, "data:image/png;base64," + d.text_layer_b64, "文字图层");
    } else {
      setImg(boxText, "data:image/png;base64," + d.mask_b64, "文字图层");
    }
    setImg(boxResult, "data:image/png;base64," + d.result_b64, "结果");
    showDeglow(d);
    resultB64 = d.result_b64;
    if (resultBlobUrl) URL.revokeObjectURL(resultBlobUrl);
    resultBlobUrl = URL.createObjectURL(base64ToBlob(d.result_b64, "image/png"));
    downloadBtn.href = "#";
    const dlName = (nameHint || (currentFile && currentFile.name) || "img.png");
    const stem = dlName.replace(/\.[^.]+$/, "");
    downloadBtn.onclick = (e) => {
      e.preventDefault();
      downloadB64(d.result_b64, "image/png", "erased_" + dlName);
    };
    downloadBtn.hidden = false;

    // 中间产物：文字图层 & 蒙版透明版（RGBA）
    if (d.text_layer_b64) {
      textTb64 = d.text_layer_b64;
      downloadTextBtn.hidden = false;
      downloadTextBtn.onclick = (e) => {
        e.preventDefault();
        downloadB64(textTb64, "image/png", "text_layer_" + stem + ".png");
      };
    } else {
      downloadTextBtn.hidden = true;
    }
    if (d.mask_transparent_b64) {
      maskTb64 = d.mask_transparent_b64;
      downloadMaskBtn.hidden = false;
      downloadMaskBtn.onclick = (e) => {
        e.preventDefault();
        downloadB64(maskTb64, "image/png", "mask_alpha_" + stem + ".png");
      };
    } else {
      downloadMaskBtn.hidden = true;
    }

    const boxes = d.boxes || [];
    const cfg = d.cfg || {};
    const cfgTxt = (cfg.glow_mode && cfg.deglow_scheme !== "v4")
      ? ` • 发光[${cfg.glow_mode} 阈值${cfg.deglow_green_thr} 范围${cfg.deglow_range} 亮度${cfg.deglow_glo} 保护${cfg.deglow_protect} 软扩${cfg.deglow_mask_soft} 强度${cfg.deglow_strength}]`
      : "";
    setStatus(
      `完成 — 用时 ${elapsedMs} ms（后端 ${d.elapsed}s） • mask ${d.mask_pix}px • 检测到 ${boxes.length} 个文字框${cfgTxt}`,
      "success"
    );
  }

  /* deglow_first 实验：显示「去除发光后的全图」中间结果 */
  function showDeglow(d) {
    if (d && d.deglow_b64) {
      setImg(boxDeglow, "data:image/png;base64," + d.deglow_b64, "去发光后");
      panelDeglow.hidden = false;
    } else {
      panelDeglow.hidden = true;
    }
  }

  /* ---- 批量结果翻页 ---- */
  function clearBatch() {
    batchResults = [];
    batchIdx = -1;
    batchPager.hidden = true;
  }

  function showBatchPage(i) {
    if (i < 0 || i >= batchResults.length) return;
    batchIdx = i;
    const { name, d } = batchResults[i];
    // 显示当前张：原图(取历史原图接口或本地 blob) — 批量轮询时原图来自后端
    if (d.orig_data_url) {
      setImg(boxOrig, d.orig_data_url, "原图");
    }
    setImg(boxMask, "data:image/png;base64," + d.overlay_b64, "红蒙版(叠原图)");
    if (d.text_layer_b64) {
      setImg(boxText, "data:image/png;base64," + d.text_layer_b64, "文字图层");
    } else {
      setImg(boxText, "data:image/png;base64," + d.mask_b64, "文字图层");
    }
    setImg(boxResult, "data:image/png;base64," + d.result_b64, "结果");
    resultB64 = d.result_b64;
    if (resultBlobUrl) URL.revokeObjectURL(resultBlobUrl);
    resultBlobUrl = URL.createObjectURL(base64ToBlob(d.result_b64, "image/png"));
    downloadBtn.href = "#";
    const stem = (name || "img").replace(/\.[^.]+$/, "");
    downloadBtn.onclick = (e) => {
      e.preventDefault();
      downloadB64(d.result_b64, "image/png", "erased_" + name);
    };
    downloadBtn.hidden = false;
    showDeglow(d);
    downloadTextBtn.hidden = !d.text_layer_b64;
    if (d.text_layer_b64) {
      downloadTextBtn.onclick = (e) => {
        e.preventDefault();
        downloadB64(d.text_layer_b64, "image/png", "text_layer_" + stem + ".png");
      };
    }
    downloadMaskBtn.hidden = !d.mask_transparent_b64;
    if (d.mask_transparent_b64) {
      downloadMaskBtn.onclick = (e) => {
        e.preventDefault();
        downloadB64(d.mask_transparent_b64, "image/png", "mask_alpha_" + stem + ".png");
      };
    }
    batchCounter.textContent = `${i + 1}/${batchResults.length}`;
    batchName.textContent = name || "";
    batchPrev.disabled = i <= 0;
    batchNext.disabled = i >= batchResults.length - 1;
    setStatus(`批量结果 ${i + 1}/${batchResults.length} — ${name || ""}`, "success");
  }

  batchPrev.addEventListener("click", () => showBatchPage(batchIdx - 1));
  batchNext.addEventListener("click", () => showBatchPage(batchIdx + 1));

  runBtn.addEventListener("click", async () => {
    if (!currentFile) return;
    runBtn.disabled = true;
    try {
      await submitErase(currentFile);
    } catch (e) {
      setStatus("失败: " + e.message, "error");
    } finally {
      runBtn.disabled = false;
    }
  });

  // ---- 历史记录面板 ----
  const histBtn = $("histBtn");
  const histPanel = $("histPanel");
  const histGrid = $("histGrid");
  const histNote = $("histNote");
  const histRunAll = $("histRunAll");
  let histItems = [];

  function histNoteEl(msg, cls) {
    histNote.textContent = msg;
    histNote.className = "hist-note" + (cls ? " " + cls : "");
  }

  async function loadHistory() {
    histNoteEl("加载历史…", "busy");
    try {
      const r = await fetch("/api/history");
      const j = await r.json();
      histItems = j.items || [];
      renderHistory();
      histNoteEl(histItems.length ? `${histItems.length} 条记录` : "暂无历史记录", "ok");
    } catch (e) {
      histNoteEl("加载失败: " + e.message, "err");
    }
  }

  function renderHistory() {
    histGrid.innerHTML = "";
    for (const it of histItems) {
      const el = document.createElement("div");
      el.className = "hist-item";
      el.dataset.id = it.id;
      const date = it.ts ? new Date(it.ts * 1000).toLocaleString() : "";
      el.innerHTML =
        `<img src="data:image/png;base64,${it.thumb_b64}" alt="">` +
        `<div class="meta">${(it.name || "image").slice(0, 26)}${it.w ? ` · ${it.w}×${it.h}` : ""}<br>${date}</div>`;
      el.addEventListener("click", () => selectHistoryImage(it));
      histGrid.appendChild(el);
    }
  }

  /* 点击历史缩略图 → 作为"当前选择图片"（走选择图片路径，需再点擦除） */
  async function selectHistoryImage(it) {
    histNoteEl(`已加载「${it.name}」到预览 — 点“擦除”或调整参数后再跑`, "ok");
    try {
      const r = await fetch(`/api/history/${it.id}/orig`);
      if (!r.ok) throw new Error(r.statusText);
      const blob = await r.blob();
      const f = new File([blob], it.name || "history.png", { type: "image/png" });
      const dt = new DataTransfer();
      dt.items.add(f);
      fileInput.files = dt.files;
      fileInput.dispatchEvent(new Event("change"));
    } catch (e) {
      histNoteEl("加载失败: " + e.message, "err");
    }
  }

  /* 轮询：按历史顺序依次对每张图启动擦除并展示结果 */
  histRunAll.addEventListener("click", async () => {
    if (!histItems.length) { await loadHistory(); }
    if (!histItems.length) { histNoteEl("历史为空", "err"); return; }
    histRunAll.disabled = true;
    clearBatch();
    try {
      for (let i = 0; i < histItems.length; i++) {
        const it = histItems[i];
        histNoteEl(`正在处理 ${i + 1}/${histItems.length} — ${it.name}`, "busy");
        const r = await fetch(`/api/history/${it.id}/orig`);
        if (!r.ok) continue;
        const blob = await r.blob();
        const f = new File([blob], it.name || "history.png", { type: "image/png" });
        const d = await submitErase(f, it.name || "history.png");
        // 记录批量结果 → 翻页可回看每一张
        d.orig_data_url = URL.createObjectURL(blob);
        batchResults.push({ name: it.name || "history.png", d });
        const card = histGrid.querySelector(`.hist-item[data-id="${it.id}"]`);
        if (card) card.classList.add("done");
        // 每处理完一张就把翻页条切到最新一张
        batchPager.hidden = false;
        showBatchPage(batchResults.length - 1);
      }
      histNoteEl(`轮询完成：处理了 ${histItems.length} 张`, "ok");
    } catch (e) {
      histNoteEl("轮询中断: " + e.message, "err");
    } finally {
      histRunAll.disabled = false;
    }
  });

  histBtn.addEventListener("click", () => {
    const willShow = histPanel.hidden;
    histPanel.hidden = !willShow;
    if (willShow) loadHistory();
  });
  $("histClose").addEventListener("click", () => { histPanel.hidden = true; });

  function base64ToBlob(b64, mime) {
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return new Blob([arr], { type: mime });
  }

  resetPreview();
})();
