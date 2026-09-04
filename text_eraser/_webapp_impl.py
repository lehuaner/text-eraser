"""Web 界面实现（fastapi/uvicorn 属于 web extra）。

启动：
  text-eraser                      # 推荐入口 (缺失依赖时给出友好提示)
  python -m text_eraser            # 同上
  uvicorn text_eraser.webapp:app --host 127.0.0.1 --port 8765
"""
from __future__ import annotations

import io
import hashlib
import json
import logging
import mimetypes
import os
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Browsers refuse to execute module scripts / Workers served as text/plain.
# Python's stdlib mimetypes does not know .mjs → force it to JavaScript.
mimetypes.add_type("application/javascript", ".mjs")

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from text_eraser import __version__
from text_eraser.eraser import erase_text
# Shared WASM algorithm core — single source of truth shared with the browser.
# using_shared_core() reports whether the backend is dispatching through it (wasm
# loaded by wasmtime) or falling back to cv2. Surfaced to the UI as a badge.
# python_core() forces the ORIGINAL pure Python core (cv2/numpy) for one request —
# what the UI's「使用 Python 核心」switch sends as use_python_core=true.
from text_eraser._shared_core import python_core, using_shared_core

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent
STATIC_DIR = _PACKAGE_DIR / "static"


@asynccontextmanager
async def _lifespan(app):
    # Best-effort: make sure the "本地浏览器计算" assets (opencv.js, onnxruntime-web,
    # DBNet model, bundled pipeline) are present before the first request. They are
    # large 3rd-party artifacts that are NOT committed (see .gitignore), so we
    # download/copy them once on boot. Run in a background thread so an offline box
    # never stalls server boot. On failure we log; the toggle then surfaces a clear
    # error instead of the worker hanging 15s.
    import asyncio

    async def _bootstrap():
        try:
            from text_eraser._browser_assets import ensure_browser_assets

            res = await asyncio.to_thread(ensure_browser_assets)
            missing = [k for k, v in res.items() if v.startswith("error")]
            if missing:
                logger.warning("[browser-assets] 缺失/不可用: %s（离线或网络受限；"
                               "本地浏览器计算将不可用，可改用后端计算）", missing)
            else:
                logger.info("[browser-assets] 浏览器计算资源已就绪")
        except Exception as e:  # never block boot on asset issues
            logger.warning("[browser-assets] 准备失败（已忽略，不影响后端计算）: %s", e)

    task = asyncio.create_task(_bootstrap())
    try:
        yield
    finally:
        task.cancel()


def _env_first(*names: str) -> str:
    """按优先级取第一个非空环境变量 (0.2.0 统一为 TEXTERASER_*,
    旧名 TEXT_ERASER_* 仅作 0.1.x 兼容回退)。"""
    for name in names:
        v = os.environ.get(name)
        if v:
            return v
    return ""


def _default_data_dir() -> Path:
    """仓库 checkout 用 <repo>/data; pip 安装落到 ~/.text_eraser/data
    (site-packages 不应写运行数据)。环境变量 TEXTERASER_DATA_DIR 优先
    （旧名 TEXT_ERASER_DATA_DIR 兼容回退）。"""
    env = _env_first("TEXTERASER_DATA_DIR", "TEXT_ERASER_DATA_DIR")
    if env:
        return Path(env)
    if (_REPO_ROOT / "data").is_dir():
        return _REPO_ROOT / "data"
    return Path.home() / ".text_eraser" / "data"


DATA_DIR = _default_data_dir()
HISTORY_DIR = DATA_DIR / "history"

app = FastAPI(title="TextEraser", version=__version__, lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Cross-Origin-Isolation headers (COOP/COEP).
#
# The "本地浏览器计算" (pure-browser) path runs onnxruntime-web (DBNet text
# detection) in a Web Worker. onnxruntime-web's *threaded* wasm build spawns
# pthread workers that REQUIRE SharedArrayBuffer, which only exists when the
# document is cross-origin isolated (COOP: same-origin + COEP: require-corp).
# Without these headers the threaded wasm silently hangs for minutes during
# InferenceSession.create / session.run. Setting them makes the page
# `crossOriginIsolated` so the wasm works (and also lets the non-threaded
# fallback path resolve correctly).
# ---------------------------------------------------------------------------
@app.middleware("http")
async def _add_cross_origin_isolation(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    response.headers.setdefault("Cross-Origin-Embedder-Policy", "require-corp")
    # Allow same-origin workers / importScripts under COEP.
    response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
    # Never cache the browser-engine assets (worker + importScripts + wasm/models):
    # a stale cached bundle/worker is the usual cause of "still initializing".
    if request.url.path.startswith(("/static/", "/browser/", "/shared/")):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


# static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# serve the pure-browser ESM port so the web UI can run algorithms client-side
# (the "本地浏览器计算" toggle loads /browser/dist/te-bundle.js + opencv.js +
#  onnxruntime-web + DBNet model, all auto-bootstrapped into browser/vendor &
#  browser/dist by text_eraser._browser_assets on startup)
_BROWSER_DIR = _REPO_ROOT / "browser"
if _BROWSER_DIR.is_dir():
    app.mount("/browser", StaticFiles(directory=str(_BROWSER_DIR)), name="browser")

# serve the SHARED algorithm core wasm (textcore.wasm). BOTH the browser Worker
# (fetched over HTTP) and the Python backend (loaded via wasmtime from the same
# file on disk) run the exact same operators — this is the "前后端共用一套算法"
# single source of truth. The browser binding fetches /shared/build/textcore.wasm.
_SHARED_DIR = _REPO_ROOT / "shared"
if _SHARED_DIR.is_dir():
    app.mount("/shared", StaticFiles(directory=str(_SHARED_DIR)), name="shared")


@app.get("/")
async def index():
    """首页：上传 + 预览界面."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": __version__, "ts": int(time.time())}


@app.get("/api/browser-assets")
async def browser_assets():
    """浏览器计算所需资源是否已就位（供前端在开启「本地浏览器计算」前判断）。"""
    from text_eraser._browser_assets import browser_assets_status

    return browser_assets_status()


@app.get("/api/example.png")
async def example_png():
    """返回示例图: 仓库样图优先, pip 安装时回退到包内合成示例图."""
    p = DATA_DIR / "needExtractAndPatch.png"
    if not p.is_file():
        p = _PACKAGE_DIR / "assets" / "example.png"
    if not p.is_file():
        raise HTTPException(404, "内置示例图缺失")
    return FileResponse(str(p), media_type="image/png")


# ---------------------------------------------------------------------------
# 历史持久化 + 列表/原图接口（供前端「历史/轮询」使用）
# ---------------------------------------------------------------------------
def _save_history(raw: bytes, pil: Image.Image, meta: dict, data: dict,
                  params: dict) -> str:
    """把本次擦除的原始上传图 + 元信息写入 data/history/{id}/。返回 id。

    去重：按原图内容 MD5 —— 同一张图(字节相同)重复上传只保留一条历史，
    并刷新为最近一次的时间与参数(旧 id 复用，不新增目录堆积)。
    """
    digest = hashlib.md5(raw).hexdigest()
    # 1) 查重：已存在同内容记录 → 复用该 id
    if HISTORY_DIR.is_dir():
        for old in sorted(HISTORY_DIR.iterdir()):
            if not old.is_dir():
                continue
            om = _read_meta(old / "meta.json")
            if om is None:
                continue
            if om.get("md5") == digest:
                _write_meta(old, digest, data, meta, params)
                return om.get("id", old.name)
    # 2) 无重复 → 新建
    hid = str(int(time.time() * 1000))
    d = HISTORY_DIR / hid
    d.mkdir(parents=True, exist_ok=True)
    (d / "orig.bin").write_bytes(raw)          # 原图原始字节(PNG/JPG 原样)
    _write_meta(d, digest, data, meta, params)
    # 缩略图(最长边 180)
    th = pil.copy()
    th.thumbnail((180, 180))
    th.convert("RGB").save(d / "thumb.png", "PNG")
    _prune_history(20)
    return hid


def _write_meta(d: Path, digest: str, data: dict, meta: dict, params: dict):
    (d / "meta.json").write_text(json.dumps({
        "id": d.name,
        "name": data.get("orig_name", "image"),
        "ts": int(time.time()),
        "size": data.get("orig_size", [0, 0]),
        "params": params,
        "mask_pix": meta.get("mask_pix", 0),
        "elapsed": meta.get("inpaint_seconds", 0.0),
        "boxes": meta.get("boxes", []),
        "md5": digest,
    }, ensure_ascii=False, indent=1), encoding="utf-8")


def _read_meta(p: Path):
    """读 meta.json：UTF-8 优先，兼容历史 GBK 写入。失败返回 None."""
    try:
        raw = p.read_bytes()
    except Exception:
        return None
    for enc in ("utf-8", "gbk"):
        try:
            return json.loads(raw.decode(enc))
        except Exception:
            continue
    return None


def _prune_history(max_items: int = 20):
    """历史只保留最近 max_items 条(旧目录删除)。"""
    if not HISTORY_DIR.is_dir():
        return
    dirs = sorted([p for p in HISTORY_DIR.iterdir() if p.is_dir()],
                  key=lambda p: p.name)
    for p in dirs[:-max_items]:
        import shutil
        shutil.rmtree(p, ignore_errors=True)


def _dedupe_history():
    """迁移 + 去重（启动时执行）：为无 md5 的旧记录补算并清理重复内容记录。"""
    if not HISTORY_DIR.is_dir():
        return
    hashes: dict[str, list] = {}
    for d in list(HISTORY_DIR.iterdir()):
        if not d.is_dir():
            continue
        om = _read_meta(d / "meta.json")
        if om is None:
            continue
        orig = d / "orig.bin"
        if not orig.is_file():
            continue
        digest = om.get("md5")
        if not digest:
            digest = hashlib.md5(orig.read_bytes()).hexdigest()
            om["md5"] = digest
            (d / "meta.json").write_text(json.dumps(om, ensure_ascii=False, indent=1))
        hashes.setdefault(digest, []).append((int(om.get("ts", 0) or 0), d))
    for digest, entries in hashes.items():
        if len(entries) <= 1:
            continue
        entries.sort(key=lambda e: e[0], reverse=True)   # 最新在前
        for _, dup in entries[1:]:
            import shutil
            shutil.rmtree(dup, ignore_errors=True)


_dedupe_history()


@app.get("/api/history")
async def history_list():
    """历史列表: 新→旧, 每项含缩略图 base64."""
    from base64 import b64encode
    items = []
    if HISTORY_DIR.is_dir():
        dirs = sorted([p for p in HISTORY_DIR.iterdir() if p.is_dir()],
                      key=lambda p: p.name, reverse=True)
        for d in dirs:
            meta_p = d / "meta.json"
            if not meta_p.is_file():
                continue
            m = _read_meta(meta_p)
            if m is None:
                continue
            thumb_b64 = ""
            tp = d / "thumb.png"
            if tp.is_file():
                buf = io.BytesIO()
                Image.open(tp).save(buf, "PNG")
                thumb_b64 = b64encode(buf.getvalue()).decode("ascii")
            items.append({
                "id": m.get("id", d.name),
                "name": m.get("name", "image"),
                "ts": m.get("ts", 0),
                "w": (m.get("size") or [0, 0])[0],
                "h": (m.get("size") or [0, 0])[1],
                "thumb_b64": thumb_b64,
            })
    # 按最近处理时间排序(旧 id 复用后目录名不变, 需以 meta.ts 为准)
    items.sort(key=lambda it: it["ts"] or 0, reverse=True)
    return JSONResponse({"ok": True, "items": items})


@app.get("/api/history/{hid}/orig")
async def history_orig(hid: str):
    """返回历史原图（原始上传的 PNG/JPG 字节）。"""
    p = HISTORY_DIR / hid / "orig.bin"
    if not p.is_file():
        raise HTTPException(404, "历史记录不存在")
    return Response(content=p.read_bytes(),
                    media_type="image/png" if p.name.endswith(("png", "bin")) else "application/octet-stream")


@app.delete("/api/history/{hid}")
async def history_delete(hid: str):
    """删除单条历史记录（含原图、缩略图、元信息目录）。"""
    p = HISTORY_DIR / hid
    if not p.is_dir():
        raise HTTPException(404, "历史记录不存在")
    try:
        shutil.rmtree(p)
    except Exception as e:
        raise HTTPException(500, f"删除失败: {e}")
    return JSONResponse({"ok": True, "id": hid})


@app.post("/api/erase")
async def erase(
    image: UploadFile = File(...),
    q_off: float = Form(55.0),
    edge: int = Form(1),
    max_area_ratio: float = Form(0.40),
    max_box_ratio: float = Form(0.40),
    direction: float = Form(None),
    edge_aware: bool = Form(False),
    return_overlay: bool = Form(True),
    glow_mode: str = Form("auto"),
    deglow_strength: float = Form(1.0),
    deglow_green_thr: float = Form(6.0),
    deglow_range: int = Form(24),
    deglow_glo: float = Form(85.0),
    deglow_protect: float = Form(1.0),
    deglow_mask_soft: float = Form(0.0),
    deglow_zone_ratio: float = Form(0.6),
    deglow_zone_expand: int = Form(10),
    deglow_protect_px: int = Form(1),
    deglow_chroma_keep: bool = Form(True),
    # 唯一去发光算法 = v2(减绿度去发光 → 非高亮算法去字); "off" = 关闭去发光
    deglow_scheme: str = Form("v2"),
    fill_white: bool = Form(True),
    fill_max_dist: int = Form(12),
    auto_edge: bool = Form(True),
    auto_max_edge: int = Form(2),
    # 计算核心选择：true = 走「原本的 Python 核心」(cv2/numpy 全流程, 不碰 wasm);
    # false(默认) = 走共享 WASM 算法核(与浏览器同一份 textcore.wasm)。
    use_python_core: bool = Form(False),
):
    """擦除上传图片中的文字。返回 JSON:
        {
          "ok": true,
          "data": {
            "result_b64": "...",
            "overlay_b64": "...",  # 仅当 return_overlay=True
            "mask_pix": 2357,
            "elapsed": 0.871,
            "boxes": [{"x0":99,"y0":39,"x1":154,"y1":93}, ...]
          }
        }
    """
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "空文件")

    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"图片解析失败: {e}")

    rgb = np.asarray(pil, dtype=np.uint8)
    H, W = rgb.shape[:2]
    if H * W == 0:
        raise HTTPException(400, "空图像")

    # use_python_core=True → 整条后端流水线(算子/去发光 v2/PatchMatch 填充)全部回到
    # 原本的 Python 实现; 否则维持共享 WASM 核。开关只作用于本次请求(ContextVar)。
    erase_kwargs = dict(
        edge=edge,
        q_off=q_off,
        max_area_ratio=max_area_ratio,
        max_box_ratio=max_box_ratio,
        direction=direction,
        edge_aware=edge_aware,
        glow_mode=glow_mode,
        deglow_strength=deglow_strength,
        deglow_green_thr=deglow_green_thr,
        deglow_range=deglow_range,
        deglow_glo=deglow_glo,
        deglow_protect=deglow_protect,
        deglow_mask_soft=deglow_mask_soft,
        deglow_zone_ratio=deglow_zone_ratio,
        deglow_zone_expand=deglow_zone_expand,
        deglow_protect_px=deglow_protect_px,
        deglow_chroma_keep=deglow_chroma_keep,
        deglow_scheme=deglow_scheme,
        fill_white=fill_white,
        fill_max_dist=fill_max_dist,
        auto_edge=auto_edge,
        auto_max_edge=auto_max_edge,
        return_mask=True,
    )
    try:
        with python_core(use_python_core):
            result, mask, meta = erase_text(rgb, **erase_kwargs)
            # 在开关生效的作用域内快照实际引擎, 供前端徽标显示(离开 with 后即失效)
            engine_wasm = bool(using_shared_core())
    except Exception as e:
        raise HTTPException(500, f"算法失败: {e}")

    # 编码 result 与 overlay (红底蒙版叠原图)
    def _png(arr: np.ndarray) -> str:
        from base64 import b64encode
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, "PNG", optimize=False)
        return b64encode(buf.getvalue()).decode("ascii")

    data = {
        "result_b64": _png(result),
        "mask_pix": meta.get("mask_pix", 0),
        "elapsed": meta.get("inpaint_seconds", 0.0),
        "boxes": meta.get("boxes", []),
        "orig_size": [W, H],
        "orig_name": image.filename or "image",
        "edge_used": meta.get("edge_used", edge),
        "auto_edge": meta.get("auto_edge", False),
        "cfg": {
            "glow_mode": glow_mode,
            "deglow_scheme": deglow_scheme,
            "deglow_strength": deglow_strength,
            "deglow_green_thr": deglow_green_thr,
            "deglow_range": deglow_range,
            "deglow_glo": deglow_glo,
            "deglow_protect": deglow_protect,
            "deglow_mask_soft": deglow_mask_soft,
            "deglow_zone_ratio": deglow_zone_ratio,
            "deglow_zone_expand": deglow_zone_expand,
            "deglow_protect_px": deglow_protect_px,
            "deglow_chroma_keep": deglow_chroma_keep,
            "fill_white": fill_white,
            "fill_max_dist": fill_max_dist,
            "auto_edge": auto_edge,
            "auto_max_edge": auto_max_edge,
            "use_python_core": use_python_core,
        },
        # 当前是否经「共享算法核」(textcore.wasm) 计算 —— 后端与浏览器共用同一份算子，
        # 前端据此点亮「共享核」徽标，让用户直观看到两种模式都跑的是同一套算法。
        # 勾选「使用 Python 核心」时为 false(本次请求走纯 Python 实现)。
        "shared_core": engine_wasm,
        # 实际使用的计算核心: "wasm" = 共享 WASM 核; "python" = 原本的 Python 核心。
        # use_python_core=false 但 wasm 加载失败时也会是 "python"(自动回退)。
        "engine": "wasm" if engine_wasm else "python",
        # 用户是否显式要求 Python 核心(区分"主动选择"与"wasm 不可用被动回退")
        "engine_requested": "python" if use_python_core else "wasm",
    }

    if return_overlay:
        # 发光区非空 = v2 实际执行了去发光。此时文字蒙版/填充蒙版的检测与
        # 填充底图都是「去发光图」, 分步展示的蒙版叠加也用它 —— 与算法一致,
        # 否则会看到「红蒙版盖着发光字」的错觉。
        gz = meta.get("glow_zone")
        has_glow = gz is not None and bool((gz > 0).any())
        base = rgb
        if has_glow:
            dglow0 = meta.get("deglow_img")
            if dglow0 is not None:
                base = dglow0

        overlay = base.copy()
        m_bool = mask > 0
        if m_bool.any():
            overlay[m_bool] = (
                base[m_bool].astype(np.int32) * 0.35
                + np.array([255, 60, 60]) * 0.65
            ).clip(0, 255).astype(np.uint8)
        # 透明度扩展的软带: 以衰减的半透明红显示("红蒙版范围扩大")
        sa = meta.get("soft_alpha")
        if sa is not None and (sa > 0).any():
            sb = sa > 0
            overlay[sb] = (
                base[sb].astype(np.int32) * 0.72
                + (np.array([255, 60, 60], np.float32) * 0.28
                   * sa[sb, None]).clip(0, 255).astype(np.int32)
            ).clip(0, 255).astype(np.uint8)
        data["overlay_b64"] = _png(overlay)
        data["mask_b64"] = _png(mask)

        # 移动边缘前的文字蒙版(红蒙版叠加) —— 前端「文字蒙版」分步展示
        m_pre = meta.get("mask_pre_edge")
        if m_pre is not None and (m_pre > 0).any():
            ov_pre = base.copy()
            pb = m_pre > 0
            ov_pre[pb] = (
                base[pb].astype(np.int32) * 0.35
                + np.array([255, 60, 60]) * 0.65
            ).clip(0, 255).astype(np.uint8)
            data["overlay_pre_b64"] = _png(ov_pre)

        # 发光区蒙版(红蒙版叠原图, 展示"原图哪里在发光")
        if has_glow:
            ov_gz = rgb.copy()
            gb = gz > 0
            ov_gz[gb] = (
                rgb[gb].astype(np.int32) * 0.35
                + np.array([255, 60, 60]) * 0.65
            ).clip(0, 255).astype(np.uint8)
            data["glow_zone_b64"] = _png(ov_gz)

        # 去发光后的全图 —— 仅在「确实发生了去发光」时返回:
        # v2 对无发光图零改动(zone 空), 不返回 deglow_b64 → 前端不显示该面板;
        # 无 zone 概念的路径(如 deglow_first 实验)保持原行为。
        dglow = meta.get("deglow_img")
        if dglow is not None and (gz is None or has_glow):
            data["deglow_b64"] = _png(dglow)

        # v4 通用方案的每域结构化报告（溯源/模式/α 分位数）
        drep = meta.get("dglow_report")
        if drep is not None:
            data["dglow_report"] = drep

        # 中间产物 ①：蒙版透明版(RGBA) —— 蒙版区半透明红, 其余全透明
        mt = np.zeros((H, W, 4), np.uint8)
        mt[m_bool] = (255, 60, 60, 150)
        data["mask_transparent_b64"] = _png(mt)
        # 中间产物 ②：文字图层(RGBA) —— 只保留蒙版选中的文字像素, 其余透明。
        # 从 base(有发光=去发光图, 与真实填充底图一致)提取 —— 原图上的绿字
        # 已被去发光处理, 展示"实际将被擦除的文字"应是去发光后的样子。
        tl = np.zeros((H, W, 4), np.uint8)
        tl[m_bool, :3] = base[m_bool]
        tl[m_bool, 3] = 255
        data["text_layer_b64"] = _png(tl)

    # 持久化到历史: 原图(原始字节) + 元信息, 供前端历史/轮询复用
    try:
        hid = _save_history(raw, pil, meta, data, {
            "edge": edge, "q_off": q_off,
            "max_area_ratio": max_area_ratio, "max_box_ratio": max_box_ratio,
            "direction": direction, "edge_aware": edge_aware,
            "glow_mode": glow_mode,
            "deglow_strength": deglow_strength, "deglow_green_thr": deglow_green_thr,
            "deglow_range": deglow_range, "deglow_glo": deglow_glo,
            "deglow_protect": deglow_protect,             "deglow_mask_soft": deglow_mask_soft,
            "deglow_zone_ratio": deglow_zone_ratio,
            "deglow_zone_expand": deglow_zone_expand,
            "deglow_protect_px": deglow_protect_px,
            "deglow_chroma_keep": deglow_chroma_keep,
            "deglow_scheme": deglow_scheme,
            "fill_white": fill_white,
            "fill_max_dist": fill_max_dist,
            "auto_edge": auto_edge,
            "auto_max_edge": auto_max_edge,
        })
        data["history_id"] = hid
    except Exception:
        pass  # 历史保存失败不阻断主流程

    return JSONResponse({"ok": True, "code": 0, "msg": "ok", "data": data})


def main() -> None:
    """命令行入口: python -m text_eraser / text_eraser 命令。"""
    import uvicorn
    host = _env_first("TEXTERASER_HOST", "TEXT_ERASER_HOST") or "127.0.0.1"
    port = int(_env_first("TEXTERASER_PORT", "TEXT_ERASER_PORT") or "8765")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
