"""Web 后端：POST /api/erase 上传图片，返回擦除结果 PNG + meta。

启动：
  python -m text_eraser            # 等价于 uvicorn text_eraser.webapp:app
  uvicorn text_eraser.webapp:app --host 127.0.0.1 --port 8765
"""
from __future__ import annotations

import io
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from text_eraser import __version__
from text_eraser.eraser import erase_text

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent
STATIC_DIR = _PACKAGE_DIR / "static"


def _default_data_dir() -> Path:
    """仓库 checkout 用 <repo>/data; pip 安装落到 ~/.text_eraser/data
    (site-packages 不应写运行数据)。环境变量 TEXT_ERASER_DATA_DIR 优先。"""
    env = os.environ.get("TEXT_ERASER_DATA_DIR")
    if env:
        return Path(env)
    if (_REPO_ROOT / "data").is_dir():
        return _REPO_ROOT / "data"
    return Path.home() / ".text_eraser" / "data"


DATA_DIR = _default_data_dir()
HISTORY_DIR = DATA_DIR / "history"

app = FastAPI(title="Text Eraser", version=__version__)

# static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    """首页：上传 + 预览界面."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": __version__, "ts": int(time.time())}


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

    try:
        result, mask, meta = erase_text(
            rgb,
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
        },
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
    host = os.environ.get("TEXT_ERASER_HOST", "127.0.0.1")
    port = int(os.environ.get("TEXT_ERASER_PORT", "8765"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
