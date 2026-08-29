# -*- coding: utf-8 -*-
"""二分定位: 556 文字区黑色填充由本轮哪个改动引入。

构造 4 个 clean 变体(关/开各机制), 分别走完整 erase_text, 对比填充区亮度:
  A. HEAD 原始行为(全局标量 comp, 无尾外扩, B 只做旧式全局shift)
  B. A + B场逐像素对齐
  C. B + 局部暖度减绿(无尾外扩)
  D. C + 晕尾外扩14px(= 当前完整实现)
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from text_eraser.text_select import detect_text_mask, _geodesic_background  # noqa: E402
import text_eraser.eraser as eraser  # noqa: E402

OLD = ROOT / "data" / "_diag556seam" / "_text_select_old.py"
OLD.write_text(
    subprocess.run(["git", "show", "70847b2:core/text_select.py"], cwd=ROOT,
                   capture_output=True, text=True, encoding="utf-8").stdout,
    encoding="utf-8")
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("tso", OLD)
old_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_mod)

meta = json.loads((ROOT / "data/history/1787822778556/meta.json").read_text(encoding="utf-8"))
raw = (ROOT / "data/history/1787822778556/orig.bin").read_bytes()
rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR),
                   cv2.COLOR_BGR2RGB)
H, W = rgb.shape[:2]
tmask, _ = detect_text_mask(rgb, method="ml", q_off=55.0, max_area_ratio=0.4,
                            max_box_ratio=0.4, max_side=960, tint_fill=False,
                            fill_white=True, fill_max_dist=12)
KW = dict(strength=1.0, zone_ratio=0.6, zone_expand=10, protect_px=1,
          deglow_chroma_keep=True, return_zone=True)


def make_variant(use_local, use_tail, use_align):
    """按开关组合构造 _deglow_full_green_v2 变体(镜像当前实现的结构)。"""
    def variant(rgb, tmask, g_thr=2, g_lo=60, min_strong=30, white_floor=120,
                rounds_max=400, strength=1.0, alpha_core=0.65, zone_ratio=0.6,
                zone_expand=10, protect_px=1, deglow_chroma_keep=False,
                debug=False, return_zone=False):
        out = rgb.astype(np.int16)
        r = rgb[..., 0].astype(np.int16)
        g = rgb[..., 1].astype(np.int16)
        b = rgb[..., 2].astype(np.int16)
        s = float(np.clip(strength, 0.0, 1.5))
        empty = np.zeros((H, W), np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        green = (g - np.maximum(r, b) > 2) & (g > 60)
        strong_green = (g - np.maximum(r, b) > 8) & (g > 95)
        if strong_green.any():
            _n, _lab, _stats = cv2.connectedComponentsWithStats(
                strong_green.astype(np.uint8), 8)[:3]
            _max_cc = int(_stats[1:, 4].max()) if _n > 1 else 0
        else:
            _max_cc = 0
        if _max_cc < 30 or s <= 0:
            tm = (tmask > 0).astype(np.uint8) * 255
            return (rgb, tm, zone_grown) if False else ((rgb, tm, empty) if return_zone else (rgb, tm))
        min_rgb = np.minimum(np.minimum(r, g), b)
        text_stroke = (min_rgb > 120) & ((g - np.maximum(r, b)) < 40)
        bg_cand = gray[~strong_green]
        bg_lum = float(np.median(bg_cand))
        _gg = np.maximum(g - np.maximum(r, b), 0)
        bright = ((gray > (bg_lum + 6)) & (gray > 55) & (_gg > 2))
        faint = (g - np.maximum(r, b) > 3) & (g > 55)
        zone = (strong_green | (tmask > 0)).copy()
        cur = zone
        budget = int(H * W * zone_ratio)
        k3 = np.ones((3, 3), np.uint8)
        for _ in range(400):
            dil = cv2.dilate(cur.astype(np.uint8), k3) > 0
            add = dil & (green | bright | faint) & ~zone
            if not add.any():
                break
            zone |= add
            if int(zone.sum()) > budget:
                zone &= ~add
                break
            cur = zone
        zone_grown = zone.copy()
        if zone_expand > 0:
            _ze = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (zone_expand * 2 + 1,) * 2)
            zone = cv2.dilate(zone.astype(np.uint8), _ze) > 0
        m_zone = zone
        greenness = np.maximum(g.astype(np.int16) - np.maximum(r, b), 0)
        _ring = (cv2.dilate(zone.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0) & ~zone
        d_warm = max(0.0, float(np.median((r - g)[_ring]))) if _ring.any() else 0.0
        B = D_rg = D_gb = None
        if zone.any() and zone.sum() < 0.8 * H * W:
            geo = cv2.erode(zone.astype(np.uint8), k3, iterations=3) > 0
            _dout = cv2.distanceTransform((~zone).astype(np.uint8), cv2.DIST_L2, 5)
            _rc = ((~zone) & (_dout >= 10.0) & (_dout <= 26.0) & (greenness <= 6))
            if _rc.any():
                B, (D_rg, D_gb) = _geodesic_background(
                    rgb, geo,
                    extra=[(r - g).astype(np.float32), (g - b).astype(np.float32)],
                    extra_src=_rc)
            else:
                B = _geodesic_background(rgb, geo)
        if use_local and d_warm > 0 and D_rg is not None:
            glow = np.maximum(D_rg - (r - g).astype(np.float32), 0.0)
            glow[text_stroke] = greenness[text_stroke]
            if use_tail:
                m_zone = cv2.dilate(m_zone.astype(np.uint8), cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (29, 29))) > 0
        else:
            comp = np.where(text_stroke, 0.0, d_warm).astype(np.float32)
            glow = np.maximum(greenness + comp, 0)
        Gn = out[m_zone, 1].astype(np.float32) - glow[m_zone] * s
        out[m_zone, 1] = np.clip(Gn, 0, 255).astype(np.int16)
        protect2 = cv2.dilate(text_stroke.astype(np.uint8), k3,
                              iterations=max(0, int(protect_px))) > 0
        _bg25 = float(np.percentile(gray[~zone], 25))
        _cand = ((gray > _bg25 + 20) & (min_rgb >= 92) &
                 (greenness >= 25) & (greenness < 80))
        _cur = text_stroke & zone
        for _ in range(10):
            _add = (cv2.dilate(_cur.astype(np.uint8), k3) > 0) & _cand & ~_cur
            if not _add.any():
                break
            _cur |= _add
        protect2 |= (_cur & zone)
        fb = zone & ~protect2
        if fb.any() and zone.sum() < 0.8 * H * W and B is not None:
            if use_align and d_warm > 0 and D_rg is not None:
                B = np.stack([B[..., 0], B[..., 0] - D_rg,
                              B[..., 0] - D_rg - D_gb], axis=-1)
                np.clip(B, 0, 255, out=B)
            elif not use_align and d_warm > 0 and fb.any():
                _bR = B[..., 0].astype(np.float32)[fb]
                _bG = B[..., 1].astype(np.float32)[fb]
                _shift = float(d_warm - np.median(_bR - _bG))
                if _shift > 0:
                    B = B.astype(np.float32)
                    B[..., 1] = np.clip(B[..., 1] - _shift, 0, 255)
            imgf = rgb.astype(np.float32)
            dtext = cv2.distanceTransform((~protect2).astype(np.uint8) * 255,
                                          cv2.DIST_L2, 5)
            dw = np.clip(dtext / 8.0, 0.0, 1.0)[..., None]
            detail = (imgf - cv2.GaussianBlur(imgf, (0, 0), 2.0)) * dw
            rebuilt = np.clip(B + detail, 0, 255)
            if deglow_chroma_keep:
                rk = r[fb].astype(np.float32)
                gk = g[fb].astype(np.float32)
                bk = b[fb].astype(np.float32)
                ggreen = gk - np.maximum(rk, bk)
                rb_hot = (np.abs(rk - bk) > 8).astype(np.float32)
                a = (ggreen < 20.0).astype(np.float32) * rb_hot * 0.85
                a_map = np.zeros((H, W), np.float32)
                a_map[fb] = a
                a_map = cv2.GaussianBlur(a_map, (0, 0), 1.5)
                keep_layer = imgf.copy()
                keep_layer[..., 1] = out[..., 1]
                rebuilt = rebuilt * (1 - a_map[..., None]) + keep_layer * a_map[..., None]
            for c in range(3):
                out[..., c] = np.where(fb, rebuilt[..., c].astype(np.int16),
                                       out[..., c])
            _w = np.clip((greenness - 5.0) / 20.0, 0.0, 1.0)[..., None]
            _mix = out.astype(np.float32) * (1 - _w) + rebuilt * _w
            for c in range(3):
                out[..., c] = np.where(fb, _mix[..., c].astype(np.int16),
                                       out[..., c])
        _k8 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        tsz = text_stroke & (cv2.dilate(strong_green.astype(np.uint8), _k8) > 0)
        core_mask = tsz.astype(np.uint8) * 255
        clean = out.clip(0, 255).astype(np.uint8)
        return (clean, core_mask, zone) if return_zone else (clean, core_mask)
    return variant


PK = ["edge", "q_off", "max_area_ratio", "max_box_ratio", "direction", "edge_aware",
      "glow_mode", "deglow_strength", "deglow_green_thr", "deglow_range", "deglow_glo",
      "deglow_protect", "deglow_mask_soft", "deglow_zone_ratio", "deglow_zone_expand",
      "deglow_protect_px", "deglow_chroma_keep", "deglow_scheme", "fill_white",
      "fill_max_dist", "auto_edge", "auto_max_edge"]
kw = {k: meta["params"][k] for k in PK if k in meta["params"]}

VARIANTS = [
    ("A HEAD原行为",         dict(use_local=False, use_tail=False, use_align=False)),
    ("B +B场对齐",           dict(use_local=False, use_tail=False, use_align=True)),
    ("C +局部暖度减绿",       dict(use_local=True,  use_tail=False, use_align=True)),
    ("D +晕尾外扩(当前)",     dict(use_local=True,  use_tail=True,  use_align=True)),
]
new_fn = eraser._deglow_full_green_v2
box = (slice(75, 171), slice(134, 233))
for name, cfg in VARIANTS:
    eraser._deglow_full_green_v2 = make_variant(**cfg)
    try:
        res, mask, m = eraser.erase_text(rgb, return_mask=True, **kw)
    finally:
        eraser._deglow_full_green_v2 = new_fn
    L = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dark = int((L[box] < 55).sum())
    print(f"{name:<22s} 文字框均亮度={L[box].mean():6.1f}  暗于55的px={dark:5d}")
    cv2.imwrite(str(ROOT / "data/_diag556seam" / f"bisect_{name[0]}.png"),
                cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
