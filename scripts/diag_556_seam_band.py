# -*- coding: utf-8 -*-
"""556(history 1787822778556) 去发光接缝「暗带+残绿」成因诊断。

复现前端 v2 去发光结果(clean), 并用插桩副本逐级输出中间量:
zone / protect2 / fb / B场 / detail / rebuilt / chroma-keep a_map / 软混合 w。
副本最终结果与真函数逐位断言一致, 中间量才可信。

输出: data/_diag556seam/
  - clean.png / orig.png
  - zoom_*.png 接缝放大
  - 控制台: 分带统计 + 径向 profile
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from textpatch.text_select import detect_text_mask, _deglow_full_green_v2, _geodesic_background  # noqa: E402

OUT = ROOT / "data" / "_diag556seam"
OUT.mkdir(parents=True, exist_ok=True)

HID = "1787822778556"
raw = (ROOT / "data" / "history" / HID / "orig.bin").read_bytes()
img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
assert img is not None and img.shape[:2] == (231, 369), img.shape
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = rgb.shape[:2]
cv2.imwrite(str(OUT / "orig.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

# ---- 1) 前端同参复现 --------------------------------------------------
tmask, _ = detect_text_mask(rgb, method="ml", q_off=55.0,
                            max_area_ratio=0.40, max_box_ratio=0.40,
                            max_side=960, tint_fill=False,
                            fill_white=True, fill_max_dist=12)
clean_real, _core, zone_real = _deglow_full_green_v2(
    rgb, tmask, strength=1.0, zone_ratio=0.6, zone_expand=10,
    protect_px=1, deglow_chroma_keep=True, return_zone=True)
cv2.imwrite(str(OUT / "clean.png"), cv2.cvtColor(clean_real, cv2.COLOR_RGB2BGR))

# ---- 2) 插桩副本(逐行等价 _deglow_full_green_v2) -----------------------
def instrumented(rgb, tmask, strength=1.0, alpha_core=0.65, zone_ratio=0.6,
                 zone_expand=10, protect_px=1, deglow_chroma_keep=True):
    cap = {}
    out = rgb.astype(np.int16)
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    s = float(np.clip(strength, 0.0, 1.5))
    Hh, Ww = rgb.shape[:2]
    empty = np.zeros((Hh, Ww), np.uint8)
    if s <= 0:
        return rgb, empty, empty, cap
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    green = (g - np.maximum(r, b) > 2) & (g > 60)
    strong_green = (g - np.maximum(r, b) > 8) & (g > 95)
    if strong_green.any():
        _n, _lab, _stats = cv2.connectedComponentsWithStats(
            strong_green.astype(np.uint8), 8)[:3]
        _max_cc = int(_stats[1:, 4].max()) if _n > 1 else 0
    else:
        _max_cc = 0
    if _max_cc < 30:
        return rgb, empty, empty, cap
    min_rgb = np.minimum(np.minimum(r, g), b)
    text_stroke = (min_rgb > 120) & ((g - np.maximum(r, b)) < 40)
    bg_cand = gray[~strong_green]
    bg_lum = float(np.median(bg_cand)) if bg_cand.size else 80.0
    _greenness_grow = np.maximum(g - np.maximum(r, b), 0)
    bright = ((gray > (bg_lum + 6)) & (gray > 55) & (_greenness_grow > 2))
    faint_green = (g - np.maximum(r, b) > 3) & (g > 55)
    grow_cond = green | bright | faint_green
    zone = (strong_green | (tmask > 0)).copy()
    cur = zone
    budget = int(Hh * Ww * zone_ratio)
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(400):
        dil = cv2.dilate(cur.astype(np.uint8), k3) > 0
        add = dil & grow_cond & ~zone
        if not add.any():
            break
        zone |= add
        if int(zone.sum()) > budget:
            zone &= ~add
            break
        cur = zone
    cap["zone_grown"] = zone.copy()          # 外扩前(仅诊断用)
    if zone_expand > 0:
        _ze = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (zone_expand * 2 + 1, zone_expand * 2 + 1))
        zone = cv2.dilate(zone.astype(np.uint8), _ze) > 0
    m_zone = zone
    cap["zone"] = zone.copy()
    if not m_zone.any():
        return rgb, (tmask > 0).astype(np.uint8) * 255, zone, cap
    greenness = np.maximum(g.astype(np.int16) - np.maximum(r, b), 0)
    cap["greenness"] = greenness.copy()
    _ring = (cv2.dilate(zone.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0) & ~zone
    d_warm = max(0.0, float(np.median((r - g)[_ring]))) if _ring.any() else 0.0
    cap["d_warm"] = d_warm
    B = D_rg = D_gb = None
    if zone.any() and zone.sum() < 0.8 * Hh * Ww:
        geo_mask0 = cv2.erode(zone.astype(np.uint8), k3, iterations=3) > 0
        _dout = cv2.distanceTransform((~zone).astype(np.uint8), cv2.DIST_L2, 5)
        _ring_clean = ((~zone) & (_dout >= 10.0) & (_dout <= 26.0)
                       & (greenness <= 6))
        cap["ring_clean"] = _ring_clean
        if _ring_clean.any():
            B, (D_rg, D_gb) = _geodesic_background(
                rgb, geo_mask0,
                extra=[(r - g).astype(np.float32), (g - b).astype(np.float32)],
                extra_src=_ring_clean)
        else:
            B = _geodesic_background(rgb, geo_mask0)
    cap["B"] = B.copy() if B is not None else None
    cap["D_rg"] = D_rg
    cap["D_gb"] = D_gb
    if d_warm > 0 and D_rg is not None:
        glow = np.maximum(D_rg - (r - g).astype(np.float32), 0.0)
        glow[text_stroke] = greenness[text_stroke]
        m_zone = cv2.dilate(m_zone.astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (29, 29))) > 0
    else:
        comp = np.where(text_stroke, 0.0, d_warm).astype(np.float32)
        glow = np.maximum(greenness + comp, 0)
    cap["glow"] = glow.copy()
    Gn = out[m_zone, 1].astype(np.float32) - glow[m_zone] * s
    out[m_zone, 1] = np.clip(Gn, 0, 255).astype(np.int16)
    cap["after_degreen"] = out.copy()
    protect2 = cv2.dilate(text_stroke.astype(np.uint8), k3,
                          iterations=max(0, int(protect_px))) > 0
    if zone.any():
        _zo = ~zone
        _bg25 = float(np.percentile(gray[_zo], 25)) if _zo.any() else 80.0
        _cand = ((gray > _bg25 + 20) & (min_rgb >= 92) &
                 (greenness >= 25) & (greenness < 80))
        _cur = text_stroke & zone
        for _ in range(10):
            _add = (cv2.dilate(_cur.astype(np.uint8), k3) > 0) & _cand & ~_cur
            if not _add.any():
                break
            _cur |= _add
        protect2 |= (_cur & zone)
    cap["protect2"] = protect2.copy()
    fb = zone & ~protect2
    cap["fb"] = fb.copy()
    rebuilt = None
    if fb.any() and zone.sum() < 0.8 * Hh * Ww and B is not None:
        from textpatch.text_select import _harmonic_background
        if d_warm > 0:
            _init = None
            if D_rg is not None:
                _init = np.stack([B[..., 0],
                                  B[..., 0] - D_rg,
                                  B[..., 0] - D_rg - D_gb], axis=-1)
                np.clip(_init, 0, 255, out=_init)
            _Bh = _harmonic_background(out, zone, init=_init)
            if _Bh is not None and D_rg is not None:
                _gL = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8),
                                   cv2.COLOR_RGB2GRAY).astype(np.float32)
                _gx = cv2.Sobel(_gL, cv2.CV_32F, 1, 0, ksize=3)
                _gy = cv2.Sobel(_gL, cv2.CV_32F, 0, 1, ksize=3)
                _str = np.clip(np.sqrt(_gx ** 2 + _gy ** 2), 0, 40).astype(np.float32)
                _dout2 = cv2.distanceTransform((~zone).astype(np.uint8),
                                               cv2.DIST_L2, 5)
                _rc2 = ((~zone) & (_dout2 >= 10.0) & (_dout2 <= 26.0)
                        & (greenness <= 6))
                if _rc2.any():
                    _, (_S,) = _geodesic_background(
                        rgb, cv2.erode(zone.astype(np.uint8), k3,
                                       iterations=3) > 0,
                        extra=[_str], extra_src=_rc2)
                    w = np.clip((_S - 4.0) / 10.0, 0.0, 1.0)
                    w = cv2.GaussianBlur(w, (0, 0), 4.0)[..., None]
                    B = w * _init + (1 - w) * _Bh
                    cap["w_struct"] = w.copy()
                else:
                    B = _Bh
            elif _init is not None:
                B = _init
            cap["B_aligned"] = B.copy()
        imgf = rgb.astype(np.float32)
        dtext = cv2.distanceTransform((~protect2).astype(np.uint8) * 255,
                                      cv2.DIST_L2, 5)
        dw = np.clip(dtext / 8.0, 0.0, 1.0)[..., None]
        cap["dw"] = dw[..., 0].copy()
        detail = (imgf - cv2.GaussianBlur(imgf, (0, 0), 2.0)) * dw
        rebuilt = np.clip(B + detail, 0, 255)
        cap["rebuilt_pre_keep"] = rebuilt.copy()
        if deglow_chroma_keep:
            rk = r[fb].astype(np.float32)
            gk = g[fb].astype(np.float32)
            bk = b[fb].astype(np.float32)
            ggreen = gk - np.maximum(rk, bk)
            rb_hot = (np.abs(rk - bk) > 8).astype(np.float32)
            a = (ggreen < 20.0).astype(np.float32) * rb_hot * 0.85
            a_map = np.zeros((Hh, Ww), np.float32)
            a_map[fb] = a
            a_map = cv2.GaussianBlur(a_map, (0, 0), 1.5)
            cap["a_map"] = a_map.copy()
            am = a_map[..., None]
            keep_layer = imgf.copy()
            keep_layer[..., 1] = out[..., 1]
            rebuilt = rebuilt * (1 - am) + keep_layer * am
        cap["rebuilt"] = rebuilt.copy()
        out_pre = out.copy()
        for c in range(3):
            out[..., c] = np.where(fb, rebuilt[..., c].astype(np.int16),
                                   out[..., c])
        _w = np.clip((greenness - 5.0) / 20.0, 0.0, 1.0)[..., None]
        _mix = out.astype(np.float32) * (1 - _w) + rebuilt * _w
        cap["out_pre_mix"] = out_pre
        cap["w"] = _w[..., 0].copy()
        _mix_dup = np.abs(_mix - rebuilt).max() if rebuilt is not None else 0
        cap["mix_minus_rebuilt_max"] = float(_mix_dup)
        for c in range(3):
            out[..., c] = np.where(fb, _mix[..., c].astype(np.int16),
                                   out[..., c])
    _k8 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    text_stroke_z = text_stroke & (cv2.dilate(strong_green.astype(np.uint8), _k8) > 0)
    core_mask = text_stroke_z.astype(np.uint8) * 255
    clean = out.clip(0, 255).astype(np.uint8)
    return clean, core_mask, zone, cap


clean_i, core_i, zone_i, cap = instrumented(rgb, tmask)
assert np.array_equal(clean_i, clean_real), "插桩副本与真函数结果不一致!"
assert np.array_equal(zone_i, zone_real), "zone 不一致!"
print("[ok] 插桩副本 == 真函数 (逐位一致)")
print(f"[info] d_warm = {cap.get('d_warm')}")
print(f"[info] B_shift = {cap.get('B_shift')}")
print(f"[info] mix_minus_rebuilt_max = {cap.get('mix_minus_rebuilt_max')}  "
      f"(=0 则软混合完全被架空)")

# ---- 3) 可视化 --------------------------------------------------------
zone = cap["zone"]; fb = cap["fb"]; prot = cap["protect2"]
ov = clean_real.copy()
ov[zone] = (ov[zone] * 0.5 + np.array([0, 60, 0]) * 0.5).astype(np.uint8)   # zone 红? 用BGR注意
ov[fb] = (ov[fb] * 0.55 + np.array([0, 0, 160]) * 0.45).astype(np.uint8)
ov[prot] = (ov[prot] * 0.55 + np.array([160, 0, 0]) * 0.45).astype(np.uint8)
cv2.imwrite(str(OUT / "ov_zone_fb_protect.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
# 图例: 红=fb重建区, 蓝=protect2, 绿罩=zone

B = cap["B"]
cv2.imwrite(str(OUT / "B_field.png"), cv2.cvtColor(B.clip(0, 255).astype(np.uint8),
                                                   cv2.COLOR_RGB2BGR))
amap = (cap.get("a_map", np.zeros((H, W), np.float32)) * 255).astype(np.uint8)
cv2.imwrite(str(OUT / "a_map.png"), amap)
wmap = (cap.get("w", np.zeros((H, W), np.float32)) * 255).astype(np.uint8)
cv2.imwrite(str(OUT / "w_map.png"), wmap)

# ---- 4) 分带量化 -------------------------------------------------------
r16 = rgb[..., 0].astype(np.int16); g16 = rgb[..., 1].astype(np.int16)
b16 = rgb[..., 2].astype(np.int16)
greenness = np.maximum(g16 - np.maximum(r16, b16), 0)
cg = clean_real[..., 1].astype(np.int16)
cr = clean_real[..., 0].astype(np.int16)
cb = clean_real[..., 2].astype(np.int16)
clean_green = cg - np.maximum(cr, cb)          # 结果图绿度(可负)
gray0 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
gray1 = cv2.cvtColor(clean_real, cv2.COLOR_RGB2GRAY).astype(np.float32)
dlum = gray1 - gray0                            # 结果-原图 亮度差

amap = cap.get("a_map", np.zeros((H, W), np.float32))
w = cap.get("w", np.zeros((H, W), np.float32))
Bint = B.clip(0, 255)

def band_stats(name, sel):
    if not sel.any():
        print(f"{name:<28s} 无像素")
        return
    n = int(sel.sum())
    print(f"{name:<28s} n={n:6d}  "
          f"clean绿度 med={np.median(clean_green[sel]):6.1f}  "
          f"Δlum med={np.median(dlum[sel]):6.1f}  "
          f"a_map med={np.median(amap[sel]):4.2f}  w med={np.median(w[sel]):4.2f}")

print("\n—— 结果图绿度异常区(clean绿度>6) ———")
resid = clean_green > 6
band_stats("全部残绿", resid)
band_stats("  其中 protect2 内", resid & prot)
band_stats("  其中 fb(重建区) 内", resid & fb)
band_stats("  其中 zone 外", resid & ~zone)

print("\n—— 变暗区(Δlum<−6) ———")
darker = dlum < -6
band_stats("全部变暗", darker)
band_stats("  fb 内", darker & fb)
band_stats("  protect2 内", darker & prot)
band_stats("  zone 外", darker & ~zone)

# 暗带定位: fb 内 Δlum 显著为负、且 a_map≈0(即完全走 B 场) 的环
print("\n—— 候选暗带 = fb & Δlum<−6 & a_map<0.1 ———")
cand = fb & (dlum < -6) & (amap < 0.1)
band_stats("候选暗带", cand)
if cand.any():
    ys, xs = np.nonzero(cand)
    print(f"  范围 x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]")

# 与 zone 边界的距离关系: 暗带到 zone 外边的距离分布
if cand.any():
    zo_edge = zone & ~cv2.erode(zone.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    dist_in = cv2.distanceTransform((~zo_edge).astype(np.uint8), cv2.DIST_L2, 5)
    print(f"  暗带像素到 zone 外边界距离: p10={np.percentile(dist_in[cand],10):.1f} "
          f"p50={np.percentile(dist_in[cand],50):.1f} p90={np.percentile(dist_in[cand],90):.1f}")

# ---- 5) 径向 profile --------------------------------------------------
def profile(x0, y0, dx, dy, n, tag):
    print(f"\n—— profile {tag}: from ({x0},{y0}) dir({dx},{dy}) ——")
    print(f"{'t':>3s} {'xy':>10s} {'origRGB':>15s} {'cleanRGB':>15s} "
          f"{'g0':>4s} {'g1':>4s} {'dlum':>5s} {'a':>4s} {'w':>4s} flags")
    for i in range(n):
        x = int(round(x0 + dx * i)); y = int(round(y0 + dy * i))
        if not (0 <= x < W and 0 <= y < H):
            break
        fl = ("Z" if zone[y, x] else "-") + ("F" if fb[y, x] else "-") + \
             ("P" if prot[y, x] else "-")
        print(f"{i:3d} ({x:3d},{y:3d}) "
              f"({r16[y,x]:3d},{g16[y,x]:3d},{b16[y,x]:3d}) "
              f"({cr[y,x]:3d},{cg[y,x]:3d},{cb[y,x]:3d}) "
              f"{greenness[y,x]:4d} {clean_green[y,x]:4d} "
              f"{dlum[y,x]:5.0f} {amap[y,x]:4.2f} {w[y,x]:4.2f} {fl}")

# 文字框(主框 x134-233 y75-171)中心向四方打线, 覆盖暗带/残绿
profile(184, 123, 0, -1, 70, "向上穿浅色块")
profile(184, 123, 0, +1, 70, "向下穿深色块")
profile(184, 123, -1, 0, 95, "向左")
profile(184, 123, +1, 0, 95, "向右")

# ---- 6) 接缝放大图 ----------------------------------------------------
def zoom(x0, y0, x1, y1, name, sc=4):
    a = rgb[y0:y1, x0:x1]
    c = clean_real[y0:y1, x0:x1]
    o = ov[y0:y1, x0:x1]
    z = np.concatenate([a, c, o], axis=1)
    z = cv2.resize(z, (z.shape[1] * sc, z.shape[0] * sc),
                   interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(OUT / name), cv2.cvtColor(z, cv2.COLOR_RGB2BGR))

if cand.any():
    ys, xs = np.nonzero(cand)
    cx, cy = int(np.median(xs)), int(np.median(ys))
    zoom(max(0, cx - 60), max(0, cy - 45), min(W, cx + 60), min(H, cy + 45),
         "zoom_seam_原图|clean|overlay.png")
zoom(120, 60, 250, 180, "zoom_text_原图|clean|overlay.png")
print(f"\n输出目录: {OUT}")
