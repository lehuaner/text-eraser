# -*- coding: utf-8 -*-
"""556 接缝暗带定量: 沿边界法向对比 zone 内外(结果图), 以及 B场 vs 真背景。"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.text_select import detect_text_mask, _deglow_full_green_v2  # noqa: E402

HID = "1787822778556"
raw = (ROOT / "data" / "history" / HID / "orig.bin").read_bytes()
img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = rgb.shape[:2]
OUT = ROOT / "data" / "_diag556seam"

tmask, _ = detect_text_mask(rgb, method="ml", q_off=55.0,
                            max_area_ratio=0.40, max_box_ratio=0.40,
                            max_side=960, tint_fill=False,
                            fill_white=True, fill_max_dist=12)
clean_real, _core, zone_real = _deglow_full_green_v2(
    rgb, tmask, strength=1.0, zone_ratio=0.6, zone_expand=10,
    protect_px=1, deglow_chroma_keep=True, return_zone=True)

# 复用 band 脚本的插桩副本拿中间量
sys.path.insert(0, str(ROOT / "scripts"))
import io
import contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import diag_556_seam_band as _band  # noqa: E402  (会重跑一遍, 断言一致)
cap = _band.cap

zone = cap["zone"]; fb = cap["fb"]; prot = cap["protect2"]
B = cap["B"]
a_map = cap.get("a_map", np.zeros((H, W), np.float32))
greenness = cap["greenness"]

r = rgb[..., 0].astype(np.int16); g = rgb[..., 1].astype(np.int16)
b = rgb[..., 2].astype(np.int16)
cr = clean_real[..., 0].astype(np.float32)
cg = clean_real[..., 1].astype(np.float32)
cb = clean_real[..., 2].astype(np.float32)
lum0 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
lum1 = cv2.cvtColor(clean_real, cv2.COLOR_RGB2GRAY).astype(np.float32)

# zone 外边界(不含 protect 等)
zo_edge = zone & ~cv2.erode(zone.astype(np.uint8),
                            np.ones((3, 3), np.uint8)).astype(bool)
# 法向: 用距离变换拿梯度方向
din = cv2.distanceTransform((zone).astype(np.uint8), cv2.DIST_L2, 5)
dout = cv2.distanceTransform((~zone).astype(np.uint8), cv2.DIST_L2, 5)
gy, gx = np.gradient(dout)
norm = np.sqrt(gx * gx + gy * gy) + 1e-6
nx, ny = gx / norm, gy / norm          # 指向 zone 外的方向

# 对每个边界像素: 内侧 2~5px 处结果 vs 外侧 2~5px 处结果
ys, xs = np.nonzero(zo_edge)
step = np.hypot(ys - 123, xs - 184) > 30     # 排除文字/保护圈附近
rows = []
for i, (y, x) in enumerate(zip(ys, xs)):
    if not step[i]:
        continue
    ux, uy = nx[y, x], ny[y, x]
    def sample(img, sgn, off):
        pts = []
        for t, o in ((2, -1), (3, 0), (4, 1), (5, 0)):
            xx = int(round(x + sgn * ux * t + uy * o * 0))
            yy = int(round(y + sgn * uy * t - ux * o * 0))
            if 0 <= xx < W and 0 <= yy < H:
                pts.append((yy, xx))
        if not pts:
            return None
        return np.mean([img[p] for p in pts], axis=0)
    pi = sample(lum1, -1, 0)   # 内侧结果亮度
    po = sample(lum1, +1, 0)   # 外侧结果亮度
    ci = sample(np.dstack([cr, cg, cb]), -1, 0)
    co = sample(np.dstack([cr, cg, cb]), +1, 0)
    bi = sample(B, -1, 0)
    if pi is None or po is None:
        continue
    ang = float(np.degrees(np.arctan2(y - 123, x - 184)))
    rows.append((ang, y, x, pi, po, ci, co,
                 float(lum0[y, x]), float(din[y, x]),
                 float(a_map[y, x]), int(greenness[y, x])))

rows.sort(key=lambda t: t[0])
print("angle  y  x   | inLum outLum Δ(out-in) | inRGB->outRGB | "
      "edgeOrigLum dIn a g0")
print("-" * 100)
for ang, y, x, pi, po, ci, co, ol, dinv, am, g0 in rows[::6]:
    print(f"{ang:6.0f} {y:3d} {x:3d} | {pi:6.1f} {po:6.1f} {po-pi:+7.1f} | "
          f"({ci[0]:3.0f},{ci[1]:3.0f},{ci[2]:3.0f})->"
          f"({co[0]:3.0f},{co[1]:3.0f},{co[2]:3.0f}) | "
          f"{ol:5.0f} {dinv:4.1f} {am:4.2f} {g0:3d}")

# 分扇区汇总
print("\n—— 按 45° 扇区汇总(Δ = 外侧-内侧, 正=外侧更亮) ——")
arr = np.array([(a, pi_, po_, ci_[0], ci_[1], ci_[2],
                 co_[0], co_[1], co_[2], ol, g0_)
                for a, _, _, pi_, po_, ci_, co_, ol, _, _, g0_ in rows])
# 列: 0=ang 1=inLum 2=outLum 3..5=内侧RGB 6..8=外侧RGB 9=edgeOrigLum 10=edge绿度
sec = (arr[:, 0] // 45).astype(int)
for k in range(8):
    m = sec == k
    if not m.any():
        continue
    d = arr[m, 2] - arr[m, 1]
    ci = arr[m, 3:6].mean(0); co = arr[m, 6:9].mean(0)
    print(f"{k*45:4d}~{(k+1)*45:4d}°  n={m.sum():3d}  "
          f"Δlum med={np.median(d):+6.1f}  "
          f"内侧RGB=({ci[0]:.0f},{ci[1]:.0f},{ci[2]:.0f}) "
          f"外侧RGB=({co[0]:.0f},{co[1]:.0f},{co[2]:.0f})  "
          f"内侧R-G={ci[0]-ci[1]:+4.1f} 外侧R-G={co[0]-co[1]:+4.1f}  "
          f"edge绿度 med={np.median(arr[m,10]):.0f}")

# 结果图残绿扫描
clean_g = clean_real[..., 1].astype(np.int16)
clean_maxrb = np.maximum(clean_real[..., 0], clean_real[..., 2]).astype(np.int16)
res_g = clean_g - clean_maxrb
for thr in (2, 4, 6, 10):
    m = res_g > thr
    print(f"\n结果图绿度>{thr}: {int(m.sum())}px", end="")
    if m.any():
        yy, xx = np.nonzero(m)
        print(f"  范围 x[{xx.min()},{xx.max()}] y[{yy.min()},{yy.max()}]", end="")
print()
# 原图在 zone 外的绿度(未被处理的漏网 glow)
outz = ~zone
print(f"原图 zone 外绿度>3: {int(((greenness > 3) & outz).sum())}px, "
      f">5: {int(((greenness > 5) & outz).sum())}px")
m = (greenness > 3) & outz
if m.any():
    yy, xx = np.nonzero(m)
    print(f"  范围 x[{xx.min()},{xx.max()}] y[{yy.min()},{yy.max()}]")
