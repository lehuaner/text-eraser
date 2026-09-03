# -*- coding: utf-8 -*-
"""556 暗带弧段放大 + 差异可视化: 找「泛绿弧段」到底在哪、是什么成分。"""
import sys
import io
import contextlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

with contextlib.redirect_stdout(io.StringIO()):
    import diag_556_seam_band as _band
rgb = _band.rgb
clean = _band.clean_real
cap = _band.cap
H, W = _band.H, _band.W
OUT = _band.OUT
zone = cap["zone"]; fb = cap["fb"]; prot = cap["protect2"]
B = cap["B"]
greenness = cap["greenness"]
a_map = cap.get("a_map", np.zeros((H, W), np.float32))

# ---- 差异图(×8, 128为中性): 看每通道被改了什么 ----
diff = clean.astype(np.int16) - rgb.astype(np.int16)
vis = np.clip(128 + diff * 8, 0, 255).astype(np.uint8)
cv2.imwrite(str(OUT / "diff_x8.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

# ---- 暗带弧段放大: 沿 zone 边界取 8 个方位, 原图|结果 并排 ----
zo_edge = zone & ~cv2.erode(zone.astype(np.uint8),
                            np.ones((3, 3), np.uint8)).astype(bool)
ys, xs = np.nonzero(zo_edge)
keep = np.hypot(ys - 123, xs - 184) > 40
ys, xs = ys[keep], xs[keep]
angs = np.degrees(np.arctan2(ys - 123, xs - 184))
tiles = []
for name, lo, hi in [("right", -30, 30), ("downright", 30, 75),
                     ("down", 75, 105), ("downleft", 105, 160),
                     ("left", 160, 200), ("upleft", -160, -110),
                     ("up", -70, -20)]:
    m = (angs >= lo) & (angs < hi)
    if not m.any():
        continue
    # 取该扇区中点到边界距离中位数的像素
    k = len(ys[m]) // 2
    yy, xx = ys[m][k], xs[m][k]
    x0, y0 = max(0, xx - 38), max(0, yy - 28)
    x1, y1 = min(W, xx + 38), min(H, yy + 28)
    a = rgb[y0:y1, x0:x1].copy()
    c = clean[y0:y1, x0:x1].copy()
    # zone 边界描红
    e = zo_edge[y0:y1, x0:x1]
    c[e] = [255, 0, 0]
    t = np.concatenate([a, c], axis=1)
    tiles.append(t)
mh = max(t.shape[0] for t in tiles)
tiles = [np.pad(t, ((0, mh - t.shape[0]), (0, 0), (0, 0)),
                constant_values=30) for t in tiles]
row1 = np.concatenate(tiles[:4], axis=0) if len(tiles) >= 4 else np.concatenate(tiles, axis=0)
row2 = np.concatenate(tiles[4:], axis=0) if len(tiles) > 4 else None
if row2 is not None and row2.shape[0] != row1.shape[0]:
    row2 = np.pad(row2, ((0, row1.shape[0] - row2.shape[0]), (0, 0), (0, 0)),
                  constant_values=30)
big = row1 if row2 is None else np.concatenate([row1, row2], axis=1)
big = cv2.resize(big, (big.shape[1] * 3, big.shape[0] * 3),
                 interpolation=cv2.INTER_NEAREST)
cv2.imwrite(str(OUT / "arcs_8dir_原图|结果_红线=zone边界.png"),
            cv2.cvtColor(big, cv2.COLOR_RGB2BGR))

# ---- 「泛绿」客观度量: 结果图相对外侧背景的色相偏移 ----
# 对 fb 内每个像素, 找同角度、边界外 6~14px 的背景带, 比较 (R-G, G-B)
dout = cv2.distanceTransform((~zone).astype(np.uint8), cv2.DIST_L2, 5)
din = cv2.distanceTransform((zone.astype(np.uint8)), cv2.DIST_L2, 5)
r = rgb[..., 0].astype(np.float32); g = rgb[..., 1].astype(np.float32)
b = rgb[..., 2].astype(np.float32)
cl = clean.astype(np.float32)
# 外侧参考: 环带 6~14px, 原图=结果(未处理)
ref_m = dout > 14
ref_rg = np.median((r - g)[ref_m])          # 全局
# 分扇区: 边界内 0~6px 的结果 vs 边界外 6~14px 的结果
inner_m = (din > 0) & (din < 6)
outer_m = (dout > 6) & (dout < 14)
print("扇区      内侧带(结果) RGB/R-G/G-B   外侧参考 RGB/R-G/G-B   ΔR-G ΔG-B")
for name, lo, hi in [("right", -30, 30), ("downright", 30, 75), ("down", 75, 105),
                     ("downleft", 105, 160), ("left", 160, 200),
                     ("upleft", -160, -110), ("up", -70, -20)]:
    yy, xx = np.nonzero(inner_m)
    ang = np.degrees(np.arctan2(yy - 123, xx - 184))
    m = (ang >= lo) & (ang < hi)
    if not m.any():
        continue
    yi, xi = yy[m], xx[m]
    ci = cl[yi, xi].mean(0)
    yy2, xx2 = np.nonzero(outer_m)
    ang2 = np.degrees(np.arctan2(yy2 - 123, xx2 - 184))
    # 外侧参考取「同方向 ±30°」的环带
    m2 = (np.minimum(np.abs(ang2 - lo), np.abs(ang2 - hi)) < 30) | \
         ((ang2 >= lo) & (ang2 < hi))
    if not m2.any():
        m2 = slice(None)
    co = cl[yy2[m2], xx2[m2]].mean(0)
    print(f"{name:>9s}  ({ci[0]:5.1f},{ci[1]:5.1f},{ci[2]:5.1f}) "
          f"R-G={ci[0]-ci[1]:+5.1f} G-B={ci[1]-ci[2]:+5.1f}   "
          f"({co[0]:5.1f},{co[1]:5.1f},{co[2]:5.1f}) "
          f"R-G={co[0]-co[1]:+5.1f} G-B={co[1]-co[2]:+5.1f}   "
          f"{(ci[0]-ci[1])-(co[0]-co[1]):+5.1f} {(ci[1]-ci[2])-(co[1]-co[2]):+5.1f}")

# ---- 暗带三个候选成分的分解统计 ----
print("\n—— 成分分解 ——")
fringe = fb & (a_map > 0.3)                     # chroma-keep 主导的边缘带
bcore = fb & (a_map <= 0.3)                     # B 场主导区
print(f"fb 总数 {int(fb.sum())}: chroma-keep 主导(a>0.3) {int(fringe.sum())}px, "
      f"B场主导(a≤0.3) {int(bcore.sum())}px")
cl16 = clean.astype(np.int16)
dlum = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.float32) - \
    cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
for nm, m in [("chroma-keep 边缘带", fringe), ("B场核心区", bcore),
              ("protect2(只减绿)", prot & zone)]:
    if not m.any():
        continue
    dg = (cl16[..., 1] - g.astype(np.int16))[m]
    print(f"{nm:<18s} n={int(m.sum()):6d}  ΔG med={np.median(dg):+5.1f}  "
          f"Δlum med={np.median(dlum[m]):+6.1f}")

# detail 项里残留的原图绿高频(近字处 dw<1, 远处 dw=1)
d = cap.get("dw")
if d is not None:
    print(f"\ndw<0.9 的 fb 像素(近保护圈, detail 被压制): "
          f"{int((fb & (d < 0.9)).sum())}px; dw≥0.9: {int((fb & (d >= 0.9)).sum())}px")
print(f"\nB 场统计: fb 内 R-G 中位={np.median((B[...,0]-B[...,1])[fb]):+.1f}, "
      f"G-B 中位={np.median((B[...,1]-B[...,2])[fb]):+.1f}")
print(f"原图 zone外环带: R-G 中位={np.median((r-g)[dout>14]):+.1f}, "
      f"G-B 中位={np.median((g-b)[dout>14]):+.1f}")
