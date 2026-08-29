"""量化 556 去发光结果的 zone 边界色差:
1. 边界内外带的整体色差(R−G / G−B / 亮度 / hue);
2. 沿穿过边界的采样线画剖面(亮度与 R−G 随距离变化), 看过渡带宽窄与跳变幅度;
3. 找色差最大的边界段。
"""
import sys
import numpy as np
import cv2
from PIL import Image

ROOT = "D:/Code/Project/Python/TextPatch"
sys.path.insert(0, ROOT)
from core.eraser import erase_text

rgb = np.array(Image.open(f"{ROOT}/data/_glowcheck/556.png").convert("RGB"))
res, m, meta = erase_text(
    rgb, deglow_scheme="v2", glow_mode="auto", deglow_mask_soft=0.0,
    edge=1, deglow_strength=1, fill_white=True, fill_max_dist=12,
    deglow_zone_ratio=0.6, deglow_zone_expand=10, deglow_protect_px=1,
    return_mask=True, tint_fill=True)
clean = meta["deglow_img"]
zone = meta["glow_zone"] > 0
mask = m > 0

H, W = zone.shape
K = 12   # 边界带半宽
inner = zone & ~cv2.erode(zone.astype(np.uint8), np.ones((2*K+1, 2*K+1), np.uint8)).astype(bool)
outer = (cv2.dilate(zone.astype(np.uint8), np.ones((2*K+1, 2*K+1), np.uint8)) > 0) & ~zone
# 排除文字蒙版(文字本身不参与边界色差)
inner &= ~mask
outer &= ~mask

def chan(img):
    r = img[...,0].astype(np.float32); g = img[...,1].astype(np.float32); b = img[...,2].astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    return gray, r-g, g-b, hsv[...,0], hsv[...,1]

print("== 边界带整体色差(clean, 剔除文字) ==")
gq, rgq, gbq, hq, sq = chan(clean)
for name, sel in [("内侧带(zone内12px)", inner), ("外侧带(zone外12px)", outer)]:
    print(f"  {name}: n={int(sel.sum())}  亮度={np.median(gq[sel]):.1f}  R-G={np.median(rgq[sel]):.1f}  "
          f"G-B={np.median(gbq[sel]):.1f}  hue={np.median(hq[sel]):.0f}  sat={np.median(sq[sel]):.0f}")

# ---- 逐边界像素的局部色差(内 6px vs 外 6px 的局部中位) ----
inn6 = zone & ~cv2.erode(zone.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
out6 = (cv2.dilate(zone.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0) & ~zone
inn6 &= ~mask; out6 &= ~mask
gb_in = cv2.blur(rgq, (9, 9)); gb_out = cv2.blur(rgq, (9, 9))
lum_in = cv2.blur(gq, (9, 9)); lum_out = cv2.blur(gq, (9, 9))
edge_px = zone & ~cv2.erode(zone.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
edge_px &= ~mask
d_rg = (gb_in - gb_out)[edge_px]
d_lum = (lum_in - lum_out)[edge_px]
print(f"\n== 边界像素局部色差(内侧9px均值 − 外侧9px均值) ==")
print(f"  边界像素 n={int(edge_px.sum())}")
print(f"  Δ(R−G): p50={np.percentile(d_rg,50):.1f} p90={np.percentile(d_rg,90):.1f} p99={np.percentile(d_rg,99):.1f} max={d_rg.max():.1f}")
print(f"  Δ亮度 : p50={np.percentile(d_lum,50):.1f} p90={np.percentile(d_lum,90):.1f} p99={np.percentile(d_lum,99):.1f} max={d_lum.max():.1f}")
big = np.abs(d_rg) > 8
print(f"  |Δ(R−G)|>8 的边界像素: {int(big.sum())} ({big.sum()/max(edge_px.sum(),1)*100:.0f}%)")

# ---- 采样线剖面: 穿过边界的三条线 ----
print("\n== 采样线剖面(亮度 | R−G, 每格2px) ==")
ys, xs = np.nonzero(zone)
cy, cx = int(np.median(ys)), int(np.median(xs))
for name, (sy, sx), (ey, ex) in [
        ("西向东(y=cy)", (cy, 5), (cy, W-6)),
        ("南向北(x=cy)", (H-6, cx), (5, cx))]:
    yy = np.linspace(sy, ey, 60).astype(int)
    xx = np.linspace(sx, ex, 60).astype(int)
    in_mask = zone[yy, xx]
    # 找过界点
    trans = [i for i in range(1, 60) if in_mask[i] != in_mask[i-1]]
    prof_l = gq[yy, xx]; prof_r = rgq[yy, xx]
    seg = f"{name}: 过界@{trans[:3]}"
    print(f"  {seg}")
    for i in range(0, 60, 3):
        z = "内" if in_mask[i] else "外"
        print(f"    t{i:02d} ({yy[i]:3d},{xx[i]:3d}) {z}: 亮度={prof_l[i]:5.1f}  R−G={prof_r[i]:+5.1f}")

# ---- 可视化: 边界色差热图 ----
vis = clean.copy()
vis[edge_px & (np.abs(d_rg) > 8) if False else edge_px] = [255, 140, 0]  # 边界橙
vis[edge_px & (np.abs(gb_in - gb_out)[edge_px if False else slice(None)] > 8) if False else edge_px] = [255, 140, 0]
big = cv2.resize(vis, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
Image.fromarray(big).save(f"{ROOT}/data/_glowcheck/_556_seam_edges.png")
print("\nsaved _556_seam_edges.png (边界橙色)")
