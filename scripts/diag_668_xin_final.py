"""补齐: clean 图上白块的颜色数值 + 方案B门限逐项核对 + 邻域放大图。"""
import sys
import numpy as np
import cv2
from PIL import Image

ROOT = "D:/Code/Project/Python/TextEraser"
sys.path.insert(0, ROOT)
from text_eraser.text_select import detect_text_mask, _deglow_full_green_v2, _fill_bright_near_mask

rgb = np.array(Image.open(f"{ROOT}/data/_glowcheck/668.png").convert("RGB"))
kw = dict(method="ml", q_off=55.0, max_area_ratio=0.40, max_box_ratio=0.40,
          max_side=1280, fill_white=True, fill_max_dist=12)
tmask, _ = detect_text_mask(rgb, tint_fill=False, **kw)
clean, core = _deglow_full_green_v2(
    rgb, tmask, strength=1.15, alpha_core=0.65,
    zone_ratio=0.6, zone_expand=24, protect_px=1, deglow_chroma_keep=False)
tm_clean, _ = detect_text_mask(clean, tint_fill=True, **kw)
union = ((tmask > 0) | (tm_clean > 0)).astype(np.uint8) * 255
closed = cv2.morphologyEx(union, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
bright = _fill_bright_near_mask(clean, closed)

hole = np.zeros(rgb.shape[:2], bool); hole[171:178, 148:157] = True
X0, Y0, X1, Y1 = 138, 158, 172, 198

cr, cg, cb = clean[...,0].astype(np.int16), clean[...,1].astype(np.int16), clean[...,2].astype(np.int16)
cmin = np.minimum(np.minimum(cr, cg), cb)
cgreen = cg - np.maximum(cr, cb)
gclean = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.float32)
outside = bright == 0
bg = float(np.percentile(gclean[outside], 25))
print(f"clean 白块统计: R={cr[hole].mean():.0f} G={cg[hole].mean():.0f} B={cb[hole].mean():.0f} "
      f"gray={gclean[hole].mean():.0f} min_rgb={cmin[hole].mean():.0f} green={cgreen[hole].mean():.0f}")
print(f"方案B门限核对(clean): bg(25%分位)={bg:.0f} → 需 gray>{bg+24:.0f}: "
      f"{int((gclean[hole]>bg+24).sum())}/{hole.sum()} | min_rgb>=118: "
      f"{int((cmin[hole]>=118).sum())}/{hole.sum()} | green<26: "
      f"{int((cgreen[hole]<26).sum())}/{hole.sum()}")
dist = cv2.distanceTransform((bright == 0).astype(np.uint8), cv2.DIST_L2, 3)
print(f"到最近 mask 距离: {dist[hole].min():.1f}~{dist[hole].max():.1f}px (方案B最多长6轮=6px)")

# zone 覆盖吗? 用 v2 的 zone 逻辑近似: 绿种子生长区。这里用 core(zone核心)输出看
print("core(去发光核心区)盖白块:", int((core[hole] > 0).sum()), "/", hole.sum())

# ---- 邻域放大图: orig | clean | union | closed | brightB (白块红框标出) ----
z = 9
def pannel(img):
    p = img[Y0:Y1, X0:X1].copy()
    cv2.rectangle(p, (148 - X0, 171 - Y0), (156 - X0, 177 - Y0), (255, 0, 0), 1)
    return p
tiles = [pannel(rgb), pannel(clean)]
for m in (union, closed, bright):
    t = np.stack([(m[Y0:Y1, X0:X1] > 0).astype(np.uint8) * 200] * 3, -1).copy()
    cv2.rectangle(t, (148 - X0, 171 - Y0), (156 - X0, 177 - Y0), (255, 0, 0), 1)
    tiles.append(t)
big = cv2.resize(np.hstack(tiles), None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
Image.fromarray(big).save(f"{ROOT}/data/_glowcheck/_xin_hole_zoom.png")
print("saved _xin_hole_zoom.png")

# 白块周围一圈的 mask 距离场示意: bright 膨胀 6px 后到没到
for rounds in (6, 10, 14):
    grown = (bright > 0).astype(np.uint8)
    for _ in range(rounds):
        grown = cv2.dilate(grown, np.ones((3, 3), np.uint8))
    print(f"brightB 从mask生长{rounds:2d}轮 覆盖白块: {int((grown[hole]>0).sum())}/{hole.sum()}")
