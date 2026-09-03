"""深挖 (x≈148,y≈171) 37px 白块的蒙版丢失原因:
- 它在 DBNet 框内吗?
- tmask / tm_clean 各盖了多少?
- 它与最近 mask 的距离(方案B 6轮生长够不够)?
- 该区域像素级亮度剖面。
"""
import sys
import numpy as np
import cv2
from PIL import Image

ROOT = "D:/Code/Project/Python/TextEraser"
sys.path.insert(0, ROOT)
from text_eraser.text_select import detect_text_mask, _detect_text_mask_classic, _deglow_full_green_v2, _fill_bright_near_mask

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

X0, Y0, X1, Y1 = 138, 160, 170, 195  # 白块邻域
sl = (slice(Y0, Y1), slice(X0, X1))

gorig = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
gclean = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.float32)
r_, g_, b_ = rgb[...,0].astype(np.int16), rgb[...,1].astype(np.int16), rgb[...,2].astype(np.int16)

hole = np.zeros(rgb.shape[:2], bool); hole[171:178, 148:157] = True
print("== 白块(148..156,171..177)统计 ==")
print("orig gray mean/max:", gorig[hole].mean().round(0), gorig[hole].max())
print("orig green mean:", (g_ - np.maximum(r_, b_))[hole].mean().round(0))
print("in tmask:", int((tmask[hole] > 0).sum()), "/", hole.sum(),
      " in tm_clean:", int((tm_clean[hole] > 0).sum()),
      " in brightB:", int((bright[hole] > 0).sum()))

# DBNet 框定位(用 detect_text 的框)
from text_eraser.text_select import detect_text
boxes = detect_text(rgb, strength=1.0, method="ml", max_area_ratio=0.40,
                    max_box_ratio=0.40, work_max=1280, max_side=1280, min_area=30)
hit = [(bx, by, bx2, by2) for (bx, by, bx2, by2) in
       [(bb["x0"], bb["y0"], bb["x1"], bb["y1"]) for bb in boxes]
       if bx <= 152 <= bx2 and by <= 174 <= by2]
print("白块落入 DBNet 框:", hit if hit else "无(不在任何框内!)")

# 距最近 mask(brightB) 的距离
dist = cv2.distanceTransform((bright == 0).astype(np.uint8), cv2.DIST_L2, 3)
print("白块内到最近mask距离: min/mean/max =",
      dist[hole].min().round(1), dist[hole].mean().round(1), dist[hole].max().round(1))

# 沿 x=150..152 竖线 y=160..195 的亮度剖面(orig vs clean), 看 AA 断带
print("\n== 竖剖面 x=151, y=160..194: (orig_gray, clean_gray, tmask, tm_clean, brightB) ==")
for y in range(160, 195):
    print(f"y={y:3d} orig={gorig[y,151]:5.0f} clean={gclean[y,151]:5.0f} "
          f"green={(g_-np.maximum(r_,b_))[y,151]:4d} "
          f"t={int(tmask[y,151]>0)} c={int(tm_clean[y,151]>0)} B={int(bright[y,151]>0)}")

# _detect_text_mask_classic 在原图上白块框内的 Otsu 情况: 找包含白块的框重跑分割
mask_classic, _ = _detect_text_mask_classic(rgb, boxes=boxes, strength=1.0,
                                            min_area=30, q_off=55.0, upscale=True)
print("\nclassic蒙版盖白块:", int((mask_classic[hole] > 0).sum()), "/", hole.sum())

# 可视化: 邻域放大 10x —— orig / clean / tmask∪tm_clean(bright=方案B后)
z = 10
crop_o = rgb[Y0:Y1, X0:X1].copy()
crop_o[hole[Y0:Y1, X0:X1]] = [255, 0, 0]
tiles = [crop_o, clean[Y0:Y1, X0:X1].copy()]
mm = np.stack([(union > 0)*255, (closed > 0)*255, (bright > 0)*255], -1).astype(np.uint8)
mm[Y0:Y1, X0:X1][hole[Y0:Y1, X0:X1]] = [255, 0, 0]
big = cv2.resize(np.hstack([crop_o] + [clean[Y0:Y1, X0:X1]] +
                           [mm[..., i][Y0:Y1, X0:X1].astype(np.uint8) for i in range(3)]),
                 None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
Image.fromarray(big).save(f"{ROOT}/data/_glowcheck/_xin_hole_zoom.png")
print("saved _xin_hole_zoom.png (orig | clean | union | closed | brightB, 白块标红)")
