"""1787767429309(武器, 灰石背景) 文字蒙版逐级诊断:
「器」周围出现非字形蒙版 —— 逐级导出 v2 链路每一步的蒙版,
定位是哪一步(原图检测/去发光图检测/并集/方案B/亮核吸收)把
字形之外的像素锁进蒙版。

输出: data/_glowcheck/_wq_stage*.png(4x 放大, 红色=该步蒙版)
"""
import sys
import numpy as np
import cv2
from PIL import Image

ROOT = "D:/Code/Project/Python/TextPatch"
sys.path.insert(0, ROOT)
from text_eraser.text_select import (detect_text_mask, _deglow_full_green_v2,
                              _fill_bright_near_mask, _absorb_zone_bright_core)

rgb = np.array(Image.open(f"{ROOT}/data/history/1787767429309/orig.bin").convert("RGB"))
H, W = rgb.shape[:2]
print(f"image {W}x{H}")

# ---- meta.json 参数(前端默认) ----
kw = dict(method="ml", q_off=55.0, max_area_ratio=0.4, max_box_ratio=0.4, max_side=960)

# 1) 原图蒙版(保护种子, tint=False)
tmask, boxes = detect_text_mask(rgb, tint_fill=False, fill_white=True, fill_max_dist=12, **kw)
print(f"[1] tmask(orig)        pix={int((tmask>0).sum())}  boxes={boxes}")

# 2) 去发光 v2 → clean + zone
clean, _, zone = _deglow_full_green_v2(
    rgb, tmask, strength=1.15, zone_ratio=0.6, zone_expand=10,
    protect_px=1, deglow_chroma_keep=False, return_zone=True)
zone_b = zone > 0
print(f"[2] zone               pix={int(zone_b.sum())} ({zone_b.sum()/zone_b.size*100:.1f}%)  发光判定={'触发' if zone_b.any() else '未触发'}")

# 3) 去发光图蒙版(tint=True)
tm_clean, boxes_c = detect_text_mask(clean, tint_fill=True, fill_white=True, fill_max_dist=12, **kw)
print(f"[3] tm_clean           pix={int((tm_clean>0).sum())}")

# 4) 并集 + 闭运算
mask = ((tmask > 0) | (tm_clean > 0)).astype(np.uint8) * 255
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
print(f"[4] union+close        pix={int((mask>0).sum())}")

# 5) 方案B
mask_b = _fill_bright_near_mask(clean, mask)
print(f"[5] +方案B             pix={int((mask_b>0).sum())}  (+{int((mask_b>0).sum()-(mask>0).sum())})")

# 6) 亮核吸收
mask_c = _absorb_zone_bright_core(clean, rgb, mask_b, zone, min_rgb_lo=100)
print(f"[6] +亮核吸收          pix={int((mask_c>0).sum())}  (+{int((mask_c>0).sum()-(mask_b>0).sum())})")

# ---- 器字区域逐级对比 ----
# 器 box≈(155,39)-(212,94), 取外扩 12px 的窗口
x0, y0, x1, y1 = 143, 27, 224, 106
names = ["1_tmask", "3_tmclean", "4_union", "5_schemeB", "6_absorb"]
masks = [tmask, tm_clean, mask, mask_b, mask_c]
base = (mask_c > 0)
for nm, mk in zip(names, masks):
    mb = mk > 0
    only = mb & ~np.roll(mb, 0)  # placeholder
    print(f"  器窗口 {nm}: pix={int(mb[y0:y1, x0:x1].sum())}")

# 差集可视化: 方案B新增 / 吸收新增
add_b = (mask_b > 0) & ~(mask > 0)
add_c = (mask_c > 0) & ~(mask_b > 0)
print(f"[diff] 方案B新增 pix={int(add_b.sum())}  吸收新增 pix={int(add_c.sum())}")
if add_b.any():
    ys, xs = np.nonzero(add_b)
    print(f"       方案B新增范围 x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]")
if add_c.any():
    ys, xs = np.nonzero(add_c)
    print(f"       吸收新增范围 x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]")

# ---- 可视化 ----
def vis_save(img, mk, name, add=None):
    v = img.copy()
    m = mk > 0
    v[m] = [v[m][..., i] * 0.35 + np.array([190, 30, 30])[i] * 0.65 for i in range(3)] \
        if False else np.array([190, 30, 30])
    if add is not None and add.any():
        v[add] = [255, 200, 0]  # 新增=黄
    big = cv2.resize(v, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    Image.fromarray(big).save(f"{ROOT}/data/_glowcheck/_wq_{name}.png")

vis_save(rgb, tmask, "stage1_tmask")
vis_save(clean, tm_clean, "stage3_tmclean")
vis_save(clean, mask, "stage4_union")
vis_save(clean, mask_b, "stage5_schemeB", add_b)
vis_save(clean, mask_c, "stage6_absorb", add_c)
# zone 可视化
vz = rgb.copy(); vz[zone_b] = [30, 160, 255]
Image.fromarray(cv2.resize(vz, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)).save(
    f"{ROOT}/data/_glowcheck/_wq_zone.png")
print("saved _wq_stage*.png / _wq_zone.png")
