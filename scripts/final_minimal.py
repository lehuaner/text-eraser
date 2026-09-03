"""
最终方案 — 最小管线:
  1. DBNet 检测文字像素 (mask)
  2. 1px 椭圆膨胀 (吃 AA 边缘)
  3. patch_fill(sample_mask=整图 - mask)
  4. 不做 TELEA, 不做颜色匹配

如果还有 ghost, 再做"只在 ghost 像素处" 1px 精修。
"""
import os, sys
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_eraser.text_select import detect_text_mask
from text_eraser.patch_fill import inpaint as pm_inpaint

IMG = r"D:\Code\Project\Python\TextEraser\data\needExtractAndPatch.png"
OUT = r"D:\Code\Project\Python\TextEraser\data\result"
os.makedirs(OUT, exist_ok=True)

rgb = np.array(Image.open(IMG).convert("RGB"))
mask_orig, boxes = detect_text_mask(rgb, method="ml", q_off=70, max_area_ratio=0.40)

# 1px 椭圆膨胀: 仅吃抗锯齿边缘
k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.dilate(mask_orig, k1)
print(f"mask: {mask_orig.sum()//255} → {mask.sum()//255} px (after 1px dil)")

# sample_mask = 文字外全部
sample_mask = (255 - mask).astype(np.uint8)

# patch_fill
res = pm_inpaint(rgb, mask, sample_mask=sample_mask)

# 检测 ghost: 找仍像文字亮度的像素 (RGB 三通道 >180 比例)
gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
ghost = ((res[..., 0] > 170) & (res[..., 1] > 170) & (res[..., 2] > 170)).astype(np.uint8) * 255
ghost_in_mask = cv2.bitwise_and(ghost, mask)
print(f"ghost pixels (still bright in mask): {ghost_in_mask.sum() // 255}")

# 保存
Image.fromarray(res).save(os.path.join(OUT, "01_minimal.png"))
Image.fromarray(cv2.cvtColor(mask_orig, cv2.COLOR_GRAY2RGB)).save(os.path.join(OUT, "02_mask_orig.png"))
Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)).save(os.path.join(OUT, "03_mask_padded.png"))

# 三联对比
gap = 8
W, H = rgb.shape[1], rgb.shape[0]
canvas = np.full((H * 3 + gap * 4, W, 3), 30, dtype=np.uint8)
canvas[gap:gap+H] = rgb
canvas[gap*2+H:gap*2+H*2] = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
canvas[gap*3+H*2:gap*3+H*3] = res
cv2.imwrite(os.path.join(OUT, "00_compare.png"), canvas[:, :, ::-1])  # RGB->BGR for cv2
print(f"✓ {os.path.join(OUT, '00_compare.png')}")

# 局部放大: 仅 mask 区域
ys, xs = np.where(mask)
y0, y1 = max(0, ys.min()-8), min(H, ys.max()+8)
x0, x1 = max(0, xs.min()-8), min(W, xs.max()+8)
orig_crop = rgb[y0:y1, x0:x1]
res_crop = res[y0:y1, x0:x1]

# 拼: [原图 | 结果] 横向
crop_h = orig_crop.shape[0]
crop_w = orig_crop.shape[1]
zoom = 4
o_big = cv2.resize(orig_crop, (crop_w*zoom, crop_h*zoom), interpolation=cv2.INTER_NEAREST)
r_big = cv2.resize(res_crop, (crop_w*zoom, crop_h*zoom), interpolation=cv2.INTER_NEAREST)
gap2 = 4
zoom_canvas = np.full((crop_h*zoom, crop_w*zoom*2 + gap2, 3), 30, dtype=np.uint8)
zoom_canvas[:, :crop_w*zoom] = o_big
zoom_canvas[:, crop_w*zoom+gap2:] = r_big
cv2.imwrite(os.path.join(OUT, "04_zoom_compare.png"), zoom_canvas[:, :, ::-1])
print(f"✓ {os.path.join(OUT, '04_zoom_compare.png')}")