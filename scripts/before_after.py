"""
最终 before/after 三联图: 原图 | 老方案 (有 blob) | 新方案 (干净)
"""
import os, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

IMG = r"D:\Code\Project\Python\TextEraser\data\needExtractAndPatch.png"
OLD = r"D:\Code\Project\Python\TextEraser\data\server_result.png"     # 老(TELEA+dil4+color)
NEW = r"D:\Code\Project\Python\TextEraser\data\server_result_v2.png"  # 新(2px dil + sample_mask)
OUT = r"D:\Code\Project\Python\TextEraser\data\result"
os.makedirs(OUT, exist_ok=True)

rgb = np.array(Image.open(IMG).convert("RGB"))
old = np.array(Image.open(OLD).convert("RGB"))
new = np.array(Image.open(NEW).convert("RGB"))

# 上半: 原图 | 老方案 | 新方案
gap = 12
H, W = rgb.shape[:2]
canvas = np.full((H * 3 + gap * 4, W, 3), 30, dtype=np.uint8)
canvas[gap:gap+H] = rgb
canvas[gap*2+H:gap*2+H*2] = old
canvas[gap*3+H*2:gap*3+H*3] = new

img = Image.fromarray(canvas)
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("consola.ttf", 18)
    font_small = ImageFont.truetype("consola.ttf", 13)
except:
    font = ImageFont.load_default()
    font_small = font
draw.text((gap, 2), "原图 (有 武器 文字)", fill=(220, 220, 220), font=font)
draw.text((gap, gap+H+2), "OLD: 4px dil + TELEA + 颜色匹配 → 明显 blob", fill=(220, 220, 220), font=font)
draw.text((gap, gap*2+H*2+2), "NEW: 2px dil + sample_mask → 干净, 纹理自然", fill=(220, 220, 220), font=font)

# 中部局部放大三张
ys_, xs_ = np.where(np.array(Image.open(IMG).convert("L")) < 200)  # 简化
# 用 DBNet mask 更准
import sys; sys.path.insert(0, r"D:\Code\Project\Python\TextEraser")
from text_eraser.text_select import detect_text_mask
mask_tight, _ = detect_text_mask(rgb, method="ml", q_off=70, max_area_ratio=0.40)
ys_, xs_ = np.where(mask_tight > 0)
y0, y1 = max(0, ys_.min()-10), min(H, ys_.max()+10)
x0, x1 = max(0, xs_.min()-10), min(W, xs_.max()+10)

zoom = 4
o = cv2.resize(rgb[y0:y1, x0:x1], ((x1-x0)*zoom, (y1-y0)*zoom), interpolation=cv2.INTER_NEAREST)
od = cv2.resize(old[y0:y1, x0:x1], ((x1-x0)*zoom, (y1-y0)*zoom), interpolation=cv2.INTER_NEAREST)
nw = cv2.resize(new[y0:y1, x0:x1], ((x1-x0)*zoom, (y1-y0)*zoom), interpolation=cv2.INTER_NEAREST)

cw, ch = (x1-x0)*zoom, (y1-y0)*zoom
zoom_canvas = np.full((ch*3 + gap*4, cw*3 + gap*4, 3), 30, dtype=np.uint8)
zoom_canvas[gap:gap+ch, gap:gap+cw] = o
zoom_canvas[gap:gap+ch, gap*2+cw:gap*2+cw*2] = od
zoom_canvas[gap:gap+ch, gap*3+cw*2:gap*3+cw*3] = nw
zoom_canvas[gap*2+ch:gap*2+ch*2, gap:gap+cw] = o
zoom_canvas[gap*3+ch*2:gap*3+ch*3, gap:gap+cw] = nw

zoom_img = Image.fromarray(zoom_canvas)
draw2 = ImageDraw.Draw(zoom_img)
try:
    font2 = ImageFont.truetype("consola.ttf", 11)
except:
    font2 = ImageFont.load_default()

# 拼一起 (全图在上, 放大在下)
total_h = canvas.shape[0] + zoom_canvas.shape[0] + gap
total_w = max(canvas.shape[1], zoom_canvas.shape[1])
final = Image.new("RGB", (total_w, total_h), (15, 15, 18))
final.paste(Image.fromarray(canvas), (0, 0))
final.paste(zoom_img, (0, canvas.shape[0] + gap))
final.save(os.path.join(OUT, "BEFORE_AFTER.png"))
print(f"✓ {os.path.join(OUT, 'BEFORE_AFTER.png')}")

# 单图: 新方案干净对比 (原图 | 新方案 mask | 新方案结果)
mask_padded = cv2.dilate(mask_tight, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
overlay = rgb.copy()
overlay[mask_padded > 0] = (rgb[mask_padded > 0].astype(np.int32) * 0.35 + np.array([255, 60, 60]) * 0.65).clip(0, 255).astype(np.uint8)
single = np.full((H * 3 + gap * 4, W, 3), 30, dtype=np.uint8)
single[gap:gap+H] = rgb
single[gap*2+H:gap*2+H*2] = overlay
single[gap*3+H*2:gap*3+H*3] = new
single_img = Image.fromarray(single)
single_draw = ImageDraw.Draw(single_img)
single_draw.text((gap, 2), "原图 (武器)", fill=(220, 220, 220), font=font)
single_draw.text((gap, gap+H+2), "DBNet + 2px dil 蒙版 (红) - 仅覆盖文字+AA边缘", fill=(220, 220, 220), font=font)
single_draw.text((gap, gap*2+H*2+2), "patch_fill 结果 - 完全看不出原文字", fill=(220, 220, 220), font=font)
single_img.save(os.path.join(OUT, "FINAL_compare.png"))
print(f"✓ {os.path.join(OUT, 'FINAL_compare.png')}")