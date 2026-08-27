"""
干净的对比图: 单行 3 列 [原图 | OLD blob | NEW clean]
"""
import os, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

IMG = r"D:\Code\Project\Python\TextPatch\data\needExtractAndPatch.png"
OLD = r"D:\Code\Project\Python\TextPatch\data\server_result.png"
NEW = r"D:\Code\Project\Python\TextPatch\data\server_result_v2.png"
OUT = r"D:\Code\Project\Python\TextPatch\data\result"

rgb = np.array(Image.open(IMG).convert("RGB"))
old = np.array(Image.open(OLD).convert("RGB"))
new = np.array(Image.open(NEW).convert("RGB"))

H, W = rgb.shape[:2]
gap = 16

# 单行 3 列 (拼文字区域周围的横向拼接, 每个图顶部加白条做标签)
canvas = np.full((H + gap*2 + 32, W*3 + gap*4, 3), 30, dtype=np.uint8)
canvas[gap+32:gap+32+H, gap:gap+W] = rgb
canvas[gap+32:gap+32+H, gap*2+W:gap*2+W*2] = old
canvas[gap+32:gap+32+H, gap*3+W*2:gap*3+W*3] = new

img = Image.fromarray(canvas)
draw = ImageDraw.Draw(img)
try:
    font_lg = ImageFont.truetype("consola.ttf", 17)
except:
    font_lg = ImageFont.load_default()
draw.text((gap, 4), "ORIGINAL (with text '武器')", fill=(220, 220, 220), font=font_lg)
draw.text((gap*2+W, 4), "OLD: 4px dil + TELEA + color match", fill=(220, 220, 220), font=font_lg)
draw.text((gap*3+W*2, 4), "NEW: 2px dil + sample_mask only", fill=(220, 220, 220), font=font_lg)

img.save(os.path.join(OUT, "compare_3col.png"))
print(f"✓ {os.path.join(OUT, 'compare_3col.png')}")