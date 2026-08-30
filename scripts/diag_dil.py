"""
对比 0/1/2/3/4px 膨胀 — 看看到底几像素才不糊
"""
import os, sys
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_eraser.text_select import detect_text_mask
from text_eraser.patch_fill import inpaint as pm_inpaint

IMG = r"D:\Code\Project\Python\TextEraser\data\needExtractAndPatch.png"
OUT = r"D:\Code\Project\Python\TextEraser\data\dryrun_out"
os.makedirs(OUT, exist_ok=True)

rgb = np.array(Image.open(IMG).convert("RGB"))
mask_orig, _ = detect_text_mask(rgb, method="ml", q_off=70, max_area_ratio=0.40)

def k(p): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p*2+1, p*2+1))

results = []
for d in [0, 1, 2, 3, 4]:
    m = cv2.dilate(mask_orig, k(d)) if d > 0 else mask_orig
    sm = 255 - m
    res = pm_inpaint(rgb, m, sample_mask=sm)
    # 评估
    mask_eval = mask_orig
    me = mask_eval > 0
    if not me.any(): continue
    dil_o = cv2.dilate(me.astype(np.uint8), k(12))
    ring = (dil_o > 0) & (~me)
    cur = res[me]; nbr = res[ring]
    ghost = ((res[..., 0] > 170) & (res[..., 1] > 170) & (res[..., 2] > 170) & me).sum()
    added_outside = ((m > 0) & (mask_orig == 0)).sum() // 255
    print(f"  {d}px dil | fill mean={cur.mean():.1f} ring={nbr.mean():.1f} Δ={abs(cur.mean()-nbr.mean()):.1f} | ghost_in_orig_mask={ghost} | added_outside_text={added_outside}px")
    results.append((d, res, m))

# 拼成 5 行 上下叠, 标号
zoom = 3
H, W = rgb.shape[:2]
ys, xs = np.where(mask_orig > 0)
y0, y1 = max(0, ys.min()-12), min(H, ys.max()+12)
x0, x1 = max(0, xs.min()-12), min(W, xs.max()+12)
crop_orig = rgb[y0:y1, x0:x1]
cw, ch = (x1-x0)*zoom, (y1-y0)*zoom
o_big = cv2.resize(crop_orig, (cw, ch), interpolation=cv2.INTER_NEAREST)

gap = 6
n = len(results)
canvas = np.full((ch*n + gap*(n+1), cw, 3), 30, dtype=np.uint8)
for i, (d, res, _) in enumerate(results):
    rc = res[y0:y1, x0:x1]
    r_big = cv2.resize(rc, (cw, ch), interpolation=cv2.INTER_NEAREST)
    y = gap + i*(ch+gap)
    canvas[y:y+ch] = r_big

from PIL import Image, ImageDraw, ImageFont
img = Image.fromarray(canvas)
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("consola.ttf", 12)
except:
    font = ImageFont.load_default()
for i, (d, _, _) in enumerate(results):
    y = gap + i*(ch+gap) - 2
    draw.text((4, y), f"{d}px dil", fill=(220,220,220), font=font)
img.save(os.path.join(OUT, "_zoom_5way.png"))
print(f"\n✓ {os.path.join(OUT, '_zoom_5way.png')}")