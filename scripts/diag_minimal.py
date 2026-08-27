"""
A/B/C 诊断: 找到最干净的最简管线
A. tight mask + patch_fill (no dilation, no TELEA, no color match)
B. tight mask + patch_fill + sample_mask=outside (只加取样区限制)
C. + 1px dilation
D. + 1px dilation + sample_mask
E. = A + TELEA 兜底 (3px, 仅补小洞, 不会涂大区域)
F. = B + TELEA 兜底
"""
import os, sys, time
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.text_select import detect_text_mask
from core.patch_fill import inpaint as pm_inpaint
from core.eraser import force_color_match

IMG = r"D:\Code\Project\Python\TextPatch\data\needExtractAndPatch.png"
OUT = r"D:\Code\Project\Python\TextPatch\data\dryrun_out"
os.makedirs(OUT, exist_ok=True)

rgb = np.array(Image.open(IMG).convert("RGB"))
mask_orig, _ = detect_text_mask(rgb, method="ml", q_off=70, max_area_ratio=0.40)
print(f"text mask: {mask_orig.sum()//255} px, img {rgb.shape}")

def save(name, img):
    p = os.path.join(OUT, name)
    Image.fromarray(img).save(p)
    return p

def k1():
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

variants = []

# A. tight mask, plain patch_fill
m = mask_orig
r = pm_inpaint(rgb, m)
variants.append(("A_tight+pm", r))

# B. tight mask + sample_mask outside
sm = 255 - m
r = pm_inpaint(rgb, m, sample_mask=sm)
variants.append(("B_tight+pm+sm_outside", r))

# C. + 1px dilation, plain patch_fill
m = cv2.dilate(mask_orig, k1())
r = pm_inpaint(rgb, m)
variants.append(("C_1pxdil+pm", r))

# D. + 1px dilation + sample_mask outside
sm = 255 - m
r = pm_inpaint(rgb, m, sample_mask=sm)
variants.append(("D_1pxdil+pm+sm_outside", r))

# E. A + tiny TELEA fallback
m = mask_orig
r1 = pm_inpaint(rgb, m)
# 只对 A 仍残留的 1px 边界做兜底 (用 EROSION 后的 mask 标记)
hole = (r1.sum(-1) == 0).astype(np.uint8)  # 简化: 实际更复杂
# 简化: 直接对 r1 整体做轻量 TELEA, 只对 r1==原图未填的小洞有意义
r = cv2.inpaint(r1, m, 3, cv2.INPAINT_TELEA)
variants.append(("E_tight+pm+telea3", r))

# F. B + tiny TELEA fallback
sm = 255 - m
r1 = pm_inpaint(rgb, m, sample_mask=sm)
r = cv2.inpaint(r1, m, 3, cv2.INPAINT_TELEA)
variants.append(("F_tight+pm+sm+telea3", r))

# 评估: 文字残留 (mask 内) 的均值与背景差 + 标准差 vs 周边
def evaluate(name, res, mask_used):
    m = (mask_used > 0)
    if not m.any():
        return
    # 周围 ring (mask 外的 4-12px)
    dil_outer = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    dil_inner = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    ring = (dil_outer > 0) & (~m)
    cur = res[m]
    nbr = res[ring]
    print(f"  {name:35s} | fill: mean={cur.mean():.1f} std={cur.std():.1f} | ring: mean={nbr.mean():.1f} std={nbr.std():.1f} | |Δmean|={abs(cur.mean()-nbr.mean()):.1f}")
    save(f"_diag_{name}.png", res)

# 拼成 3x2 大图 方便对比
imgs = []
labels = []
for name, r in variants:
    print(f"\n== {name} ==")
    evaluate(name, r, mask_orig)
    # 标注图
    annotated = r.copy()
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1]-1, annotated.shape[0]-1), (255,255,0), 1)
    imgs.append(annotated)
    labels.append(name)

# 拼 3x2
from PIL import Image, ImageDraw, ImageFont
COLS = 2
ROWS = (len(imgs) + COLS - 1) // COLS
W, H = rgb.shape[1], rgb.shape[0]
GAP = 24
LBL = 18
canvas = Image.new("RGB", (COLS * W + (COLS + 1) * GAP, ROWS * (H + LBL) + (ROWS + 1) * GAP), (24, 24, 28))
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("consola.ttf", 13)
except:
    font = ImageFont.load_default()

# 原图放左上
canvas.paste(Image.fromarray(rgb), (GAP, GAP + LBL))
draw.text((GAP, GAP), "ORIG", fill=(220, 220, 220), font=font)

for i, (img, label) in enumerate(zip(imgs, labels)):
    r, c = i // COLS, i % COLS
    x = GAP + c * (W + GAP)
    y = GAP + r * (H + LBL + GAP) + LBL
    canvas.paste(Image.fromarray(img), (x, y))
    draw.text((x, y - LBL), label, fill=(220, 220, 220), font=font)

canvas.save(os.path.join(OUT, "_diag_compare.png"))
print(f"\n✓ {os.path.join(OUT, '_diag_compare.png')}")
