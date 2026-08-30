"""诊断 668「亲」部两横在去发光时被抹掉的原因:
两横区域(y≈165~205, x≈140~200)像素的 原图gray/绿度/min_rgb 分布,
它们是否满足 text_stroke 保护门(min_rgb>120 & 绿度<40), 及到保护圈的距离。
"""
import sys
import numpy as np
import cv2
from PIL import Image

ROOT = "D:/Code/Project/Python/TextEraser"
sys.path.insert(0, ROOT)
from text_eraser.text_select import detect_text_mask, _deglow_full_green_v2

rgb = np.array(Image.open(f"{ROOT}/data/_glowcheck/668.png").convert("RGB"))
kw = dict(method="ml", q_off=55.0, max_area_ratio=0.40, max_box_ratio=0.40,
          max_side=1280, fill_white=True, fill_max_dist=12)
tmask, _ = detect_text_mask(rgb, tint_fill=False, **kw)
clean, _, zone = _deglow_full_green_v2(
    rgb, tmask, strength=1.15, alpha_core=0.65, zone_ratio=0.6,
    zone_expand=24, protect_px=1, deglow_chroma_keep=False, return_zone=True)

r, g, b = rgb[..., 0].astype(np.int16), rgb[..., 1].astype(np.int16), rgb[..., 2].astype(np.int16)
green = g - np.maximum(r, b)
min_rgb = np.minimum(np.minimum(r, g), b)
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
gclean = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.float32)

# 复现 v2 内部的 text_stroke 与 protect2
white_floor, g40 = 120, 40
text_stroke = (min_rgb > white_floor) & (green < g40)
# zone(近似: 直接用返回的 clean 无法拿 zone? 我们有 return_zone)
k3 = np.ones((3, 3), np.uint8)
protect2 = cv2.dilate(text_stroke.astype(np.uint8), k3, iterations=1) > 0

# 「两横」区域: 原图上 y=165..205, x=138..205 中亮于局部背景的部分(人工定义文字感)
X0, X1, Y0, Y1 = 138, 205, 163, 208
sub = np.zeros(rgb.shape[:2], bool); sub[Y0:Y1, X0:X1] = True
bg_lum = 86.0
strokes = sub & (gray > bg_lum + 15) & (zone > 0)   # zone内、亮于背景15+ 的文字感像素
print("两横感像素:", int(strokes.sum()))
prot = strokes & protect2
print("  其中在保护圈内(text_stroke+1px):", int(prot.sum()), f"({prot.sum()/max(strokes.sum(),1)*100:.0f}%)")
lost = strokes & ~protect2
print("  被重建区(圈外):", int(lost.sum()))
if lost.sum():
    print("  圈外像素特征: gray p10/50/90 =", np.percentile(gray[lost], [10, 50, 90]).round(0),
          " green p50/90 =", np.percentile(green[lost], [50, 90]).round(0),
          " min_rgb p10/50 =", np.percentile(min_rgb[lost], [10, 50]).round(0))
    print("  clean 上灰度 p10/50/90 =", np.percentile(gclean[lost], [10, 50, 90]).round(0),
          "(≈85即被重建抹平)")
dist = cv2.distanceTransform((~protect2).astype(np.uint8), cv2.DIST_L2, 3)
if lost.sum():
    print("  圈外像素到保护圈距离 p50/90/max =", np.percentile(dist[lost], [50, 90, 100]).round(1))
# 绿度门直方图: 圈外像素绿度分布
if lost.sum():
    hist, edges = np.histogram(green[lost], bins=[0, 20, 40, 50, 60, 70, 90])
    print("  圈外绿度直方图 [0,20,40,50,60,70,90):", hist)
    hist2, _ = np.histogram(min_rgb[lost], bins=[0, 90, 105, 120, 150, 255])
    print("  圈外min_rgb直方图 [0,90,105,120,150,255):", hist2)

# 可视化: 两横区域放大(原图|clean|text_stroke标红|被抹区标红)
z = 6
o = rgb[Y0:Y1, X0:X1].copy()
c = clean[Y0:Y1, X0:X1].copy()
t = rgb[Y0:Y1, X0:X1].copy(); t[text_stroke[Y0:Y1, X0:X1]] = [255, 0, 0]
l = rgb[Y0:Y1, X0:X1].copy(); l[lost[Y0:Y1, X0:X1]] = [255, 0, 0]
big = cv2.resize(np.hstack([o, c, t, l]), None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
Image.fromarray(big).save(f"{ROOT}/data/_glowcheck/_xin_strokes_diag.png")
print("saved _xin_strokes_diag.png (原图|clean|text_stroke红|被抹区红)")
