"""定位 668 最终结果里的白色残迹来源:
1. 在结果图上找"亮于背景"的像素(残迹);
2. 回查原图同位置的颜色/亮度, 与填充 mask 的覆盖关系;
3. 判断: 蒙版漏覆盖(原图就亮, mask没盖) vs 填充取样污染(mask盖了, 填充后仍白)。
"""
import sys
import numpy as np
import cv2
from PIL import Image

ROOT = "D:/Code/Project/Python/TextPatch"
sys.path.insert(0, ROOT)
from textpatch.text_select import detect_text_mask, _deglow_full_green_v2, _fill_bright_near_mask

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
mask_filled = cv2.dilate(bright, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

res = np.array(Image.open(f"{ROOT}/data/_glowcheck/_xin_full_result.png").convert("RGB"))
H, W = rgb.shape[:2]

# 结果图残迹: 明显亮于深色背景(背景~88)的白点
gres = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY).astype(np.float32)
# 只看"新"字邻域(下色块内), 排除上方浅色块分界
ys, xs = np.nonzero(union)
y0, y1 = ys.min() - 35, ys.max() + 35
roi = np.zeros((H, W), bool); roi[y0:y1, max(0, xs.min()-35):xs.max()+35] = True
# 残迹 = 结果图上亮度 > 130 且在深色块内 (背景86~99)
resid = (gres > 130) & roi
print("resid px in result:", int(resid.sum()))

# 背景参考: 深色块干净处 (x=60..100, y=200..300)
bg_sample = gres[200:300, 60:100]
print("bg dark mean:", bg_sample.mean().round(1))

# 分类: 这些残迹像素在原图的亮度 / 是否被 mask 盖住
gorig = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
gclean = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.float32)
covered = mask_filled > 0
r_, g_, b_ = rgb[...,0].astype(np.int16), rgb[...,1].astype(np.int16), rgb[...,2].astype(np.int16)
greenish = (g_ - np.maximum(r_, b_))

in_mask = resid & covered
out_mask = resid & ~covered
print(f"resid covered by fill-mask: {int(in_mask.sum())}  (mask盖了但结果仍亮 → 填充/去发光问题)")
print(f"resid NOT covered:          {int(out_mask.sum())}  (蒙版漏)")

for name, sel in [("covered", in_mask), ("not_covered", out_mask)]:
    if sel.sum() == 0:
        continue
    print(f"--- {name}: orig gray p50/p90={np.percentile(gorig[sel],[50,90]).round(0)}, "
          f"clean gray p50={np.percentile(gclean[sel],50).round(0)}, "
          f"orig green p50={np.percentile(greenish[sel],50).round(0)}")
    yy, xx = np.nonzero(sel)
    print(f"    bbox x[{xx.min()},{xx.max()}] y[{yy.min()},{yy.max()}]")

# 可视化: 结果残迹标红 → 叠在 原图 / clean / mask三联图
def triple(sel, out):
    vis = rgb.copy()
    vis[sel] = [255, 0, 0]
    z = 4
    big = cv2.resize(vis[y0:y1, max(0, xs.min()-35):xs.max()+35], None, fx=z, fy=z,
                     interpolation=cv2.INTER_NEAREST)
    Image.fromarray(big).save(f"{ROOT}/data/_glowcheck/_xin_resid_{out}.png")

triple(resid, "on_orig")
triple(out_mask, "notcov_on_orig")

# 残迹连通块统计
n, lab, stats, cent = cv2.connectedComponentsWithStats(resid.astype(np.uint8), 8)
print("resid components:", n - 1)
comps = sorted(range(1, n), key=lambda i: -stats[i, 4])[:12]
for i in comps:
    x, y, w, h, a = stats[i]
    cover_ratio = covered[y:y+h, x:x+w][lab[y:y+h, x:x+w] == i].mean()
    print(f"  comp at (x={x},y={y},w={w},h={h},area={a}) orig_gray_mean="
          f"{gorig[lab == i].mean().round(0)} mask_cover_ratio={cover_ratio:.2f}")
