"""前端默认参数下的最终回归:
1. 668 顶部小蒙版点应清零(吸收距离门);
2. 668 文字图层完整性: mask 对两横区/文字 bbox 的覆盖提升;
3. 四图 mask/残迹(前端默认参数, 与历史口径可比);
4. 文字图层对比图。
"""
import sys
import numpy as np
import cv2
from PIL import Image

ROOT = "D:/Code/Project/Python/TextPatch"
sys.path.insert(0, ROOT)
from textpatch.eraser import erase_text
from textpatch.text_select import detect_text_mask, _deglow_full_green_v2

def load(tag):
    return np.array(Image.open(f"{ROOT}/data/_glowcheck/{tag}.png").convert("RGB"))

rgb = load("668")
# 前端默认参数
res, m, meta = erase_text(
    rgb, deglow_scheme="v2", glow_mode="auto", deglow_mask_soft=0.0,
    edge=1, deglow_strength=1, fill_white=True, fill_max_dist=12,
    deglow_zone_ratio=0.6, deglow_zone_expand=10, deglow_protect_px=1,
    return_mask=True, tint_fill=True)
mp = meta["mask_pre_edge"]

# 1) 顶部小点
n, lab, stats, _ = cv2.connectedComponentsWithStats((mp > 0).astype(np.uint8), 8)
comps = sorted(range(1, n), key=lambda i: -stats[i, 4])
print("668 mask_pre_edge 连通块(前端默认参数):")
for i in comps[:6]:
    x, y, w, h, a = stats[i]
    print(f"  (x={x},y={y},w={w},h={h},area={a})")
top = (mp > 0); top[110:, :] = False
print(f"顶部小块(y<110) px: {int(top.sum())}  [修复前 26px]")

# 2) 文字图层覆盖
clean = meta.get("deglow_img")
gclean = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.float32)
hole = np.zeros(rgb.shape[:2], bool); hole[163:208, 138:205] = True
stroke = hole & (gclean > 100)
cov = int(((mp > 0) & stroke).sum())
print(f"两横区 mask 覆盖: {cov}/{int(stroke.sum())}px ({cov/max(stroke.sum(),1)*100:.0f}%)  [修复前 87%]")

# 文字图层对比图
mf = cv2.dilate(mp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
tl = np.zeros((*mp.shape, 4), np.uint8)
mb = mf > 0
tl[mb, :3] = clean[mb]; tl[mb, 3] = 255
bgimg = np.full((*mp.shape, 3), 70, np.uint8); bgimg[::8, ::8] = 60
a = tl[..., 3:4].astype(np.float32) / 255
comp = (tl[..., :3] * a + bgimg * (1 - a)).astype(np.uint8)
z = 4
Image.fromarray(cv2.resize(np.hstack([clean, comp]), None, fx=z, fy=z,
                           interpolation=cv2.INTER_NEAREST)
                ).save(f"{ROOT}/data/_glowcheck/_xin_textlayer_after.png")

# 3) 四图回归(前端默认参数)
print(f"\n{'tag':>5} {'mask_pix':>9} {'sec':>6} {'resid_px':>9}")
for tag in ["178", "556", "635", "668"]:
    im = load(tag)
    r2, m2, meta2 = erase_text(
        im, deglow_scheme="v2", glow_mode="auto", deglow_mask_soft=0.0,
        edge=1, deglow_strength=1, fill_white=True, fill_max_dist=12,
        deglow_zone_ratio=0.6, deglow_zone_expand=10, deglow_protect_px=1,
        return_mask=True, tint_fill=True)
    gres = cv2.cvtColor(r2, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gorg = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    tm, _ = detect_text_mask(im, method="ml", tint_fill=False,
                             max_area_ratio=0.40, q_off=55,
                             fill_white=True, fill_max_dist=12)
    roi = np.zeros(im.shape[:2], bool)
    if tm.any():
        ys, xs = np.nonzero(tm)
        roi[max(0, ys.min()-30):ys.max()+30, max(0, xs.min()-30):xs.max()+30] = True
    resid = int(((gres > 130) & roi & (gorg < 110)).sum())
    print(f"{tag:>5} {meta2['mask_pix']:>9} {meta2['inpaint_seconds']:>6} {resid:>9}")
print("\nsaved _xin_textlayer_after.png")
