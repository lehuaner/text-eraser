"""保护圈扩张修复回归:
1. 668「亲」两横: 去发光图上两横区域的保留情况(修复前被抹平到 82~86);
2. 四图最终结果 mask/残迹对比基线(178=1273, 556=5532, 635=1332, 668=10839, resid=0);
3. 668 去发光前后对比图。
"""
import sys
import numpy as np
import cv2
from PIL import Image

ROOT = "D:/Code/Project/Python/TextPatch"
sys.path.insert(0, ROOT)
from text_eraser.eraser import erase_text
from text_eraser.text_select import detect_text_mask, _deglow_full_green_v2

def load(tag):
    return np.array(Image.open(f"{ROOT}/data/_glowcheck/{tag}.png").convert("RGB"))

# ---- 1) 668 两横专项 ----
rgb = load("668")
kw = dict(method="ml", q_off=55.0, max_area_ratio=0.40, max_box_ratio=0.40,
          max_side=1280, fill_white=True, fill_max_dist=12)
tmask, _ = detect_text_mask(rgb, tint_fill=False, **kw)
clean, _, zone = _deglow_full_green_v2(
    rgb, tmask, strength=1.15, alpha_core=0.65, zone_ratio=0.6,
    zone_expand=24, protect_px=1, deglow_chroma_keep=False, return_zone=True)
gc = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.float32)
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
sub = np.zeros(rgb.shape[:2], bool); sub[163:208, 138:205] = True
lost_before = sub & (gray > 101)   # 原图上的笔画感像素(与修复前口径一致)
kept = int((gc[lost_before] > 100).sum())
print(f"668 两横区: 原图笔画感 {int(lost_before.sum())}px, 去发光后保留(>100): {kept}px "
      f"({kept/max(lost_before.sum(),1)*100:.0f}%)  [修复前: 两横 70% 被抹平]")

# ---- 2) 四图全流程回归 ----
print(f"\n{'tag':>5} {'mask_pix':>9} {'sec':>6} {'resid_px':>9}")
for tag in ["178", "556", "635", "668"]:
    im = load(tag)
    res, m, meta = erase_text(
        im, deglow_scheme="v2", glow_mode="auto", deglow_mask_soft=0.0,
        edge=1, deglow_strength=1.15, fill_white=True, fill_max_dist=12,
        deglow_zone_ratio=0.6, deglow_zone_expand=24,
        return_mask=True, tint_fill=True)
    gres = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gorg = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    tm, _ = detect_text_mask(im, method="ml", tint_fill=False,
                             max_area_ratio=0.40, q_off=55,
                             fill_white=True, fill_max_dist=12)
    roi = np.zeros(im.shape[:2], bool)
    if tm.any():
        ys, xs = np.nonzero(tm)
        roi[max(0, ys.min()-30):ys.max()+30, max(0, xs.min()-30):xs.max()+30] = True
    resid = int(((gres > 130) & roi & (gorg < 110)).sum())
    print(f"{tag:>5} {meta['mask_pix']:>9} {meta['inpaint_seconds']:>6} {resid:>9}")

# ---- 3) 668 去发光前后对比图(原图 | clean) ----
z = 2.4
big = cv2.resize(np.hstack([rgb, clean]), None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
Image.fromarray(big).save(f"{ROOT}/data/_glowcheck/_xin_deglow_protected.png")
print("\nsaved _xin_deglow_protected.png (原图 | 修复后去发光图)")
