import sys
sys.path.insert(0, "D:/Code/Project/Python/TextPatch")
import numpy as np
from PIL import Image
import cv2
from core.text_select import detect_text_mask, _deglow_full_green_v2

ROOT = "D:/Code/Project/Python/TextPatch"

def load(t):
    return np.array(Image.open(f"{ROOT}/data/_glowcheck/{t}.png").convert("RGB"))

def gn(im):
    r = im[..., 0].astype(np.int16); g = im[..., 1].astype(np.int16); b = im[..., 2].astype(np.int16)
    return g - np.maximum(r, b)

def save(im, name):
    Image.fromarray(im.astype(np.uint8) if im.dtype != np.uint8 else im).save(f"{ROOT}/data/_diag/{name}")

for t in ["556", "668", "635", "178"]:
    rgb = load(t)
    tm, _ = detect_text_mask(rgb, method="ml", tint_fill=False, max_area_ratio=0.40,
                             q_off=55, fill_white=True, fill_max_dist=12)
    clean, _, dbg = _deglow_full_green_v2(rgb, tm, strength=1.15, zone_ratio=0.6,
                                          zone_expand=24, debug=True)
    zone = dbg["zone"]; ts = dbg["text_stroke"]
    gz = gn(clean)
    # 残绿分类
    resid = gz > 2
    in_text = resid & ts
    out_zone = resid & ~zone
    # 标注图：原图上 红=zone外残绿(真漏盖)  蓝=白字内残绿(无伤大雅)
    ov = rgb.copy()
    ov[out_zone] = [255, 40, 40]
    ov[in_text] = [60, 130, 255]
    save(ov, f"{t}_G_残绿定位.png")

    # 文字蒙版对比：clean(tint=True) vs raw(tint=False)
    mc, _ = detect_text_mask(clean, method="ml", tint_fill=True, max_area_ratio=0.40,
                             q_off=55, fill_white=True, fill_max_dist=12)
    mr0, _ = detect_text_mask(rgb, method="ml", tint_fill=False, max_area_ratio=0.40,
                              q_off=55, fill_white=True, fill_max_dist=12)
    union = (mc > 0) | (mr0 > 0)
    # 闭运算补断裂
    k3 = np.ones((3, 3), np.uint8)
    union_c = cv2.morphologyEx(union.astype(np.uint8), cv2.MORPH_CLOSE, k3) > 0
    print(f"[{t}] 残绿: zone外(真漏盖)={int(out_zone.sum())} 白字内={int(in_text.sum())}")
    print(f"    文字mask clean={int(mc.sum())} raw0(tintF)={int(mr0.sum())} "
          f"union={int(union.sum())} union闭运算={int(union_c.sum())} "
          f"raw0_only={int(((mr0>0)&~(mc>0)).sum())}")
