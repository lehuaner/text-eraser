import sys
sys.path.insert(0, "D:/Code/Project/Python/TextPatch")
import numpy as np
from PIL import Image
import cv2
from text_eraser.text_select import detect_text_mask, _deglow_full_green_v2

ROOT = "D:/Code/Project/Python/TextPatch"

def load(t):
    return np.array(Image.open(f"{ROOT}/data/_glowcheck/{t}.png").convert("RGB"))

def gn(im):
    r = im[..., 0].astype(np.int16); g = im[..., 1].astype(np.int16); b = im[..., 2].astype(np.int16)
    return g - np.maximum(r, b)

for t in ["556", "668", "635", "178"]:
    rgb = load(t)
    H, W, _ = rgb.shape
    tm, _ = detect_text_mask(rgb, method="ml", tint_fill=False, max_area_ratio=0.40,
                             q_off=55, fill_white=True, fill_max_dist=12)
    clean, _, dbg = _deglow_full_green_v2(rgb, tm, strength=1.15, zone_ratio=0.6,
                                          zone_expand=24, debug=True)
    budget = int(H * W * 0.6)
    zone = dbg["zone"]
    ga = gn(rgb); gz = gn(clean)
    halo = zone & ~dbg["text_stroke"]
    print(f"[{t}] zone={int(zone.sum())} / budget={budget} ratio={int(zone.sum())/budget:.2f}")
    print(f"    残绿(g-max>1) 去前={int((ga>1).sum())} 去后全图={int((gz>1).sum())} "
          f"去后光晕区={int((gz[halo]>1).sum())} 最大={int(gz.max())}")
    mr, _ = detect_text_mask(rgb, method="ml", tint_fill=True, max_area_ratio=0.40,
                             q_off=55, fill_white=True, fill_max_dist=12)
    mc, boxes = detect_text_mask(clean, method="ml", tint_fill=True, max_area_ratio=0.40,
                                 q_off=55, fill_white=True, fill_max_dist=12)
    union = (mr > 0) | (mc > 0)
    print(f"    文字mask raw={int(mr.sum())} clean={int(mc.sum())} union={int(union.sum())} "
          f"raw_only={int(((mr>0)&~(mc>0)).sum())} clean_only={int(((mc>0)&~(mr>0)).sum())}")
