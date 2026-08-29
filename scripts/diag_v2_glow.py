"""验证 v2 放宽后的去发光: 残绿是否清理干净 + 发光区是否覆盖到边缘浅光。"""
import os
import sys
sys.path.insert(0, "D:/Code/Project/Python/TextPatch")
import numpy as np
from PIL import Image
import cv2

from core.text_select import detect_text_mask, _deglow_full_green_v2
from core.eraser import erase_text

ROOT = "D:/Code/Project/Python/TextPatch"
DIAG = os.path.join(ROOT, "data/_diag"); os.makedirs(DIAG, exist_ok=True)


def load_rgb(tag):
    return np.array(Image.open(os.path.join(ROOT, "data/_glowcheck", f"{tag}.png")).convert("RGB"))


def save(im, name):
    Image.fromarray(im.astype(np.uint8)).save(os.path.join(DIAG, name))


def greenness(im):
    r = im[..., 0].astype(np.int16); g = im[..., 1].astype(np.int16); b = im[..., 2].astype(np.int16)
    return g - np.maximum(r, b)


for tag in ["556", "668", "635", "178"]:
    rgb = load_rgb(tag)
    tmask, _ = detect_text_mask(rgb, method="ml", tint_fill=False, max_area_ratio=0.40,
                                q_off=55, fill_white=True, fill_max_dist=12)
    # 新参数: zone_expand=24, strength=1.15, g_lo=60 (放宽后)
    clean, core, dbg = _deglow_full_green_v2(rgb, tmask, strength=1.15,
                                            zone_ratio=0.6, zone_expand=24, debug=True)
    zone = dbg["zone"]
    zg = greenness(rgb)[zone]; zg_after = greenness(clean)[zone]
    res_green_orig = int((zg > 8).sum())     # 去发光前发光区残绿像素
    res_green_after = int((zg_after > 8).sum())  # 去发光后残绿像素
    print(f"=== {tag} ===")
    print(f"  发光区 zone 像素: {int(zone.sum())}")
    print(f"  发光区内 残绿(g-max>8): 去前={res_green_orig}  去后={res_green_after}  "
          f"清除率={100*(1-res_green_after/max(res_green_orig,1)):.0f}%")
    print(f"  去后发光区平均绿度: {float(zg_after.mean()):.1f}  最大绿度: {int(zg_after.max())}")

    # 残绿残留可视化: 去发光图上把仍偏绿(>8)的像素标红, 看是否还有边缘浅绿漏网
    resid = (greenness(clean) > 8) & (greenness(clean) > 0)
    ov = clean.copy().astype(np.float32)
    ov[resid] = np.array([255, 60, 60], np.float32)
    save(ov.astype(np.uint8), f"{tag}_F_deglow+残绿标红.png")
    save(clean, f"{tag}_A_deglow.png")
    # 去发光 | 最终结果
    res, m, meta = erase_text(rgb, deglow_scheme="v2", edge=1, deglow_strength=1.15,
                              fill_white=True, fill_max_dist=12, return_mask=True, tint_fill=True)
    save(np.hstack([clean, res]), f"{tag}_E_deglow_result.png")
    print(f"  最终: mask_pix={meta.get('mask_pix')} result中心={res[rgb.shape[0]//2, rgb.shape[1]//2].tolist()}")
    print()
print("DONE")
