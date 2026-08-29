"""诊断 v2 流程：去发光后到底有没有真正去字、为什么文字还看得见。"""
import os
import sys
sys.path.insert(0, "D:/Code/Project/Python/TextPatch")
import numpy as np
from PIL import Image
import cv2

from core.text_select import detect_text_mask, _deglow_full_green_v2
from core.eraser import erase_text

ROOT = "D:/Code/Project/Python/TextPatch"
DIAG = os.path.join(ROOT, "data/_diag")
os.makedirs(DIAG, exist_ok=True)


def load_rgb(tag):
    p = os.path.join(ROOT, "data/_glowcheck", f"{tag}.png")
    return np.array(Image.open(p).convert("RGB"))


def save(im, name):
    Image.fromarray(im.astype(np.uint8)).save(os.path.join(DIAG, name))


def overlay(rgb, mask, color=(255, 60, 60)):
    im = rgb.copy().astype(np.float32)
    m = (mask > 0)
    im[m] = (np.array(color, np.float32) * 0.6 + im[m] * 0.4)
    return im.astype(np.uint8)


for tag in ["556", "668", "635", "178"]:
    rgb = load_rgb(tag)
    H, W, _ = rgb.shape

    # 1) 去发光
    tmask, _ = detect_text_mask(rgb, method="ml", tint_fill=False,
                                max_area_ratio=0.40, q_off=55,
                                fill_white=True, fill_max_dist=12)
    clean, _ = _deglow_full_green_v2(rgb, tmask, strength=1.0,
                                     zone_ratio=0.6, zone_expand=10)

    # 2) 在 clean 上跑「去文字」算法（项目非高亮路径），fill_white=True（默认）
    mask_fw, boxes_fw = detect_text_mask(clean, method="ml", tint_fill=True,
                                         max_area_ratio=0.40, q_off=55,
                                         max_side=960, fill_white=True,
                                         fill_max_dist=12)

    # 3) 对照：fill_white=False 单独检测
    mask_nofw, _ = detect_text_mask(clean, method="ml", tint_fill=True,
                                    max_area_ratio=0.40, q_off=55,
                                    max_side=960, fill_white=False,
                                    fill_max_dist=0)

    print(f"=== {tag} ({W}x{H}) ===")
    print(f"  deglow clean 是否含文字(白字应被保留): "
          f"clean 中心={clean[H//2,W//2].tolist()}")
    print(f"  detect_text_mask(clean, fill_white=True)  mask_pix = {int(mask_fw.sum()//255)}  boxes={len(boxes_fw)}")
    print(f"  detect_text_mask(clean, fill_white=False) mask_pix = {int(mask_nofw.sum()//255)}")

    # 4) 跑完整 v2
    res, m, meta = erase_text(rgb, deglow_scheme="v2", edge=1,
                              deglow_strength=1.0, fill_white=True,
                              fill_max_dist=12, return_mask=True,
                              tint_fill=True)
    print(f"  v2 最终 mask_pix={meta.get('mask_pix')}  fill_pix={meta.get('mask_filled_pix')}  "
          f"result中心={res[H//2,W//2].tolist()}")

    # 存中间图：去发光结果、其上检测到的去字 mask、最终结果
    save(clean, f"{tag}_A_deglow.png")
    save(overlay(clean, mask_fw, (255, 60, 60)), f"{tag}_B_clean+textmask(fw).png")
    save(overlay(clean, mask_nofw, (80, 170, 255)), f"{tag}_C_clean+textmask(nofw).png")
    save(res, f"{tag}_D_result.png")
    # 把去发光结果与最终结果并排，方便一眼看出文字是否还在
    side = np.hstack([clean, res])
    save(side, f"{tag}_E_deglow_result.png")
    print(f"  已存: {tag}_A_deglow / B(fw) / C(nofw) / D_result / E_side")
    print()
print("DONE")
