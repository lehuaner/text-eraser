"""组合方案：patch_fill 完之后再 cv2.inpaint 兜底，抹掉笔画核心残留"""
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from textpatch.text_select import detect_text_mask
from textpatch.patch_fill import inpaint

img = np.asarray(Image.open(ROOT / "data/needExtractAndPatch.png").convert("RGB"), dtype=np.uint8)
H, W = img.shape[:2]
print(f"input {W}x{H}")

# 不同 q_off 对比
for q in (30, 50, 70):
    mask, boxes = detect_text_mask(img, method="ml", max_area_ratio=0.40, q_off=q)
    print(f"\nq_off={q}: boxes={len(boxes)} mask_pixels={int(mask.sum()//255)}")
    if not mask.any():
        continue
    r1 = inpaint(img, mask)  # patch_fill
    # 第 2 轮:用原 mask 对 r1 再做一次 NS inpaint 清扫笔画核心
    # 用更大半径(5) + NS 算法, 抹掉任何"残余白线"
    r2 = cv2.inpaint(r1, mask, 5, cv2.INPAINT_NS)
    # 用 TELEA 再扫一遍, 让残余纹理更"自然"(NS 会有方向性拉条)
    r3 = cv2.inpaint(r2, mask, 3, cv2.INPAINT_TELEA)
    # 看 r3 还有多少白色像素
    still_white = ((r3 > 180).all(axis=-1) & (mask > 0)).sum()
    print(f"  r3 (patch_fill + NS(5) + TELEA(3)) 仍在 mask 区白像素 = {int(still_white)}")
    Image.fromarray(r3).save(ROOT / f"data/dryrun_out/_combo_q{q:02d}.png")
