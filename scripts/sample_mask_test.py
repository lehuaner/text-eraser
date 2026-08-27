"""测试 sample_mask：用 (整图 - mask) 作为采样区强制 patch_fill 只从非文字区域取样"""
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.text_select import detect_text_mask
from core.patch_fill import inpaint

img = np.asarray(Image.open(ROOT / "data/needExtractAndPatch.png").convert("RGB"), dtype=np.uint8)
H, W = img.shape[:2]
mask, _ = detect_text_mask(img, method="ml", max_area_ratio=0.40, q_off=70)

print(f"input {W}x{H}, mask_pix={int(mask.sum()//255)}")
sample_mask = (255 - mask).astype(np.uint8)  # 整图 - mask

# 1) 不加 sample_mask
r0 = inpaint(img, mask)
still0 = ((r0 > 180).all(axis=-1) & (mask > 0)).sum()
mean0 = r0[mask > 0].mean(axis=0)
print(f"无 sample_mask: 仍白={int(still0)}  mask区均值={mean0.round(1).tolist()}")

# 2) 加 sample_mask = 整图 - mask
r1 = inpaint(img, mask, sample_mask=sample_mask)
still1 = ((r1 > 180).all(axis=-1) & (mask > 0)).sum()
mean1 = r1[mask > 0].mean(axis=0)
print(f"加 sample_mask: 仍白={int(still1)}  mask区均值={mean1.round(1).tolist()}")
Image.fromarray(r1).save(ROOT / "data/dryrun_out/_with_sample_mask.png")

# 3) sample_mask + 二次 inpaint 清核心
r2 = cv2.inpaint(r1, mask, 5, cv2.INPAINT_NS)
mean2 = r2[mask > 0].mean(axis=0)
print(f"sample_mask + NS: mask区均值={mean2.round(1).tolist()}")
Image.fromarray(r2).save(ROOT / "data/dryrun_out/_with_sample_mask_ns.png")

# 4) 跟周围 20px 的统计均值比较
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
m = mask.astype(np.uint8) * 255
ring = cv2.dilate(m, kernel) - m - m
ring = ring > 0
ring_mean = img[ring].mean(axis=0)
print(f"\nmask 周围 20px ring 均值 = {ring_mean.round(1).tolist()}  像素数={int(ring.sum())}")
print(f"img 全图均值 = {img.reshape(-1, 3).mean(0).round(1).tolist()}")
