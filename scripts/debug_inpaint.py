"""调试: 直接看 cv2.inpaint 的中间步骤"""
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from text_eraser.text_select import detect_text_mask

img = np.asarray(Image.open(ROOT / "data/needExtractAndPatch.png").convert("RGB"), dtype=np.uint8)
H, W = img.shape[:2]
mask, _ = detect_text_mask(img, method="ml", max_area_ratio=0.40, q_off=70)

# 1) 直接对原图用 cv2.inpaint
r0 = cv2.inpaint(img, mask, 5, cv2.INPAINT_NS)
still = ((r0 > 180).all(axis=-1) & (mask > 0)).sum()
print(f"直接 cv2.inpaint(NS,5) 仍在 mask 区白像素 = {int(still)}")

# 看一下 mask 内像素和 r0 mask 内像素 RGB 分布
print("原图 mask 内像素均值:", img[mask > 0].mean(axis=0))
print("r0 mask 内像素均值:", r0[mask > 0].mean(axis=0))

# 把 img 的灰度直方图(全图 vs mask 区域)输出
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
hist_full, _ = np.histogram(gray, bins=20, range=(0, 255))
hist_mask, _ = np.histogram(gray[mask > 0], bins=20, range=(0, 255))
print("\n灰度直方图 (20 bins):")
print(f"  {'bin':<8} {'full':>6} {'mask':>6}")
for i in range(20):
    print(f"  {i*12.75:5.1f}-{(i+1)*12.75:5.1f}  {hist_full[i]:>6} {hist_mask[i]:>6}")

# 查 mask 像素中"白色像素数"
white = ((img > 180).all(axis=-1) & (mask > 0)).sum()
print(f"\nmask 内白色像素 = {int(white)}")
Image.fromarray(r0).save(ROOT / "data/dryrun_out/_debug_direct_inpaint.png")
