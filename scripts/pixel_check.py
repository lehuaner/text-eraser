"""逐像素对比：原图 vs DBNet mask vs patch_fill result"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from text_eraser.text_select import detect_text_mask
from text_eraser.patch_fill import inpaint

img = np.asarray(Image.open(ROOT / "data/needExtractAndPatch.png").convert("RGB"), dtype=np.uint8)
H, W = img.shape[:2]
mask, _ = detect_text_mask(img, method="ml", max_area_ratio=0.40, q_off=70)
result = inpaint(img, mask)

# 选中"武"字内一个点 (130, 110)
pts = [(130, 110), (140, 115), (165, 110), (180, 115), (140, 100)]
print(f"{'pts':<10} {'orig RGB':<18} {'mask>0?':<10} {'result RGB':<18}")
for x, y in pts:
    o = img[y, x].tolist(); m = bool(mask[y, x] > 0); r = result[y, x].tolist()
    print(f"({x:3},{y:3})  {str(o):<18}  {m!s:<10}  {str(r):<18}")

# 比较 mask>0 像素的平均值
mask_pix = img[mask > 0]
res_pix = result[mask > 0]
print(f"\nmask 区域 ({mask.sum()//255} 像素):")
print(f"  orig mean RGB = {mask_pix.mean(axis=0).round(1).tolist()}")
print(f"  result mean RGB = {res_pix.mean(axis=0).round(1).tolist()}")

# 看 mask>0 像素的 result **到底有没有被换**
# 一个粗略检测：如果 result 在 mask 区域还有 ≥N 个原 mask_pix 平均色相近的像素，则 mask 内未换
# 改：只看 result >180 的像素（白色），应该 0
white_in_mask = ((result > 180).all(axis=-1) & (mask > 0)).sum()
print(f"result 仍在 mask 区域且全通道 >180 的像素数 = {int(white_in_mask)}")
