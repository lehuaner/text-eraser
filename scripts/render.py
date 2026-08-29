"""跑完整流水线，输出对照图：原图/蒙版/结果"""
import sys
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from textpatch.eraser import erase_text

img = np.asarray(Image.open(ROOT / "data/needExtractAndPatch.png").convert("RGB"), dtype=np.uint8)
result, mask, meta = erase_text(img, return_mask=True)
print(f"meta: {meta}")

out_dir = ROOT / "data" / "result"
out_dir.mkdir(parents=True, exist_ok=True)

# 原图
Image.fromarray(img).save(out_dir / "01_orig.png")
# 蒙版可视化 (在原图上叠红色)
overlay = img.copy()
m_bool = mask > 0
overlay[m_bool] = (img[m_bool].astype(np.int32) * 0.35 + np.array([255, 60, 60]) * 0.65).clip(0, 255).astype(np.uint8)
Image.fromarray(overlay).save(out_dir / "02_mask_overlay.png")
Image.fromarray(mask).save(out_dir / "02_mask.png")
# 结果
Image.fromarray(result).save(out_dir / "03_erased.png")

# 对照 (1:1 拼接)
H, W = img.shape[:2]
gap = 12
combined = np.zeros((H * 3 + gap * 2, W, 3), np.uint8)
combined[:H] = img
combined[H + gap:2*H + gap] = overlay
combined[2*H + 2*gap:] = result
Image.fromarray(combined).save(out_dir / "00_compare.png")
print(f"results in {out_dir}")
