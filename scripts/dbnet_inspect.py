"""直接调用 detect_text_ml 看返回值"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import ml_text_select as _ml

img = np.asarray(Image.open(ROOT / "data/needExtractAndPatch.png").convert("RGB"), dtype=np.uint8)
H, W = img.shape[:2]

# detect_text_ml 默认参数
boxes = _ml.detect_text_ml(img, strength=1.0, min_area=30, max_area_ratio=0.05,
                            max_box_ratio=0.20, box_threshold=0.30, max_side=960, pad=3)
print(f"boxes={len(boxes)}")
for b in boxes:
    print(f"  {b}")

# 看 mask 的版本
mask, boxes2 = _ml.detect_text_mask_ml(img, strength=1.0, min_area=30, max_area_ratio=0.05,
                                        max_side=960, pad=3, mask_threshold=0.30,
                                        mask_max_side=1600)
print(f"mask_pixels={int(mask.sum() // 255)}, boxes={len(boxes2)}")
for b in boxes2:
    print(f"  {b}")

# 调高阈值试试
boxes3 = _ml.detect_text_ml(img, strength=0.5, min_area=10, max_area_ratio=0.5,
                            max_box_ratio=0.5, box_threshold=0.15, max_side=960, pad=3)
print(f"\nlower-thr boxes={len(boxes3)}")
for b in boxes3:
    print(f"  {b}")
