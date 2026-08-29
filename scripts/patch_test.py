"""独立测试 patch_fill：手画一个独立方块 mask，确认它能否把那块像素彻底换掉。"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from textpatch.patch_fill import inpaint

img = np.asarray(Image.open(ROOT / "data/needExtractAndPatch.png").convert("RGB"), dtype=np.uint8)
H, W = img.shape[:2]
mask = np.zeros((H, W), np.uint8)
mask[20:60, 20:60] = 255  # 一个独立方块 test
result = inpaint(img, mask)
Image.fromarray(result).save(ROOT / "data/dryrun_out/_patch_test_isolated.png")
print("isolated test saved")

# 测试：用 DBNet 实际产出的 mask（已知正确）
sys.path.insert(0, str(ROOT))
from textpatch.text_select import detect_text_mask
mask2, _ = detect_text_mask(img, method="ml", max_area_ratio=0.40, q_off=70)
print(f"mask2 white pixels = {int(mask2.sum()//255)}")
result2 = inpaint(img, mask2)
Image.fromarray(result2).save(ROOT / "data/dryrun_out/_patch_test_dbnet_mask.png")
print("DBNet mask test saved")
