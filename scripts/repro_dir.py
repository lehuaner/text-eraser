"""复现 direction 参数导致内部错误的问题。"""
from __future__ import annotations
import sys, traceback
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from text_eraser.eraser import erase_text

IMGS = {
    "武器": Path("D:/Code/Project/Python/ExtractRole/data/needExtractAndPatch.png"),
    "座驾": ROOT / "data" / "diag_root" / "needExtractAndPatch2_orig.png",
}

for name, p in IMGS.items():
    print(f"\n===== {name} ({p.name}) =====")
    rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    try:
        res = erase_text(rgb, direction=60.0, glow_mode="off")
        print(f"  [OK] direction=60 成功, 结果 shape={res[0].shape}")
    except Exception as e:
        print(f"  [FAIL] direction=60 抛异常: {type(e).__name__}: {e}")
        traceback.print_exc()
    # 对照：无 direction 应正常
    try:
        res = erase_text(rgb, direction=None, glow_mode="off")
        print(f"  [OK] direction=None 成功")
    except Exception as e:
        print(f"  [FAIL] direction=None 也异常(基线就坏): {type(e).__name__}: {e}")
