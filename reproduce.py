"""一键复现 TextPatch 的「武器」擦除效果。

用法（必须用项目根目录的 Python，且 cwd 切到项目根）：
    cd D:\Code\Project\Python\TextPatch
    C:/Users/乐幻/AppData/Local/Programs/Python/Python313/python.exe reproduce.py

输出：
    data/result/repro_result.png   擦除结果
    data/result/repro_mask.png     DBNet+Otsu 逐像素字形 mask
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.eraser import erase_text

SRC = ROOT / "data" / "needExtractAndPatch.png"
OUT_DIR = ROOT / "data" / "result"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not SRC.is_file():
    raise SystemExit(f"源图缺失: {SRC}")

rgb = np.asarray(Image.open(SRC).convert("RGB"), dtype=np.uint8)

# ---- 唯三关键参数（见 core/eraser.py）----
# 1) method="ml" + max_area_ratio=0.40  ->  否则"武"+"器"粘连大块(>5%)被 DBNet 丢，mask 变空
# 2) q_off=70  ->  mask 最贴字形
# 3) mask_pad=2  ->  mask 2px 椭圆膨胀，吃掉抗锯齿边缘(0px→ghost572, 2px→0)
result, mask, meta = erase_text(
    rgb,
    mask_pad=2,          # 关键：2px 椭圆膨胀
    q_off=70.0,          # 关键：最高紧密度，mask 最贴字形
    max_area_ratio=0.40, # 关键：给"武+器"粘连大块放行
    return_mask=True,
)

Image.fromarray(result).save(OUT_DIR / "repro_result.png")
Image.fromarray(mask).save(OUT_DIR / "repro_mask.png")

print("mask_pix        =", meta["mask_pix"])
print("mask_filled_pix =", meta["mask_filled_pix"])
print("inpaint_seconds =", meta["inpaint_seconds"])
print("boxes           =", meta["boxes"])
print("saved:", OUT_DIR / "repro_result.png")
print("saved:", OUT_DIR / "repro_mask.png")
