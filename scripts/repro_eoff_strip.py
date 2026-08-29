"""最终三栏对比：目标 eoff / 当前代码+默认 / 当前代码+关后加旋钮"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.eraser import erase_text

ORIG = Path("D:/Code/Project/Python/ExtractRole/data/needExtractAndPatch.png")
TARGET = ROOT / "data" / "diag_root" / "needExtractAndPatch_result_eoff.png"
OUT = ROOT / "data" / "repro_eoff"
OUT.mkdir(parents=True, exist_ok=True)

rgb = np.asarray(Image.open(ORIG).convert("RGB"), dtype=np.uint8)
H, W = rgb.shape[:2]
print(f"原图 {W}x{H}")

# A: 当前代码 + 只关 edge_aware（其它全部走默认=后加旋钮全开）
rA, _, _ = erase_text(rgb, edge_aware=False, return_mask=True)
# B: 当前代码 + 关掉全部后加旋钮
rB, _, _ = erase_text(rgb, edge_aware=False, glow_mode="off",
                     tint_fill=False, edge_extend=0, return_mask=True)
# 目标: 下采样到原尺寸
rT = np.asarray(Image.open(TARGET).convert("RGB").resize((W, H), Image.LANCZOS), np.uint8)

# 4x 放大拼接
SC = 3
pad = 8
W4, H4 = W * SC, H * SC
imgT = Image.fromarray(rT).resize((W4, H4), Image.NEAREST)
imgA = Image.fromarray(rA).resize((W4, H4), Image.NEAREST)
imgB = Image.fromarray(rB).resize((W4, H4), Image.NEAREST)

cols = [imgT, imgA, imgB]
labels = [
    "目标 eoff\n(当年生成)",
    "当前[A] edge_aware=False\n(其它全默认=glow/tint/extend 开)",
    "当前[B] 关掉全部\n后加旋钮",
]
W_canvas = W4 * 3 + pad * 4
H_canvas = H4 + 60
canvas = Image.new("RGB", (W_canvas, H_canvas), (250, 250, 250))
d = ImageDraw.Draw(canvas)
for i, (im, lab) in enumerate(zip(cols, labels)):
    x = pad + i * (W4 + pad)
    canvas.paste(im, (x, 50))
    # 简单标签
    d.text((x + 8, 8), lab, fill=(0, 0, 0))
canvas.save(OUT / "STRIP_目标_vs_当前.png")

def diff(a, b):
    return float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())

print(f"\n像素均值差(满分255,越小越像):")
print(f"  [A]  edge_aware=False (其它默认): {diff(rA, rT):.3f}")
print(f"  [B]  +关掉后加旋钮:             {diff(rB, rT):.3f}")

# 算填充区的局部差(用 mask)
# 简单用差值图高亮差异区
gray_T = cv2.cvtColor(rT, cv2.COLOR_RGB2GRAY)
gray_A = cv2.cvtColor(rA, cv2.COLOR_RGB2GRAY)
gray_B = cv2.cvtColor(rB, cv2.COLOR_RGB2GRAY)
# mask 近似: 目标与原图的差>阈值
orig_g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
mask_approx = (np.abs(gray_T.astype(np.int32) - orig_g) > 20)
print(f"  文字区域(mask近似)像素数: {int(mask_approx.sum())}")
print(f"  文字区内 [A] 与目标 像素差: {float(np.abs(rA[mask_approx].astype(np.int32) - rT[mask_approx].astype(np.int32)).mean()):.3f}")
print(f"  文字区内 [B] 与目标 像素差: {float(np.abs(rB[mask_approx].astype(np.int32) - rT[mask_approx].astype(np.int32)).mean()):.3f}")
print(f"\n拼接图: {OUT / 'STRIP_目标_vs_当前.png'}")
