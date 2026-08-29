"""分析 eoff 图做法 + 验证当前代码能否像素级复现。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.eraser import erase_text

# --- 目标图（4x 放大版） ---
EOFF_4X = ROOT / "data" / "diag_root" / "needExtractAndPatch_result_eoff.png"
ORIG = Path("D:/Code/Project/Python/ExtractRole/data/needExtractAndPatch.png")
OUT = ROOT / "data" / "repro_eoff"
OUT.mkdir(parents=True, exist_ok=True)

rgb = np.asarray(Image.open(ORIG).convert("RGB"), dtype=np.uint8)
H, W = rgb.shape[:2]
print(f"原图 {W}x{H}")

# 1) 当年生成 eoff 的调用：唯一显式参数 edge_aware=False，其他全部默认
r_a, m_a, meta_a = erase_text(rgb, edge_aware=False, return_mask=True)
print(f"  [A] edge_aware=False (其它默认)  mask_pix={meta_a['mask_pix']}  elapsed={meta_a['inpaint_seconds']}s")

# 2) 关闭所有后加旋钮，尝试更接近当年"无后加参数"的状态
r_b, m_b, meta_b = erase_text(rgb, edge_aware=False, glow_mode="off",
                              tint_fill=False, edge_extend=0, return_mask=True)
print(f"  [B] +glow_mode=off +tint_fill=off +edge_extend=0  mask_pix={meta_b['mask_pix']}  elapsed={meta_b['inpaint_seconds']}s")

# 3) 目标图下采样到原尺寸对比
target = np.asarray(Image.open(EOFF_4X).convert("RGB").resize((W, H), Image.LANCZOS), np.uint8)

def diff(a, b):
    return float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())

print(f"\n像素差异(均值绝对差,越小越像):")
print(f"  当前[A] vs 目标 eoff:    {diff(r_a, target):.3f}")
print(f"  当前[B] vs 目标 eoff:    {diff(r_b, target):.3f}")

# 保存 A 和 B 的 4x 放大版做肉眼看
for tag, r in [("A_now", r_a), ("B_off_addons", r_b)]:
    Image.fromarray(r).resize((W*4, H*4), Image.NEAREST).save(OUT / f"武器_now_{tag}.png")
Image.fromarray(target).save(OUT / "武器_target_eoff_1x.png")
print(f"\n4x 放大对比图已存: {OUT}")
