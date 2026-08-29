"""
离线 dry-run：图片 → 字检 → patch_fill，输出原图/蒙版/结果三图。
用法：
    python scripts/dryrun.py \
        [--method classic|ml] [--q_off 50] data/needExtractAndPatch.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# 把项目根加进 sys.path，让 core/ 可直接 import
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from textpatch.text_select import detect_text_mask  # noqa: E402
from textpatch.patch_fill import inpaint            # noqa: E402


def run(input_path: Path, out_dir: Path, method: str = "classic", q_off: float = 50.0):
    rgb = np.asarray(Image.open(input_path).convert("RGB"), dtype=np.uint8)
    H, W = rgb.shape[:2]
    print(f"[INPUT] {input_path}  {W}x{H}")

    t0 = time.time()
    # 调高 max_area_ratio：DBNet 在中文 + 字间距小时会输出单一大概率图（粘连为一框），
    # 默认 0.05 太严会被丢。0.4 让"整块文字行"能过。
    mask, boxes = detect_text_mask(rgb, method=method, q_off=q_off,
                                   max_area_ratio=0.40)
    t1 = time.time()
    print(f"[DETECT] method={method}  boxes={len(boxes)}  mask_pixels={int(mask.sum() // 255)}  {t1-t0:.2f}s")
    for i, b in enumerate(boxes):
        print(f"    box[{i}] x0={b['x0']} y0={b['y0']} x1={b['x1']} y1={b['y1']}")

    if not mask.any():
        print("[WARN] 没检测到文字像素")
        return

    t2 = time.time()
    result = inpaint(rgb, mask)
    t3 = time.time()
    print(f"[FILL] {t3-t2:.2f}s  result shape={result.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    # 原图
    Image.fromarray(rgb).save(out_dir / f"{stem}_1_orig.png")
    # 蒙版可视化（红底叠在原图上）
    overlay = rgb.copy()
    overlay[mask > 0] = (overlay[mask > 0].astype(np.int32) * 0.4 + np.array([255, 0, 0], dtype=np.int32) * 0.6).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(out_dir / f"{stem}_2_mask_overlay.png")
    Image.fromarray(mask).save(out_dir / f"{stem}_2_mask.png")
    # 结果
    Image.fromarray(result).save(out_dir / f"{stem}_3_filled.png")
    print(f"[OUT] {out_dir}/{stem}_*.png  total={time.time()-t0:.2f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "dryrun_out")
    ap.add_argument("--method", choices=["classic", "ml"], default="classic")
    ap.add_argument("--q_off", type=float, default=50.0)
    args = ap.parse_args()
    run(args.input, args.out, args.method, args.q_off)


if __name__ == "__main__":
    main()
