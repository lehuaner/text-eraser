"""对比武器 vs 座驾: mask 占比, box 占总图比例, 跑 erase_text 看结果."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from textpatch.text_select import to_rgb_uint8, detect_text_mask
from textpatch.eraser import erase_text

for name in ['needExtractAndPatch.png', 'needExtractAndPatch2.png']:
    p = ROOT / "data" / name
    rgb = to_rgb_uint8(Image.open(p))
    H, W = rgb.shape[:2]
    total = H*W
    print(f"\n=== {name} ({W}x{H}, total={total}) ===")

    # 跑默认 erase
    r, m, meta = erase_text(rgb, return_mask=True)
    print(f"  boxes: {meta['boxes']}")
    print(f"  mask_pix: {meta['mask_pix']}  ({meta['mask_pix']/total*100:.1f}% of total)")
    if meta.get("mask_filled_pix"):
        print(f"  mask_filled_pix: {meta['mask_filled_pix']}  ({meta['mask_filled_pix']/total*100:.1f}% of total)")

    # box 占总图比例
    for b in meta['boxes']:
        box_area = (b['x1']-b['x0']) * (b['y1']-b['y0'])
        print(f"    box {b}: area={box_area}  ratio={box_area/total*100:.1f}%")

    # 保存 mask (4x) 和 result (4x) 和 orig (4x)
    out_dir = ROOT / "data" / "diag_root"
    out_dir.mkdir(exist_ok=True)
    stem = name.replace('.png','')
    Image.fromarray(rgb).resize((W*4, H*4), Image.NEAREST).save(out_dir / f"{stem}_orig.png")
    Image.fromarray(m).resize((W*4, H*4), Image.NEAREST).save(out_dir / f"{stem}_mask.png")
    Image.fromarray(r).resize((W*4, H*4), Image.NEAREST).save(out_dir / f"{stem}_result.png")

    # 计算 "mask 覆盖完整度": 字符 bbox 内有多少像素本该被 mask 覆盖
    if meta['boxes']:
        bx0 = min(b['x0'] for b in meta['boxes'])
        by0 = min(b['y0'] for b in meta['boxes'])
        bx1 = max(b['x1'] for b in meta['boxes'])
        by1 = max(b['y1'] for b in meta['boxes'])
        # 看 box 内 result 是否有"白字残留"
        # 假设 背景是 gray (~100-160), 白字 > 200, 红字 < 100
        lum_orig = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0]
        lum_res = cv2.cvtColor(r, cv2.COLOR_RGB2LAB)[..., 0]
        # 原图 box 内白字像素 (>200)
        white_orig = (lum_orig[by0:by1, bx0:bx1] > 200).sum()
        # 结果 box 内仍亮的像素
        white_res = (lum_res[by0:by1, bx0:bx1] > 200).sum()
        # 结果 box 内仍红的像素 (<80)
        red_res = (lum_res[by0:by1, bx0:bx1] < 80).sum()
        # 结果 box 内有"灰+亮"等过渡 (140-200)
        light_res = ((lum_res[by0:by1, bx0:bx1] >= 140) & (lum_res[by0:by1, bx0:bx1] <= 200)).sum()
        box_pix = (bx1-bx0)*(by1-by0)
        print(f"  box = ({bx0},{by0},{bx1},{by1}) size={(bx1-bx0)}x{(by1-by0)}")
        print(f"  orig box 内 白色像素 (>200) = {white_orig} ({white_orig/box_pix*100:.1f}% of box)")
        print(f"  res  box 内 白色残留 (>200) = {white_res} ({white_res/box_pix*100:.1f}% of box)  ← 没擦干净的'白边'")
        print(f"  res  box 内 红色残留 (<80)  = {red_res} ({red_res/box_pix*100:.1f}% of box)")
        print(f"  res  box 内 过渡像素(140-200) = {light_res} ({light_res/box_pix*100:.1f}% of box)")