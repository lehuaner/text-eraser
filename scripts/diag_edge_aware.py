"""验证: 关掉 edge_aware, 直接用 pad=2 蒙版, 看座驾/武器的擦除结果."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.text_select import to_rgb_uint8
from core.eraser import erase_text

for name in ['needExtractAndPatch.png', 'needExtractAndPatch2.png']:
    p = ROOT / "data" / name
    rgb = to_rgb_uint8(Image.open(p))
    H, W = rgb.shape[:2]
    total = H * W

    # 关 edge_aware
    r_off, m_off, meta_off = erase_text(rgb, edge_aware=False, return_mask=True)
    # 开 edge_aware
    r_on,  m_on,  meta_on  = erase_text(rgb, edge_aware=True, return_mask=True)

    # 测白边残留
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0]
    def residue(r, label):
        # 全图范围内
        lum_r = cv2.cvtColor(r, cv2.COLOR_RGB2LAB)[..., 0]
        # 用合并 box
        if meta_off['boxes']:
            bx0 = min(b['x0'] for b in meta_off['boxes'])
            by0 = min(b['y0'] for b in meta_off['boxes'])
            bx1 = max(b['x1'] for b in meta_off['boxes'])
            by1 = max(b['y1'] for b in meta_off['boxes'])
            # 原图 box 内 "亮于 200" 的白像素
            white_orig = (lum[by0:by1, bx0:bx1] > 200).sum()
            white_res = (lum_r[by0:by1, bx0:bx1] > 200).sum()
            red_res = (lum_r[by0:by1, bx0:bx1] < 80).sum()
            box_pix = (bx1-bx0)*(by1-by0)
            print(f"  [{name} {label}] mask={meta_off['mask_pix']} box=(...{bx1-bx0}x{by1-by0}) white_res={white_res} ({white_res/box_pix*100:.1f}%) red_res={red_res} ({red_res/box_pix*100:.1f}%)")
            return white_res, red_res
        return 0, 0

    print(f"\n=== {name} ===")
    w_off, r_off2 = residue(r_off, "edge_aware=OFF")
    w_on,  r_on2  = residue(r_on,  "edge_aware=ON")

    out_dir = ROOT / "data" / "diag_root"
    out_dir.mkdir(exist_ok=True)
    stem = name.replace('.png', '')
    Image.fromarray(r_off).resize((W*4, H*4), Image.NEAREST).save(out_dir / f"{stem}_result_eoff.png")
    Image.fromarray(r_on).resize((W*4, H*4), Image.NEAREST).save(out_dir / f"{stem}_result_eon.png")
    Image.fromarray(m_off).resize((W*4, H*4), Image.NEAREST).save(out_dir / f"{stem}_mask_eoff.png")
    Image.fromarray(m_on).resize((W*4, H*4), Image.NEAREST).save(out_dir / f"{stem}_mask_eon.png")

print("\nsaved")