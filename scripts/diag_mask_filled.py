"""看 mask_filled (mask_pad + edge_aware 后) 实际长什么样."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.text_select import to_rgb_uint8
from core.eraser import erase_text, _edge_aware_grow, _ellipse
from core.text_select import detect_text_mask

for name in ['needExtractAndPatch.png', 'needExtractAndPatch2.png']:
    p = ROOT / "data" / name
    rgb = to_rgb_uint8(Image.open(p))
    H, W = rgb.shape[:2]
    total = H * W
    print(f"\n=== {name} ({W}x{H}) ===")

    # 直接拿到 mask (没填充)
    mask, _ = detect_text_mask(rgb, method="ml", q_off=55.0,
                               max_area_ratio=0.40, max_box_ratio=0.40)
    print(f"  mask_pix = {int(mask.sum()/255)} ({int(mask.sum()/255)/total*100:.1f}%)")

    # mask_pad=2 椭圆膨胀
    mask_padded = cv2.dilate(mask, _ellipse(2))
    print(f"  after pad(2): {int(mask_padded.sum()/255)} ({int(mask_padded.sum()/255)/total*100:.1f}%)")

    # edge_aware_grow
    mask_filled = _edge_aware_grow(rgb, mask_padded)
    print(f"  after edge_aware: {int(mask_filled.sum()/255)} ({int(mask_filled.sum()/255)/total*100:.1f}%)")

    # 看用更小 edge_aware 半径会怎样
    def edge_grow_small(rgb, m, p=3):
        if not m.any(): return m
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lum = lab[..., 0].astype(np.float32)
        text_lum = lum[m > 0]
        lo = float(text_lum.min())
        hi = float(text_lum.max())
        bg = float(np.median(lum[m == 0]))
        band_lo = (bg + lo) / 2.0
        band_hi = hi + (hi - lo) * 0.5
        cand = cv2.dilate(m, _ellipse(p))
        keep = (lum >= band_lo) & (lum <= band_hi)
        grown = ((cand > 0) & keep).astype(np.uint8) * 255
        grown = cv2.erode(grown, _ellipse(1))
        grown = cv2.bitwise_or(grown, m)
        return grown

    for p in [2, 3, 4]:
        m_alt = edge_grow_small(rgb, mask_padded, p=p)
        print(f"    if edge_aware dil={p}: {int(m_alt.sum()/255)} ({int(m_alt.sum()/255)/total*100:.1f}%)")

    out_dir = ROOT / "data" / "diag_root"
    out_dir.mkdir(exist_ok=True)
    stem = name.replace('.png', '')
    Image.fromarray(mask_padded).resize((W*4, H*4), Image.NEAREST).save(out_dir / f"{stem}_mask_padded.png")
    Image.fromarray(mask_filled).resize((W*4, H*4), Image.NEAREST).save(out_dir / f"{stem}_mask_filled.png")
    # 小半径
    m3 = edge_grow_small(rgb, mask_padded, p=3)
    Image.fromarray(m3).resize((W*4, H*4), Image.NEAREST).save(out_dir / f"{stem}_mask_filled_p3.png")

print("\nsaved")