"""
对四张发光样图跑默认参数，输出 result + mask + 文字区局部裁剪便于对比。
"""
import os, io
import numpy as np
from PIL import Image
from core.eraser import erase_text

OUT = "data/_glowcheck"
os.makedirs(OUT, exist_ok=True)

# 默认参数（与代码默认值一致）
DEFAULTS = dict(
    edge=1, q_off=55.0, max_area_ratio=0.4, max_box_ratio=0.4,
    direction=None, edge_aware=False,
    glow_mode="auto", deglow_strength=1.0,
    deglow_green_thr=6.0, deglow_range=24, deglow_glo=85.0,
    deglow_protect=1.0, deglow_mask_soft=0.0, deglow_scheme="channel",
    fill_white=True, fill_max_dist=12,
    auto_edge=False, auto_max_edge=2,
    tint_fill=True, return_mask=True,
)

samples = {
    "635": ("data/_glowcheck/635.png", (7,13,70,72)),
    "178": ("data/_glowcheck/178.png", (0,0,63,54)),
    "556": ("data/_glowcheck/556.png", (120,62,242,182)),
    "668": ("data/_glowcheck/668.png", (115,83,307,275)),
}

for tag, (path, box) in samples.items():
    rgb = np.array(Image.open(path).convert("RGB"))
    result, mask, meta = erase_text(rgb, **DEFAULTS)
    Image.fromarray(result).save(f"{OUT}/{tag}_result.png")
    Image.fromarray(mask).save(f"{OUT}/{tag}_mask.png")

    # 文字区局部裁剪（原图 / 结果 / 残差）
    x0,y0,x1,y1 = box
    H,W = rgb.shape[:2]
    pad = 20
    cx0 = max(0, x0-pad); cy0 = max(0, y0-pad)
    cx1 = min(W, x1+pad); cy1 = min(H, y1+pad)
    orig_crop = rgb[cy0:cy1, cx0:cx1]
    res_crop  = result[cy0:cy1, cx0:cx1]
    diff = np.abs(orig_crop.astype(int) - res_crop.astype(int)).astype(np.uint8)

    Image.fromarray(orig_crop).save(f"{OUT}/{tag}_orig_crop.png")
    Image.fromarray(res_crop).save(f"{OUT}/{tag}_result_crop.png")
    Image.fromarray(diff).save(f"{OUT}/{tag}_diff_crop.png")

    boxes = meta.get("boxes", [])
    print(f"=== {tag} (size {W}x{H}) ===")
    print(f"  boxes={boxes}  mask_pix={int((mask>0).sum())}  elapsed={meta.get('inpaint_seconds','?'):.3f}s")
    print(f"  glow_mode_eff={meta.get('glow_mode_eff','?')}  pre_glow_pix={meta.get('pre_glow_pix','?')}  tint={meta.get('tint_pix','?')}")