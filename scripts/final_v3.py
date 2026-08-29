"""最终验证: 改 edge_aware 默认 False 后, 两图 (武器 + 座驾) 是否都干净."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from text_eraser.text_select import to_rgb_uint8
from text_eraser.eraser import erase_text

def font():
    try: return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 14)
    except: return ImageFont.load_default()

def make_strip(orig, mask, result, name, out_path):
    """3栏对比: 原图 / mask (4x) / result (4x), 带标签."""
    H, W = orig.shape[:2]
    k = 4
    pH, pW = H + 30, W * 3 + 20 * 4
    canvas = Image.new("RGB", (pW, pH), (245, 245, 245))
    dr = ImageDraw.Draw(canvas)
    # 原图 (1x)
    canvas.paste(Image.fromarray(orig), (10, 25))
    # mask overlay (4x)
    m4 = Image.fromarray(mask).resize((W*k, H*k), Image.NEAREST)
    # 叠原图当背景, 半透明红
    bg = Image.fromarray(orig).resize((W*k, H*k), Image.NEAREST).convert("RGB")
    overlay = bg.copy()
    arr = np.array(overlay)
    m_bool = np.array(m4) > 0
    if m_bool.any():
        arr[m_bool] = (arr[m_bool].astype(np.int32) * 0.35
                       + np.array([255, 60, 60]) * 0.65).clip(0, 255).astype(np.uint8)
    canvas.paste(Image.fromarray(arr), (W + 30, 25))
    # result (4x)
    r4 = Image.fromarray(result).resize((W*k, H*k), Image.NEAREST)
    canvas.paste(r4, (W * 2 + 50, 25))
    dr.text((10, 5), f"{name}  原图(1x)", fill=(50,50,50), font=font())
    dr.text((W + 30, 5), f"{name}  mask 4x (红=擦除区)", fill=(180,30,30), font=font())
    dr.text((W * 2 + 50, 5), f"{name}  擦除结果 4x", fill=(30,130,30), font=font())
    canvas.save(out_path)

def stats(orig, result, name):
    lum_o = cv2.cvtColor(orig, cv2.COLOR_RGB2LAB)[..., 0]
    lum_r = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)[..., 0]
    # 全图统计
    white_o = int((lum_o > 200).sum())
    white_r = int((lum_r > 200).sum())
    red_o = int((lum_o < 80).sum())
    red_r = int((lum_r < 80).sum())
    print(f"  [{name}] 全图统计:")
    print(f"    白像素: 原 {white_o} → 结果 {white_r} (差 {white_r - white_o:+d})")
    print(f"    红像素: 原 {red_o} → 结果 {red_r} (差 {red_r - red_o:+d})")

out_dir = ROOT / "data" / "final"
out_dir.mkdir(exist_ok=True)

for name in ['needExtractAndPatch.png', 'needExtractAndPatch2.png']:
    p = ROOT / "data" / name
    rgb = to_rgb_uint8(Image.open(p))
    H, W = rgb.shape[:2]

    # 默认参数 (edge_aware 现在默认 False)
    r, m, meta = erase_text(rgb, return_mask=True)
    print(f"\n=== {name} ({W}x{H}) ===")
    print(f"  mask_pix={meta['mask_pix']} ({meta['mask_pix']/(H*W)*100:.1f}%) boxes={len(meta['boxes'])}")
    print(f"  elapsed={meta['inpaint_seconds']:.3f}s")
    stats(rgb, r, name)

    # 保存单图
    stem = name.replace('.png', '')
    Image.fromarray(rgb).save(out_dir / f"{stem}_orig.png")
    Image.fromarray(r).save(out_dir / f"{stem}_result.png")
    Image.fromarray(m).save(out_dir / f"{stem}_mask.png")

    # 3栏对比 (1x原图 + 4x mask overlay + 4x result)
    make_strip(rgb, m, r, stem, out_dir / f"{stem}_compare.png")

print("\nsaved to", out_dir)