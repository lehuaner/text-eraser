"""验证 patch_fill 去平均/去颜色匹配后，文字擦除是否恢复锐利（复原刚做完效果）。

模糊量化：填充区拉普拉斯方差 / 周边原纹理拉普拉斯方差。
  ratio≈1  => 填充纹理与原纹理锐度一致（无涂抹/糊）
  ratio<<1 => 填充区被平均抹平（模糊）
"""
import sys, time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from text_eraser.eraser import erase_text

IMGS = {
    "武器": Path(r"D:\Code\Project\Python\ExtractRole\data\needExtractAndPatch.png"),
    "座驾": ROOT / "data" / "diag_root" / "needExtractAndPatch2_orig.png",
}
OUT = ROOT / "data" / "verify_fix"
OUT.mkdir(parents=True, exist_ok=True)


def lap_var(img, m):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    return float(lap[m].var())


def blur_ratio(result, mask):
    m = mask > 0
    if not m.any():
        return float("nan")
    # 周边原纹理环：mask 外扩 8px 再减去 mask
    ring = (cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))) > 0) & ~m
    if not ring.any():
        return float("nan")
    return lap_var(result, m) / lap_var(result, ring)


for name, p in IMGS.items():
    rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    for gm in ("off", "auto"):
        t0 = time.time()
        res, mask, meta = erase_text(
            rgb, mask_pad=2, q_off=55.0,
            max_area_ratio=0.40, max_box_ratio=0.40,
            glow_mode=gm, return_mask=True,
        )
        dt = time.time() - t0
        r = blur_ratio(res, mask)
        # 残留度量：白(>200) / 红(>150 且 R 主导) 像素在原图文字附近
        Image.fromarray(res).save(OUT / f"{name}_glow-{gm}.png")
        Image.fromarray(mask).save(OUT / f"{name}_glow-{gm}_mask.png")
        print(f"[{name}] glow={gm:4s}  mask_pix={meta['mask_pix']:5d} "
              f"mask_filled={meta['mask_filled_pix']:5d}  blur_ratio={r:.3f}  {dt:.2f}s")
    print()

# 拼一张「武器 off」原图/结果/蒙版对比，便于肉眼复核锐度
a = np.asarray(Image.open(IMGS["武器"]).convert("RGB"))
r, m, _ = erase_text(a, glow_mode="off", return_mask=True)
cmp = np.concatenate([a, r, np.stack([m]*3, -1)], axis=1)
Image.fromarray(cmp).save(OUT / "_weapon_compare.png")
b = np.asarray(Image.open(IMGS["座驾"]).convert("RGB"))
r2, m2, _ = erase_text(b, glow_mode="off", return_mask=True)
cmp2 = np.concatenate([b, r2, np.stack([m2]*3, -1)], axis=1)
Image.fromarray(cmp2).save(OUT / "_zuojia_compare.png")
print("saved compare images ->", OUT)
