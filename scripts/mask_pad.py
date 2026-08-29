"""最终：mask 预外扩 + patch_fill + force_color_match"""
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from text_eraser.text_select import detect_text_mask
from text_eraser.patch_fill import inpaint


def force_color_match(img, mask, ring_radius=15):
    """把 mask 内像素按周围 ring 的均值/方差线性变换 (LAB 空间)."""
    H, W = img.shape[:2]
    m = mask.astype(bool)
    if not m.any():
        return img.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_radius * 2 + 1, ring_radius * 2 + 1))
    dil = cv2.dilate(m.astype(np.uint8) * 255, k) > 0
    ring = dil & ~m
    if ring.sum() < 50:
        ring = ~m
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    nbr_pix = lab[ring]; cur_pix = lab[m]
    t_mean = nbr_pix.mean(0); t_std = nbr_pix.std(0) + 1e-3
    c_mean = cur_pix.mean(0); c_std = cur_pix.std(0) + 1e-3
    scale = t_std / c_std
    scale[0] = max(0.6, min(scale[0], 1.4))
    new_pix = (cur_pix - c_mean) * scale + t_mean
    new_pix[:, 0] = np.clip(new_pix[:, 0], 0, 255)
    new_pix[:, 1:] = np.clip(new_pix[:, 1:], 0, 255)
    lab2 = lab.copy(); lab2[m] = new_pix
    return cv2.cvtColor(np.clip(lab2, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


img = np.asarray(Image.open(ROOT / "data/needExtractAndPatch.png").convert("RGB"), dtype=np.uint8)
H, W = img.shape[:2]
mask, _ = detect_text_mask(img, method="ml", max_area_ratio=0.40, q_off=70)

# mask 外扩 (按图尺寸比例, 大约 2-4px)
PAD = 4
mask_padded = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PAD*2+1, PAD*2+1)))
print(f"mask_padded: {int(mask_padded.sum()//255)} px (扩张 {PAD}px)")

# 整图减去扩张后 mask -> sample_mask (让 patch_fill 严格只在非 mask 区取样)
sample_mask = (255 - mask_padded).astype(np.uint8)

step1 = inpaint(img, mask_padded, sample_mask=sample_mask)
step2 = force_color_match(step1, mask_padded, ring_radius=18)

# 再做一次轻量 NS/TELEA 兜底 (这次 mask_padded 已经含缓冲)
step3 = cv2.inpaint(step2, mask_padded, 3, cv2.INPAINT_TELEA)
step4 = force_color_match(step3, mask_padded, ring_radius=18)

for name, r in [("pm_pad", step1), ("pm_pad+colormatch", step2),
                ("pm_pad+cm+ns_re", step3), ("final", step4)]:
    still = ((r > 180).all(axis=-1) & (mask > 0)).sum()
    if name == "final":
        still_real = ((r > 180).all(axis=-1) & (mask > 0)).sum()
    print(f"{name:<24} 真实mask仍白={int(still)} mask_padded区均值={r[mask_padded>0].mean(axis=0).round(1).tolist()}")
    Image.fromarray(r).save(ROOT / f"data/dryrun_out/_pad_{name}.png")

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
ring = (cv2.dilate(mask, kernel) > 0) & (mask == 0)
print(f"周围 ring 均值={img[ring].mean(axis=0).round(1).tolist()}")
