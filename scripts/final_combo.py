"""组合：patch_fill (含 sample_mask) + 整 mask 区域颜色匹配到周围 ring"""
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from textpatch.text_select import detect_text_mask
from textpatch.patch_fill import inpaint


def force_color_match(img, mask, ring_radius=15):
    """把 mask 内像素按周围 ring 的均值/方差做线性变换 (LAB 颜色空间更稳)."""
    H, W = img.shape[:2]
    m = mask.astype(bool)
    if not m.any():
        return img.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_radius * 2 + 1, ring_radius * 2 + 1))
    dil = cv2.dilate(m.astype(np.uint8) * 255, k) > 0
    ring = dil & ~m
    if ring.sum() < 50:
        ring = ~m
    # LAB 空间:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    nbr_pix = lab[ring]
    cur_pix = lab[m]
    t_mean = nbr_pix.mean(0); t_std = nbr_pix.std(0) + 1e-3
    c_mean = cur_pix.mean(0); c_std = cur_pix.std(0) + 1e-3
    # 通道独立: 缩放到目标 std, 平移到目标 mean. L 通道稍温和 (不破坏纹理对比)
    scale = t_std / c_std
    scale[0] = max(0.6, min(scale[0], 1.4))   # L 通道仅做温和缩放 (保留纹理明暗对比)
    new_pix = (cur_pix - c_mean) * scale + t_mean
    new_pix[:, 0] = np.clip(new_pix[:, 0], 0, 255)
    new_pix[:, 1:] = np.clip(new_pix[:, 1:], 0, 255)
    lab2 = lab.copy()
    lab2[m] = new_pix
    return cv2.cvtColor(np.clip(lab2, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


img = np.asarray(Image.open(ROOT / "data/needExtractAndPatch.png").convert("RGB"), dtype=np.uint8)
H, W = img.shape[:2]
mask, _ = detect_text_mask(img, method="ml", max_area_ratio=0.40, q_off=70)
print(f"input {W}x{H}, mask_pix={int(mask.sum() // 255)}")
sample_mask = (255 - mask).astype(np.uint8)

# 1) patch_fill (含 sample_mask) 提供纹理
step1 = inpaint(img, mask, sample_mask=sample_mask)
# 2) 整 mask 区颜色匹配到 ring
step2 = force_color_match(step1, mask, ring_radius=15)
# 3) NS 二次清扫残余笔画痕迹 (用缩小版 mask 内 mask, 二次清扫 = 用 mask 本身)
step3 = cv2.inpaint(step2, mask, 3, cv2.INPAINT_TELEA)
# 4) 再做一次颜色匹配
step4 = force_color_match(step3, mask, ring_radius=15)

# 评估
for name, r in [("step1_pm", step1), ("step2+colormatch", step2),
                ("step3+NS", step3), ("step4+recolor", step4)]:
    still = ((r > 180).all(axis=-1) & (mask > 0)).sum()
    print(f"{name:<22} 仍白={int(still):3d}  mask区均值={r[mask>0].mean(axis=0).round(1).tolist()}")
    Image.fromarray(r).save(ROOT / f"data/dryrun_out/_final_{name}.png")

# ring 均值参考
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
dil = cv2.dilate(mask, kernel); ring = (dil > 0) & (mask == 0)
print(f"周围 ring 均值={img[ring].mean(axis=0).round(1).tolist()}  全图均值={img.reshape(-1,3).mean(0).round(1).tolist()}")
