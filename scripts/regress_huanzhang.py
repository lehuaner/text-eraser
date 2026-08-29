"""换装.png(6464)修复回归: 平滑渐变背景自适应 TELEA。
覆盖: 换装(该走TELEA) / 178 556 635 668(应保持 patchmatch) / 反例验证。
"""
import sys
import numpy as np
import cv2
from PIL import Image

ROOT = "D:/Code/Project/Python/TextPatch"
sys.path.insert(0, ROOT)
from textpatch.eraser import erase_text
from textpatch.patch_fill import inpaint as pm_inpaint

def load(p):
    return np.array(Image.open(p).convert("RGB"))

# ---- 换装: 前端默认参数全流程 ----
rgb = load(f"{ROOT}/data/_glowcheck/huanzhang.png")
res, m, meta = erase_text(
    rgb, deglow_scheme="v2", glow_mode="auto", deglow_mask_soft=0.0,
    edge=1, deglow_strength=1, fill_white=True, fill_max_dist=12,
    deglow_zone_ratio=0.6, deglow_zone_expand=10, deglow_protect_px=1,
    return_mask=True, tint_fill=True)
g = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY).astype(np.float32)
mask = m
hole_px = int((mask > 0).sum())
# 洞内黑碎块量化: 填充区明显暗于周边背景的像素
ring = (cv2.dilate(mask, np.ones((25, 25), np.uint8)) > 0) & (mask == 0)
ringmed = np.median(g[ring])
dark_blocks = int(((g < ringmed - 30) & (mask > 0)).sum())
print(f"换装: 洞 {hole_px}px, 环带中位 {ringmed:.0f}, 洞内暗碎块(<中位-30): {dark_blocks}px")
z = 4
Image.fromarray(cv2.resize(np.hstack([rgb, res]), None, fx=z, fy=z,
                           interpolation=cv2.INTER_NEAREST)
                ).save(f"{ROOT}/data/_glowcheck/_hz_final.png")

# ---- 四样图回归(前端默认参数) ----
print(f"\n{'tag':>5} {'mask_pix':>9} {'resid_px':>9}")
for tag in ["178", "556", "635", "668"]:
    im = load(f"{ROOT}/data/_glowcheck/{tag}.png")
    r2, m2, meta2 = erase_text(
        im, deglow_scheme="v2", glow_mode="auto", deglow_mask_soft=0.0,
        edge=1, deglow_strength=1, fill_white=True, fill_max_dist=12,
        deglow_zone_ratio=0.6, deglow_zone_expand=10, deglow_protect_px=1,
        return_mask=True, tint_fill=True)
    gres = cv2.cvtColor(r2, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gorg = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    from textpatch.text_select import detect_text_mask
    tm, _ = detect_text_mask(im, method="ml", tint_fill=False,
                             max_area_ratio=0.40, q_off=55,
                             fill_white=True, fill_max_dist=12)
    roi = np.zeros(im.shape[:2], bool)
    if tm.any():
        ys, xs = np.nonzero(tm)
        roi[max(0, ys.min()-30):ys.max()+30, max(0, xs.min()-30):xs.max()+30] = True
    resid = int(((gres > 130) & roi & (gorg < 110)).sum())
    print(f"{tag:>5} {meta2['mask_pix']:>9} {resid:>9}")

# ---- patch_fill 直填对照: 强制关自适应 vs 自适应(换装) ----
mask_filled = m
sample = (255 - mask_filled).astype(np.uint8)
r_adapt = pm_inpaint(rgb, mask_filled, sample_mask=sample)
r_pm = pm_inpaint(rgb, mask_filled, sample_mask=sample, flat_span=99999)  # 关自适应
diff = int((np.abs(r_adapt.astype(int) - r_pm.astype(int)).sum(-1) > 10).sum())
print(f"\npatch_fill 自适应 vs 关自适应 差异px: {diff} (应>0, 即自适应生效走TELEA)")
