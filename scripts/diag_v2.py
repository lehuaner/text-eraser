"""
更细: D 方案 + 局部 TELEA 兜底 (只对残留的"白点"打补丁, 不糊整片)
G. D + 局部 TELEA (只在 ghost 像素)
H. 2px dil + sample_mask
I. 2px dil + sample_mask + 局部 TELEA
"""
import os, sys
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from textpatch.text_select import detect_text_mask
from textpatch.patch_fill import inpaint as pm_inpaint

IMG = r"D:\Code\Project\Python\TextPatch\data\needExtractAndPatch.png"
OUT = r"D:\Code\Project\Python\TextPatch\data\dryrun_out"
os.makedirs(OUT, exist_ok=True)

rgb = np.array(Image.open(IMG).convert("RGB"))
mask_orig, _ = detect_text_mask(rgb, method="ml", q_off=70, max_area_ratio=0.40)


def k(p): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p*2+1, p*2+1))


def ghost_mask(res, original_mask, threshold=170):
    """找还在 mask 内仍像文字亮度的像素."""
    bright = ((res[..., 0] > threshold) & (res[..., 1] > threshold) & (res[..., 2] > threshold))
    return bright & (original_mask > 0)


def stats(name, res, mask_eval):
    m = mask_eval > 0
    if not m.any():
        return
    dil_outer = cv2.dilate(m.astype(np.uint8), k(12))
    ring = (dil_outer > 0) & (~m)
    cur = res[m]; nbr = res[ring]
    print(f"  {name:35s} fill mean={cur.mean():.1f} std={cur.std():.1f} | ring mean={nbr.mean():.1f} std={nbr.std():.1f} | Δ={abs(cur.mean()-nbr.mean()):.1f}")


def patch_run(mask_for_fill, sample_mask_used, mask_for_eval, name):
    res = pm_inpaint(rgb, mask_for_fill, sample_mask=sample_mask_used)
    stats(name, res, mask_for_eval)
    return res


print("=== H: 2px dil + sample_mask ===")
mask2 = cv2.dilate(mask_orig, k(2))
sm2 = 255 - mask2
res_H = patch_run(mask2, sm2, mask_orig, "H_2pxdil+sm")

print("\n=== I: 2px dil + sample_mask + 局部 TELEA (ghost only) ===")
res_H2 = pm_inpaint(rgb, mask2, sample_mask=sm2)
gh = ghost_mask(res_H2, mask_orig)
gh_mask = (gh.astype(np.uint8) * 255)
gh_dil = cv2.dilate(gh_mask, k(1))  # 仅 1px 局部
res_I = cv2.inpaint(res_H2, gh_dil, 2, cv2.INPAINT_TELEA)
stats("I_2pxdil+sm+localTELEA", res_I, mask_orig)

print("\n=== J: 1px dil + sample_mask + 局部 TELEA (ghost only) ===")
mask1 = cv2.dilate(mask_orig, k(1))
sm1 = 255 - mask1
res_J0 = pm_inpaint(rgb, mask1, sample_mask=sm1)
gh = ghost_mask(res_J0, mask_orig)
gh_mask = (gh.astype(np.uint8) * 255)
gh_dil = cv2.dilate(gh_mask, k(1))
res_J = cv2.inpaint(res_J0, gh_dil, 2, cv2.INPAINT_TELEA)
stats("J_1pxdil+sm+localTELEA", res_J, mask_orig)

print("\n=== K: 1px dil + sample_mask + 局部 NS (cv2.INPAINT_NS) ===")
res_K0 = pm_inpaint(rgb, mask1, sample_mask=sm1)
gh = ghost_mask(res_K0, mask_orig)
gh_mask = (gh.astype(np.uint8) * 255)
gh_dil = cv2.dilate(gh_mask, k(1))
res_K = cv2.inpaint(res_K0, gh_dil, 2, cv2.INPAINT_NS)
stats("K_1pxdil+sm+localNS", res_K, mask_orig)

# 保存 I 的 zoom 看效果
def zoom_save(name, res, src=rgb):
    H, W = src.shape[:2]
    ys, xs = np.where(mask_orig > 0)
    y0, y1 = max(0, ys.min()-8), min(H, ys.max()+8)
    x0, x1 = max(0, xs.min()-8), min(W, xs.max()+8)
    orig_c = src[y0:y1, x0:x1]
    res_c = res[y0:y1, x0:x1]
    zoom = 4
    o = cv2.resize(orig_c, (orig_c.shape[1]*zoom, orig_c.shape[0]*zoom), interpolation=cv2.INTER_NEAREST)
    r = cv2.resize(res_c, (res_c.shape[1]*zoom, res_c.shape[0]*zoom), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(OUT, f"_zoom_{name}.png"),
                np.concatenate([o, np.full((o.shape[0], 4, 3), 30, dtype=np.uint8), r], axis=1)[:, :, ::-1])
    return os.path.join(OUT, f"_zoom_{name}.png")

for nm, r in [("H_2px", res_H), ("I_2px_local", res_I), ("J_1px_local", res_J), ("K_1px_NS", res_K)]:
    p = zoom_save(nm, r)
    print(f"  ✓ {p}")

# 拼 4x2: 4 zoom 对比 (原图|结果 原图|结果 ...)
imgs = [res_H, res_I, res_J, res_K]
labels = ["H 2px+dil", "I 2px+localTELEA", "J 1px+localTELEA", "K 1px+localNS"]
zoom = 3
H, W = rgb.shape[:2]
ys, xs = np.where(mask_orig > 0)
y0, y1 = max(0, ys.min()-8), min(H, ys.max()+8)
x0, x1 = max(0, xs.min()-8), min(W, xs.max()+8)
crop = rgb[y0:y1, x0:x1]
cw, ch = (x1-x0)*zoom, (y1-y0)*zoom
o_big = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_NEAREST)
gap = 6
canvas = np.full((ch*4 + gap*5, cw, 3), 30, dtype=np.uint8)
for i, (r, l) in enumerate(zip(imgs, labels)):
    rc = r[y0:y1, x0:x1]
    r_big = cv2.resize(rc, (cw, ch), interpolation=cv2.INTER_NEAREST)
    y = gap + i*(ch+gap)
    canvas[y:y+ch] = r_big
from PIL import Image, ImageDraw, ImageFont
img = Image.fromarray(canvas)
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("consola.ttf", 11)
except:
    font = ImageFont.load_default()
for i, l in enumerate(labels):
    y = gap + i*(ch+gap) - 2
    draw.text((4, y), f"{i}. {l}", fill=(220,220,220), font=font)
img.save(os.path.join(OUT, "_zoom_4way.png"))
print(f"  ✓ {os.path.join(OUT, '_zoom_4way.png')}")