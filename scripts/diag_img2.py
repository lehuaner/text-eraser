"""诊断 needExtractAndPatch2 (座驾) 漏白边根因，对比 mask 补救方案。"""
import sys, time
sys.path.insert(0, r'D:\Code\Project\Python\TextPatch')
import numpy as np, cv2
from PIL import Image
from text_eraser.text_select import detect_text_mask, to_rgb_uint8
from text_eraser.patch_fill import inpaint as pm_inpaint

P = r'D:\Code\Project\Python\TextPatch\data\needExtractAndPatch2.png'
rgb = to_rgb_uint8(Image.open(P).convert('RGB'))
H, W = rgb.shape[:2]

mask, boxes = detect_text_mask(rgb, method="ml", q_off=70.0,
                               max_area_ratio=0.40, max_box_ratio=0.40, max_side=960)
print("boxes:", boxes, "mask_pix:", int(mask.sum()//255))

# 文字 bbox
ys, xs = np.where(mask>0)
by0,by1,bx0,bx1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
# 扩张一些看周围
pad = 6
by0=max(0,by0-pad); by1=min(H,by1+pad); bx0=max(0,bx0-pad); bx1=min(W,bx1+pad)

orig_crop = rgb[by0:by1, bx0:bx1]
lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[...,0].astype(np.float32)
# 文字亮度带
text_lum = lum[mask>0]
lo, hi = float(text_lum.min()), float(text_lum.max())
bg_lum = float(np.median(lum[(mask==0) & (ys.min()<=np.arange(H)[:,None]) & (np.arange(W)[None,:]>=0)])) if False else None
# 用 bbox 外区域估背景
bboxmask = np.zeros((H,W),bool); bboxmask[by0:by1, bx0:bx1]=True
bg_lum = float(np.median(lum[bboxmask & (mask==0)]))
print(f"text_lum band = [{lo:.0f}, {hi:.0f}], bbox bg lum = {bg_lum:.0f}")

def fill_and_measure(fill_mask, tag):
    t0=time.time()
    sample = (255 - fill_mask).astype(np.uint8)
    res = pm_inpaint(rgb, fill_mask, sample_mask=sample)
    el = time.time()-t0
    # 在 bbox 内，原图白色文字像素(亮于 (lo+bg)/2)但结果里仍亮 -> 漏
    crop = res[by0:by1, bx0:bx1]
    lum_res = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)[...,0].astype(np.float32)
    thr = (lo + bg_lum)/2.0
    leaked = int((lum_res > thr).sum())
    print(f"[{tag}] elapsed={el:.2f}s  bbox内残留亮像素(>{(lo+bg_lum)/2:.0f})={leaked}")
    return res

def ell(p): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(p*2+1,p*2+1))

# 方案A: pad=2 (当前)
m2 = cv2.dilate(mask, ell(2))
resA = fill_and_measure(m2, "pad=2")

# 方案B: pad=3
m3 = cv2.dilate(mask, ell(3))
resB = fill_and_measure(m3, "pad=3")

# 方案C: 边缘感知 — 在 pad=3 基础上，把"靠近 mask 且亮度属于文字带"的原图像素也并入
cand = cv2.dilate(mask, ell(4))
# 文字带: 高于背景与文字阈值的折中，且不超过文字 hi
band_lo = (bg_lum + lo)/2.0
band_hi = hi + (hi-lo)*0.5
keep = (lum >= band_lo) & (lum <= band_hi)
mc = ((cand>0) & keep).astype(np.uint8)*255
mc = cv2.dilate(mc, ell(1))  # 收一下
resC = fill_and_measure(mc, "edge-aware(band)")

# 4x 放大对比
def zoom(p, k=4):
    im = Image.fromarray(p).resize((p.shape[1]*k, p.shape[0]*k), Image.NEAREST)
    return im
outA = zoom(resA[by0:by1, bx0:bx1]); outA.save(r'D:\Code\Project\Python\TextPatch\data\dryrun_out\_img2_pad2_4x.png')
outB = zoom(resB[by0:by1, bx0:bx1]); outB.save(r'D:\Code\Project\Python\TextPatch\data\dryrun_out\_img2_pad3_4x.png')
outC = zoom(resC[by0:by1, bx0:bx1]); outC.save(r'D:\Code\Project\Python\TextPatch\data\dryrun_out\_img2_edge_4x.png')
origz = zoom(orig_crop); origz.save(r'D:\Code\Project\Python\TextPatch\data\dryrun_out\_img2_orig_4x.png')
print("saved zooms")
