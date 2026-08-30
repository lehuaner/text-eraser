import sys, time, traceback
sys.path.insert(0, r'D:\Code\Project\Python\TextEraser')
import numpy as np, cv2
from PIL import Image
from text_eraser.eraser import erase_text

for name in ['needExtractAndPatch.png', 'needExtractAndPatch2.png']:
    path = r'D:\Code\Project\Python\TextEraser\data\\' + name
    rgb = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
    # 默认（无方向）
    r0, m0, meta0 = erase_text(rgb, return_mask=True)
    ys, xs = np.where(m0 > 0)
    by0,by1,bx0,bx1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
    olum = cv2.cvtColor(rgb[by0:by1, bx0:bx1], cv2.COLOR_RGB2LAB)[...,0]
    text_lo = float(olum[m0[by0:by1,bx0:bx1]>0].min()) if (m0[by0:by1,bx0:bx1]>0).any() else 0
    bg = float(np.median(olum[m0[by0:by1,bx0:bx1]==0])) if (m0[by0:by1,bx0:bx1]==0).any() else 128
    thr = (text_lo + bg)/2
    lum0 = cv2.cvtColor(r0[by0:by1, bx0:bx1], cv2.COLOR_RGB2LAB)[...,0]
    dir60_ok = False
    try:
        r1, m1, meta1 = erase_text(rgb, direction=60.0, return_mask=True)
        lum1 = cv2.cvtColor(r1[by0:by1, bx0:bx1], cv2.COLOR_RGB2LAB)[...,0]
        dir60_ok = True
    except Exception as e:
        print(f"[{name}] direction=60 FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
    print(f"[{name}] default mask_pix={meta0['mask_pix']}" + (f" dir60 mask_pix={meta1['mask_pix']}" if dir60_ok else " dir60 mask_pix=N/A"))
    print(f"    bbox 内残余亮像素(>{thr:.0f}): 默认={int((lum0>thr).sum())}" + (f"  方向60={int((lum1>thr).sum())}" if dir60_ok else ""))
    if dir60_ok:
        Image.fromarray(r1).save(r'D:\Code\Project\Python\TextEraser\data\result\\' + name.replace('.png','') + '_dir60.png')
        print(f"    saved *_dir60.png")
