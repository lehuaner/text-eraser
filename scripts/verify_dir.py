import sys, time
sys.path.insert(0, r'D:\Code\Project\Python\TextEraser')
import numpy as np, cv2
from PIL import Image
from text_eraser.eraser import erase_text

for name in ['needExtractAndPatch.png', 'needExtractAndPatch2.png']:
    path = r'D:\Code\Project\Python\TextEraser\data\\' + name
    rgb = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
    r0, m0, meta0 = erase_text(rgb, return_mask=True)
    r1, m1, meta1 = erase_text(rgb, direction=60.0, return_mask=True)
    r2, m2, meta2 = erase_text(rgb, direction=-30.0, return_mask=True)
    ys, xs = np.where(m0 > 0)
    by0,by1,bx0,bx1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
    olum = cv2.cvtColor(rgb[by0:by1, bx0:bx1], cv2.COLOR_RGB2LAB)[...,0]
    text_lo = float(olum[m0[by0:by1,bx0:bx1]>0].min()) if (m0[by0:by1,bx0:bx1]>0).any() else 0
    bg = float(np.median(olum[m0[by0:by1,bx0:bx1]==0])) if (m0[by0:by1,bx0:bx1]==0).any() else 128
    thr = (text_lo + bg)/2
    for tag, r in [('default', r0), ('dir60', r1), ('dir-30', r2)]:
        lum = cv2.cvtColor(r[by0:by1, bx0:bx1], cv2.COLOR_RGB2LAB)[...,0]
        print(f"[{name}][{tag}] mask_pix={meta0['mask_pix'] if tag=='default' else (meta1['mask_pix'] if tag=='dir60' else meta2['mask_pix'])}  bbox残余亮像素(>{thr:.0f})={int((lum>thr).sum())}")
    Image.fromarray(r1).save(r'D:\Code\Project\Python\TextEraser\data\result\\' + name.replace('.png','') + '_dir60.png')
    Image.fromarray(r2).save(r'D:\Code\Project\Python\TextEraser\data\result\\' + name.replace('.png','') + '_dir-30.png')
    print(f"  saved {name} *_dir60.png *_dir-30.png")
print("ALL DONE")
