import sys; sys.path.insert(0, r'D:\Code\Project\Python\TextPatch')
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from core.text_select import detect_text_mask

orig = np.asarray(Image.open(r'D:\Code\Project\Python\TextPatch\data\needExtractAndPatch2.png').convert('RGB'))
def_ = np.asarray(Image.open(r'D:\Code\Project\Python\TextPatch\data\result\needExtractAndPatch2_F_default.png').convert('RGB'))
d60  = np.asarray(Image.open(r'D:\Code\Project\Python\TextPatch\data\result\needExtractAndPatch2_F_dir60.png').convert('RGB'))
d150 = np.asarray(Image.open(r'D:\Code\Project\Python\TextPatch\data\result\needExtractAndPatch2_F_dir150.png').convert('RGB'))

# 用 DBNet bbox
m, _ = detect_text_mask(orig, method='ml', q_off=55.0, max_area_ratio=0.40, max_box_ratio=0.40)
ys, xs = np.where(m>0)
y0,y1,x0,x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
y0=max(0,y0-2); y1=min(orig.shape[0],y1+2); x0=max(0,x0-2); x1=min(orig.shape[1],x1+2)
k=6
def z(a): return Image.fromarray(a[y0:y1, x0:x1]).resize(((x1-x0)*k,(y1-y0)*k), Image.NEAREST)
try: font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
except: font = ImageFont.load_default()
panels = [('1. 原图(座驾)',z(orig)), ('2. 默认擦除',z(def_)), ('3. 方向=60°',z(d60)), ('4. 方向=150°',z(d150))]
h = max(p[1].size[1] for p in panels) + 32
w = sum(p[1].size[0] for p in panels) + 8*(len(panels)+1)
canvas = Image.new('RGB', (w,h), (38,38,44))
draw = ImageDraw.Draw(canvas)
x=8
for lab, im in panels:
    canvas.paste(im, (x, 4))
    draw.text((x+2, im.size[1]+8), lab, fill=(220,220,220), font=font)
    x += im.size[0] + 8
canvas.save(r'D:\Code\Project\Python\TextPatch\data\result\FINAL_img2_strip.png')
print('saved FINAL_img2_strip.png')
