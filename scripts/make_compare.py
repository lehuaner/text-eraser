import sys; sys.path.insert(0, r'D:\Code\Project\Python\TextPatch')
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

def load(p): return np.asarray(Image.open(p).convert('RGB'))
def zoom(im, k=5):
    return Image.fromarray(im).resize((im.shape[1]*k, im.shape[0]*k), Image.NEAREST)
def strip(panels, title, out):
    # panels: [(label, img_array_hwc), ...]
    k=5
    h_im = max(p[1].shape[0] for p in panels) * k
    label_h = 24
    pad = 8
    w = sum(p[1].shape[1]*k for p in panels) + pad*(len(panels)+1)
    canvas = Image.new('RGB', (w, h_im + label_h + 30), (38,38,44))
    draw = ImageDraw.Draw(canvas)
    try: font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    except: font = ImageFont.load_default()
    draw.text((pad, 4), title, fill=(230,230,230), font=font)
    x = pad
    y = label_h + 6
    for lab, im in panels:
        canvas.paste(zoom(im, k), (x, y))
        draw.text((x+2, y + h_im + 4), lab, fill=(210,210,210), font=font)
        x += im.shape[1]*k + pad
    canvas.save(out)
    print(f"saved {out}")

orig1 = load(r'D:\Code\Project\Python\TextPatch\data\needExtractAndPatch.png')
orig2 = load(r'D:\Code\Project\Python\TextPatch\data\needExtractAndPatch2.png')
# 默认结果（重新跑以保证一致）
from core.eraser import erase_text
def1, _, _ = erase_text(orig1, return_mask=True)
def2, _, _ = erase_text(orig2, return_mask=True)
dir60_1 = load(r'D:\Code\Project\Python\TextPatch\data\result\needExtractAndPatch_dir60.png')
dir60_2 = load(r'D:\Code\Project\Python\TextPatch\data\result\needExtractAndPatch2_dir60.png')
dirN30_2 = load(r'D:\Code\Project\Python\TextPatch\data\result\needExtractAndPatch2_dir-30.png')

strip([('原图',orig1),('默认(无方向)',def1),('方向60°',dir60_1)],
      'img1 武器  — 默认 vs 方向60°', r'D:\Code\Project\Python\TextPatch\data\result\cmp_img1.png')
strip([('原图',orig2),('默认(无方向)',def2),('方向60°',dir60_2),('方向-30°',dirN30_2)],
      'img2 座驾 — 默认 vs 方向60° vs 方向-30°', r'D:\Code\Project\Python\TextPatch\data\result\cmp_img2.png')
