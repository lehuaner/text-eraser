import sys; sys.path.insert(0, r'D:\Code\Project\Python\TextEraser')
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from text_eraser.eraser import erase_text
from text_eraser.text_select import detect_text_mask

def load(p): return np.asarray(Image.open(p).convert('RGB'))
def zoom8(im): return Image.fromarray(im).resize((im.shape[1]*8, im.shape[0]*8), Image.NEAREST)
def font():
    try: return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    except: return ImageFont.load_default()

for name in ['needExtractAndPatch.png', 'needExtractAndPatch2.png']:
    path = r'D:\Code\Project\Python\TextEraser\data\\' + name
    orig = load(path)
    r_def, m_def, _ = erase_text(orig, return_mask=True)            # q_off=55 默认
    r_d60, _, _ = erase_text(orig, direction=60.0, return_mask=True)
    r_d150, _, _ = erase_text(orig, direction=150.0, return_mask=True)
    Image.fromarray(r_def).save(r'D:\Code\Project\Python\TextEraser\data\result\\' + name.replace('.png','') + '_F_default.png')
    Image.fromarray(r_d60).save(r'D:\Code\Project\Python\TextEraser\data\result\\' + name.replace('.png','') + '_F_dir60.png')
    Image.fromarray(r_d150).save(r'D:\Code\Project\Python\TextEraser\data\result\\' + name.replace('.png','') + '_F_dir150.png')
    print(f"[{name}] default mask_pix={int((m_def>0).sum())}")
