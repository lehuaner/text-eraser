import sys; sys.path.insert(0, r'D:\Code\Project\Python\TextPatch')
import numpy as np, cv2
from PIL import Image
from core.eraser import erase_text
for name in ['needExtractAndPatch.png', 'needExtractAndPatch2.png']:
    path = r'D:\Code\Project\Python\TextPatch\data\\' + name
    rgb = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
    r_def, m_def, _ = erase_text(rgb, return_mask=True)
    r_d60, m_d60, _ = erase_text(rgb, direction=60.0, return_mask=True)
    Image.fromarray(r_def).save(r'D:\Code\Project\Python\TextPatch\data\result\\' + name.replace('.png','') + '_v2_default.png')
    Image.fromarray(r_d60).save(r'D:\Code\Project\Python\TextPatch\data\result\\' + name.replace('.png','') + '_v2_dir60.png')
print("regenerated")
