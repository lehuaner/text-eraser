"""测(1787981077691)「测」字竖笔画蒙版修复回归 + 基线七图 + 合成分隔线用例。

根因: _clean_text_mask 的反分隔线门(h/w>6)把「测」贝部孤立竖笔画(5x40,
长宽比 8.0)当「垂直分隔线」整条删除 → 蒙版缺笔画(白条 0/160)。
修复: 细长门加「远超同伴」约束 —— 长/宽需同时超长宽比门与排除自己的
同伴中位尺寸 1.5 倍才判分隔线; 无同伴时维持纯长宽比判定。
"""
import sys
import numpy as np
import cv2
from PIL import Image

ROOT = "D:/Code/Project/Python/TextPatch"
sys.path.insert(0, ROOT)
from textpatch.eraser import erase_text
from textpatch.text_select import _clean_text_mask

def run(path):
    rgb = np.array(Image.open(path).convert("RGB"))
    res, m, meta = erase_text(
        rgb, deglow_scheme="v2", glow_mode="auto", deglow_mask_soft=0.0,
        edge=1, q_off=55.0, max_area_ratio=0.4, max_box_ratio=0.4,
        deglow_strength=1.0, fill_white=True, fill_max_dist=12,
        deglow_zone_ratio=0.6, deglow_zone_expand=10, deglow_protect_px=1,
        return_mask=True, tint_fill=True, auto_edge=True)
    return res, meta

fails = []

# 1) 测: 贝部竖笔画(x163-167, y78-118)覆盖率 + 整图蒙版量
_, meta = run(f"{ROOT}/data/history/1787981077691/orig.bin")
mp = meta["mask_pre_edge"]
cov = int((mp[78:118, 163:167] > 0).sum())
tot = int((mp > 0).sum())
print(f"测1787981: mask={tot} (基线5284)  竖笔画覆盖 {cov}/160 (修复前 0)")
if cov < 140 or tot != 5284:
    fails.append("测1787981")

# 2) 武器1787: 上一个任务的修复不得回退(方案B背景亮纹理门)
_, meta = run(f"{ROOT}/data/history/1787767429309/orig.bin")
tot = int((meta["mask_pre_edge"] > 0).sum())
print(f"武器1787: mask={tot} (基线3075)")
if tot != 3075:
    fails.append("武器1787")

# 3) 基线七图(与 WORKLOG 验证基线一致)
import subprocess
r = subprocess.run([sys.executable, f"{ROOT}/scripts/regress_hz4462.py"],
                   capture_output=True, text=True)
tail = r.stdout.strip().splitlines()[-8:]
print("\n".join(tail))
for line, want in [("178", "1273"), ("556", "5518"), ("635", "1325"), ("668", "10995")]:
    ok = any(line in l and want in l for l in tail)
    if not ok:
        fails.append(line)

# 4) 合成分隔线用例: 竖分隔线删 / 字形样孤立竖笔画留 / 纯分隔线图删
img = np.zeros((200, 400), np.uint8)
for x in (50, 120, 190):
    cv2.rectangle(img, (x, 60), (x + 30, 110), 255, -1)
cv2.rectangle(img, (250, 20), (253, 185), 255, -1)   # 竖分隔线 4x166
cv2.rectangle(img, (330, 70), (335, 110), 255, -1)   # 字形样孤立竖笔画 6x41
out = _clean_text_mask((img > 0).astype(np.uint8) * 255, 200, 400,
                       min_area=8, max_area_ratio=0.9)
divider_gone = not (out[20:186, 250:254] > 0).any()
stroke_kept = int((out[70:111, 330:336] > 0).sum())
img2 = np.zeros((200, 400), np.uint8)
cv2.rectangle(img2, (250, 10), (253, 190), 255, -1)
out2 = _clean_text_mask((img2 > 0).astype(np.uint8) * 255, 200, 400,
                        min_area=8, max_area_ratio=0.9)
pure_gone = int((out2 > 0).sum()) == 0
print(f"合成用例: 竖分隔线删除={divider_gone}  字形竖笔画保留={stroke_kept}/246"
      f"  纯分隔线图删除={pure_gone}")
if not divider_gone or stroke_kept < 200 or not pure_gone:
    fails.append("synthetic")

print("\n==>", "全部通过" if not fails else f"失败: {fails}")
sys.exit(1 if fails else 0)
