"""单独输出四张「去发光结果」(流程图第③步, 原分辨率, 不改内容)。"""
import sys
import numpy as np
from PIL import Image

ROOT = "D:/Code/Project/Python/TextEraser"
sys.path.insert(0, ROOT)
from text_eraser.text_select import detect_text_mask, _deglow_full_green_v2

TAGS = ["178", "556", "635", "668"]


def main():
    for tag in TAGS:
        rgb = np.array(Image.open(
            f"{ROOT}/data/_glowcheck/{tag}.png").convert("RGB"))
        tmask, _ = detect_text_mask(rgb, method="ml", tint_fill=False,
                                    max_area_ratio=0.40, q_off=55,
                                    fill_white=True, fill_max_dist=12)
        clean, core = _deglow_full_green_v2(
            rgb, tmask, strength=1.15, alpha_core=0.65,
            zone_ratio=0.6, zone_expand=24)
        out = f"{ROOT}/data/_glowcheck/deglow_{tag}.png"
        Image.fromarray(clean).save(out)
        print(f"saved {out}  {clean.shape[1]}x{clean.shape[0]}  (原分辨率)")


if __name__ == "__main__":
    main()