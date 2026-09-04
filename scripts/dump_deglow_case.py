# -*- coding: utf-8 -*-
"""B 场跨端对比台 (handover §13.5-1a)。

从 img1 (1787767556635) 提取暖路径测地背景阶段的真实输入:
  rgb / zone(扩边后) / ring_clean
落盘 data/_pmparity/deglow_case.bin, 并用 cv2 `_geodesic_background`
算出参照 B / D_rg / D_gb 存 .npy。

case.bin 布局: h(i32) w(i32) | rgb f32 (n*3) | zone u8 (n) | ring u8 (n)
"""
import sys
import numpy as np
import cv2

sys.path.insert(0, r"D:/Code/Project/Python/TextPatch")
import text_eraser.eraser as er
from text_eraser.text_select import _deglow_full_green_v2, _geodesic_background

DET = dict(q_off=55.0, max_area_ratio=0.4, max_box_ratio=0.4,
           tint_fill=False, fill_white=True, fill_max_dist=12)

OUT = r"D:/Code/Project/Python/TextPatch/data/_pmparity"


def main(iid="1787767556635"):
    bgr = cv2.imread(rf"D:/Code/Project/Python/TextPatch/data/history/{iid}/orig.bin",
                     cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tmask, _ = er.detect_text_mask(rgb, method="ml", **DET)
    clean, core, dbg = _deglow_full_green_v2(
        rgb, tmask, strength=1.15, zone_ratio=0.6, zone_expand=10,
        protect_px=1, deglow_chroma_keep=True, debug=True)
    zone = dbg["zone"]
    greenness = dbg["greenness"]
    h, w = zone.shape
    n = h * w
    k3 = np.ones((3, 3), np.uint8)
    dout = cv2.distanceTransform((~zone).astype(np.uint8), cv2.DIST_L2, 5)
    ring = ((~zone) & (dout >= 10.0) & (dout <= 26.0) & (greenness <= 6))
    print(f"h,w={h}x{w} zone={int(zone.sum())} ring={int(ring.sum())}")

    r16 = rgb[..., 0].astype(np.int16)
    g16 = rgb[..., 1].astype(np.int16)
    b16 = rgb[..., 2].astype(np.int16)
    geo_mask = cv2.erode(zone.astype(np.uint8), k3, iterations=3) > 0
    B, (D_rg, D_gb) = _geodesic_background(
        rgb, geo_mask,
        extra=[(r16 - g16).astype(np.float32), (g16 - b16).astype(np.float32)],
        extra_src=ring)

    with open(rf"{OUT}/deglow_case.bin", "wb") as f:
        f.write(np.asarray([h, w], np.int32).tobytes())
        f.write(np.ascontiguousarray(rgb, np.float32).tobytes())
        f.write(zone.astype(np.uint8).tobytes())
        f.write(ring.astype(np.uint8).tobytes())
    for name, arr in [("B", B), ("D_rg", D_rg), ("D_gb", D_gb)]:
        np.save(rf"{OUT}/deglow_cv2_{name}.npy", arr)
    print("dumped deglow_case.bin + cv2 refs")


if __name__ == "__main__":
    main(*sys.argv[1:])
