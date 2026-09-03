"""M0 · 检测器确定性契约测试（规格 §8 R1 / M0 验收门）。

同输入、同方法（ml，DBNet ONNX 静态图 + 固定 padding）重复 n 次，
断言 mask 与 boxes **逐位一致**（100 次）。检测器版本/尺寸漂移由
本契约 + ONNX pin 兜底。
"""
from __future__ import annotations

import time

import cv2
import numpy as np


def make_contract_image(size: int = 512) -> np.ndarray:
    """固定合成文字图（与 GT 无关，纯契约输入）。"""
    img = np.full((size, size, 3), 200, np.uint8)
    cv2.putText(img, "CONTRACT-100x", (40, int(size * 0.5)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (30, 30, 40), 3, cv2.LINE_AA)
    cv2.putText(img, "DEGLOW-DET", (70, int(size * 0.72)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 60, 30), 2, cv2.LINE_AA)
    return img


def test_det_contract(rgb: np.ndarray | None = None, n: int = 100,
                      method: str = "ml", verbose: bool = True) -> bool:
    """同输入重复 n 次检测，断言逐位一致。返回是否通过。"""
    from core.text_select import detect_text_mask
    rgb = rgb if rgb is not None else make_contract_image()
    t0 = time.time()
    first = None
    consistent = True
    for i in range(n):
        mask, boxes = detect_text_mask(rgb, method=method, max_side=960)
        mask = np.asarray(mask, np.uint8)
        if first is None:
            first = (mask, boxes)
            continue
        if not np.array_equal(mask, first[0]) or boxes != first[1]:
            consistent = False
            if verbose:
                print(f"  [det contract] 第 {i + 1} 次不一致！")
            break
    dt = time.time() - t0
    if verbose:
        n_run = n - 1 if first is not None else 0
        print(f"[det contract] {n} 次 detect_text_mask(method={method}) "
              f"逐位一致: {'PASS' if consistent else 'FAIL'} "
              f"({dt:.1f}s, {n_run} 次比较)")
    return consistent
