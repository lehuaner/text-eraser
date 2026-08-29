"""
文字模式 - 轻量 ML 引擎 (DBNet, PP-OCRv4 det).

与 text_select.py 的经典 CV 检测接口一致, 输出 [{x0,y0,x1,y1}] (原图坐标).
首次调用时若模型不存在, 自动从 HuggingFace 下载 (~5MB, 仓库 checkout 落在
textpatch/models/det/, pip 安装落在 ~/.textpatch/models/det/).

为何选这个模型:
  * PP-OCRv4 det = DBNet++ + MobileNetV3 骨干, Apache-2.0;
  * 检测阶段语言无关 (中英日韩皆可), 不依赖识别器;
  * ONNX 4.7MB, CPU 推理 0.08s/图 (4096x2160), 满足"轻量"要求.

依赖: onnxruntime + cv2 + numpy (见 pyproject/requirements).
"""
from __future__ import annotations

import os
import ssl
import threading
import urllib.request
from typing import Optional

import cv2
import numpy as np

# 延迟导入 onnxruntime, 避免 import 即加载; 也让经典 CV 路径不被拖累
_ort = None

# ---------------------------------------------------------------------------
# 模型路径与下载
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))


def _default_model_dir() -> str:
    """仓库 checkout 用包内 models/det（保留已下载模型）; pip 安装落到用户目录
    （site-packages 未必可写）。可用环境变量 TEXTPATCH_MODEL_DIR 覆盖。"""
    env = os.environ.get("TEXTPATCH_MODEL_DIR")
    if env:
        return env
    if os.path.isdir(os.path.join(os.path.dirname(_HERE), "data")):
        return os.path.join(_HERE, "models", "det")
    return os.path.join(os.path.expanduser("~"), ".textpatch", "models", "det")


MODEL_DIR = _default_model_dir()
MODEL_PATH = os.path.join(MODEL_DIR, "ch_PP-OCRv4_det.onnx")
# HuggingFace Heliosoph/paddleocr-v4-det-onnx (Apache-2.0)
MODEL_URL = (
    "https://huggingface.co/Heliosoph/paddleocr-v4-det-onnx/"
    "resolve/main/ch_PP-OCRv4_det.onnx"
)

# PaddleOCR det 标准化参数 (与上游预处理完全一致; BGR 通道顺序, 因为 cv2 读入即 BGR)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 会话缓存 (进程内单例)
_SESSION = None
_SESSION_LOCK = threading.Lock()

# 下载状态/锁 (防止并发触发多次下载)
_DOWNLOAD_LOCK = threading.Lock()
_DOWNLOAD_DONE = False  # 仅作"已成功完成一次"标记


def is_model_available() -> bool:
    """模型文件是否已在本地. 用于前端/端点快速判断是否需要走下载路径."""
    return os.path.isfile(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000


def get_model_path() -> str:
    """若模型已存在直接返回路径, 否则阻塞下载 (~3s, 5MB). 线程安全."""
    if is_model_available():
        return MODEL_PATH
    ensure_model()
    return MODEL_PATH


def ensure_model() -> str:
    """保证模型在本地. 线程安全; 并发调用不会重复下载. 失败抛 RuntimeError."""
    global _DOWNLOAD_DONE
    if is_model_available():
        return MODEL_PATH
    with _DOWNLOAD_LOCK:
        if is_model_available():
            return MODEL_PATH
        os.makedirs(MODEL_DIR, exist_ok=True)
        ctx = ssl.create_default_context()
        req = urllib.request.Request(
            MODEL_URL, headers={"User-Agent": "textpatch/" + __import__("textpatch").__version__}
        )
        tmp = MODEL_PATH + ".part"
        try:
            with urllib.request.urlopen(req, timeout=120, context=ctx) as r, \
                    open(tmp, "wb") as f:
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
            # 落盘后验证大小再改名, 避免半截文件被当成"已存在"
            if os.path.getsize(tmp) < 1_000_000:
                raise RuntimeError(
                    "downloaded model looks too small ({} bytes)".format(
                        os.path.getsize(tmp)))
            os.replace(tmp, MODEL_PATH)
            _DOWNLOAD_DONE = True
        except Exception:
            # 清理残片
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise
    return MODEL_PATH


def _get_session():
    """懒加载 onnxruntime InferenceSession, 进程内单例. 线程安全."""
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None:
            return _SESSION
        import onnxruntime as ort  # 局部导入
        path = get_model_path()
        # 本机 onnxruntime 是 CPU 版, 无 CUDA EP; 直接用默认 providers
        providers = ort.get_available_providers() or ["CPUExecutionProvider"]
        sess = ort.InferenceSession(path, providers=providers)
        _SESSION = sess
        return _SESSION


# ---------------------------------------------------------------------------
# 主入口: detect_text_ml
# ---------------------------------------------------------------------------
def _dbnet_infer(rgb, strength, box_threshold, max_side):
    """DBNet(PP-OCRv4 det) 推理核心：返回 (prob, nw, nh, H, W, thr)。
    供 detect_text_ml(取框) 与 detect_text_mask_ml(取蒙版) 复用，避免重复推理。"""
    H, W = rgb.shape[:2]
    s = float(np.clip(strength, 0, 1))
    # strength 越高 -> 阈值越低 -> 更多弱响应被保留
    thr = float(np.clip(box_threshold - 0.15 * s, 0.1, 0.9))

    # 1) 尺度归一: 32 的倍数, 最长边 <= max_side
    scale = min(max_side / max(H, W), 1.0)
    nh = max(32, (int(round(H * scale)) // 32) * 32)
    nw = max(32, (int(round(W * scale)) // 32) * 32)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    rimg = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    # 2) 预处理: 与 PaddleOCR det 一致 (BGR /255 - mean) / std, NCHW float32
    blob = rimg.astype(np.float32) / np.float32(255.0)
    blob = (blob - _MEAN) / _STD
    blob = np.transpose(blob, (2, 0, 1))[None, ...].astype(np.float32)

    # 3) 推理 -> 概率图
    sess = _get_session()
    in_name = sess.get_inputs()[0].name
    out = sess.run(None, {in_name: blob})
    prob = out[0][0, 0]  # HxW float32, 与 rimg 同分辨率
    return prob, nw, nh, H, W, thr


def detect_text_ml(
    raw: np.ndarray | "Image.Image",
    strength: float = 1.0,
    min_area: int = 30,
    max_area_ratio: float = 0.05,
    max_box_ratio: float = 0.20,
    box_threshold: float = 0.3,
    max_side: int = 960,
    pad: int = 3,
) -> list[dict]:
    """
    轻量 ML 文字检测 (DBNet / PP-OCRv4 det ONNX).

    返回: 原图坐标的 [{x0,y0,x1,y1}, ...]. 未检测到返回 [].

    参数:
      strength: 0~1, 用户"检测灵敏度"滑块. 越高 -> 阈值越低 (更灵敏, 可能多框).
      min_area: 概率图上连通域最小面积 (工作尺度), <此丢弃.
      max_area_ratio: 单块超过工作图比例 -> 丢弃 (大色块/面板).
      max_box_ratio: 最终框超过原图比例 -> 丢弃 (兜底防全图框).
      box_threshold: 概率阈值 (0.1~0.9), 实际阈值 = box_threshold - 0.15*strength.
      max_side: resize 后最长边像素 (32 的倍数). 越大越慢但小字召回越好.
      pad: 框外扩像素 (原图坐标).
    """
    from textpatch.text_select import to_rgb_uint8

    rgb = to_rgb_uint8(raw)
    H, W = rgb.shape[:2]
    total = H * W
    if total == 0:
        return []

    prob, nw, nh, H, W, thr = _dbnet_infer(rgb, strength, box_threshold, max_side)

    # 4) 后处理: 阈值 + 轻度膨胀 (合笔画断口) + 外轮廓 + boundingRect
    binmask = (prob > thr).astype(np.uint8) * 255
    # 轻度膨胀: 防止单字笔画在概率图上被阈值"咬断" -> 一个字被切成 2 个
    binmask = cv2.dilate(
        binmask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    cnts, _ = cv2.findContours(
        binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    invW = W / nw
    invH = H / nh
    wTotal = nw * nh
    boxes: list[dict] = []
    for c in cnts:
        a = float(cv2.contourArea(c))
        if a < min_area:
            continue
        if a > wTotal * max_area_ratio:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            continue
        # 缩放回原图坐标
        X0 = int(round(x * invW))
        Y0 = int(round(y * invH))
        X1 = int(round((x + w) * invW))
        Y1 = int(round((y + h) * invH))
        # 外扩
        X0 = max(0, X0 - pad)
        Y0 = max(0, Y0 - pad)
        X1 = min(W - 1, X1 + pad)
        Y1 = min(H - 1, Y1 + pad)
        if X1 - X0 <= 1 or Y1 - Y0 <= 1:
            continue
        if (X1 - X0) * (Y1 - Y0) > total * max_box_ratio:
            continue
        boxes.append({"x0": X0, "y0": Y0, "x1": X1, "y1": Y1})
    return boxes


def detect_text_mask_ml(
    raw: np.ndarray | "Image.Image",
    strength: float = 1.0,
    min_area: int = 30,
    max_area_ratio: float = 0.05,
    box_threshold: float = 0.3,
    max_side: int = 960,
    mask_threshold: float = 0.4,    # 蒙版专属阈值：明显高于框检测阈值，只取字形实体、保留字符间隙
    mask_max_side: int = 1600,      # 蒙版用更高分辨率，避免低分辨率下 3x3 膨胀把相邻字连成团
):
    """
    轻量 ML 文字「边缘蒙版」检测 (DBNet 概率图, 高分辨率 + 高阈值)。

    返回 (mask, boxes):
      mask : HxW uint8, 255=文字像素（逐像素字形，非整框）
      boxes: 显示用文字框列表（原图坐标）

    与 detect_text_ml(只返回整框) 的区别：这里给出**逐像素**文字蒙版，
    使 patchmode 只填充字形本身、参考区自动取文字之外的全部纹理。

    关键改进（修复此前"蒙版糊成一团、无字符边界"）：
      * 在更高分辨率(mask_max_side)推理 -> 字符间隙在低分辨率下被吞掉的问题消失；
      * 用更高阈值(mask_threshold)取概率核心 -> 排除 halo 与字符粘连；
      * 仅做 2x2 极小闭运算修复笔画内断口, 不复用框检测的 3x3 膨胀(那会连字)。
    """
    from textpatch.text_select import to_rgb_uint8
    from textpatch import text_select as _ts

    rgb = to_rgb_uint8(raw)
    H, W = rgb.shape[:2]
    total = H * W
    if total == 0:
        return np.zeros((H, W), np.uint8), []

    s = float(np.clip(strength, 0, 1))
    # 蒙版阈值: 灵敏度越高 -> 略低(多召回), 但始终明显高于框检测阈值以保留间隙
    mthr = float(np.clip(mask_threshold - 0.12 * s, 0.25, 0.85))

    # 高分辨率推理（与框检测的 max_side 解耦）
    prob, nw, nh, H, W, _ = _dbnet_infer(rgb, strength, mask_threshold, mask_max_side)

    # 仅取高概率核心（字形实体），排除 halo 与字符间粘连
    core = (prob > mthr).astype(np.uint8) * 255
    # 极小闭运算：修复笔画内细小断口（2x2 在高分辨率下仅约 1 原图像素，不会连字）
    core = cv2.morphologyEx(
        core, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )
    # 上采样回原图分辨率（双线性后阈值，得到平滑字形边缘；间隙处概率低 -> 自然留空）
    up = cv2.resize(core, (W, H), interpolation=cv2.INTER_LINEAR)
    mask = (up > 127).astype(np.uint8) * 255
    # 连通域清理：保留字形大组件（蒙版后会与用户文字框求交，大块无妨）；
    # 仅去掉极小噪点 / 过细过粗(长UI线)组件。故放宽面积上限(max_area_ratio 提到 0.9)。
    mask = _ts._clean_text_mask(mask, H, W, min_area=min_area, max_area_ratio=0.9)
    boxes = _ts._mask_to_boxes(mask)
    return mask, boxes
