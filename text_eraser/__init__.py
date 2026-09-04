"""TextEraser — 基于 DBNet 文字检测 + PatchMatch 内容识别填充的图片文字擦除工具。

0.3.0 起算法核心只有一份实现：**textcore.wasm 共享核**（后端经 wasmtime、
浏览器经 WebAssembly 调同一份字节码，逐字节一致）。Python/cv2 核心实现已删除。

快速上手（库调用，后端引擎）::

    from text_eraser import erase_text, to_rgb_uint8
    from PIL import Image

    rgb = to_rgb_uint8(Image.open("demo.png"))
    result, mask, meta = erase_text(rgb, return_mask=True)

算法执行位置（引擎选择）:
  - 后端引擎: 上面这样直接调 ``erase_text``（Python 进程内跑 wasm）;
  - 浏览器引擎: ESM 包 ``text-eraser-browser``（见仓库 browser/ 目录），
    ``erase()`` / ``eraseTextGlyphs()`` / ``inpaint()`` 全在用户浏览器本地跑;
  - 深度自定义: ``from text_eraser import core`` 取共享核算子自由编排
    （deglow_full_green_v2 / patchmatch_inpaint_fill / erase_text_glyphs / …）。

cv2 (OpenCV) 由使用方环境提供 (opencv-python / opencv-python-headless 二选一,
严禁同环境混装; 仅 DBNet 检测链需要); DBNet 文字检测另装可选依赖:
pip install "text-eraser[ml]"。

启动 Web 界面（需先安装 web extra: pip install "text-eraser[web]"）::

    python -m text_eraser          # http://127.0.0.1:8765/
"""
from text_eraser import core as core  # noqa: F401  (自定义管线门面, 见 core.__doc__)
from text_eraser.eraser import erase_batch, erase_text
from text_eraser.ml_text_select import detect_text_ml, ensure_model, is_model_available
from text_eraser.patch_fill import inpaint
from text_eraser.text_select import detect_text, detect_text_mask, to_rgb_uint8

__version__ = "0.3.1"

__all__ = [
    "erase_text",
    "erase_batch",
    "inpaint",
    "detect_text",
    "detect_text_mask",
    "detect_text_ml",
    "to_rgb_uint8",
    "ensure_model",
    "is_model_available",
    "core",
    "__version__",
]
