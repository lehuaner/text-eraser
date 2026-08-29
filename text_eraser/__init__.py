"""Text Eraser — 基于 DBNet 文字检测 + PatchMatch 内容识别填充的图片文字擦除工具。

快速上手（库调用）::

    from text_eraser import erase_text, to_rgb_uint8
    from PIL import Image

    rgb = to_rgb_uint8(Image.open("demo.png"))
    result, mask, meta = erase_text(rgb, return_mask=True)

启动 Web 界面::

    python -m text_eraser          # http://127.0.0.1:8765/
"""
from text_eraser.eraser import erase_text
from text_eraser.ml_text_select import detect_text_ml, ensure_model, is_model_available
from text_eraser.patch_fill import inpaint
from text_eraser.text_select import detect_text, detect_text_mask, to_rgb_uint8

__version__ = "0.1.1"

__all__ = [
    "erase_text",
    "inpaint",
    "detect_text",
    "detect_text_mask",
    "detect_text_ml",
    "to_rgb_uint8",
    "ensure_model",
    "is_model_available",
    "__version__",
]
