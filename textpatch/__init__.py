"""TextPatch — 基于 DBNet 文字检测 + PatchMatch 内容识别填充的图片文字擦除工具。

快速上手（库调用）::

    from textpatch import erase_text, to_rgb_uint8
    from PIL import Image

    rgb = to_rgb_uint8(Image.open("demo.png"))
    result, mask, meta = erase_text(rgb, return_mask=True)

启动 Web 界面::

    python -m textpatch          # http://127.0.0.1:8765/
"""
from textpatch.eraser import erase_text
from textpatch.ml_text_select import detect_text_ml, ensure_model, is_model_available
from textpatch.patch_fill import inpaint
from textpatch.text_select import detect_text, detect_text_mask, to_rgb_uint8

__version__ = "0.1.0"

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
