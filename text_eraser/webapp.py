"""Web 界面入口：先做依赖可用性检查，再懒加载实际实现 (_webapp_impl)。

把 fastapi/uvicorn 的导入推迟到本模块之外，使得 text-eraser 命令与
`python -m text_eraser` 在未安装 web extra 时打印安装提示后正常退出，
而不是抛出裸 ImportError 堆栈。`uvicorn text_eraser.webapp:app` 仍可用。
"""
from __future__ import annotations

import sys

_WEB_HINT = (
    'Web 界面需要：pip install "text-eraser[web]"\n'
    '(DBNet 文字检测一并安装: pip install "text-eraser[web,ml]")'
)


def _require_web_deps() -> None:
    """导入 fastapi/uvicorn 前的可用性检查; 缺失时打印提示并以退出码 1 结束。"""
    from importlib.util import find_spec

    missing = [name for name in ("fastapi", "uvicorn") if find_spec(name) is None]
    if missing:
        print(_WEB_HINT, file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    """命令行入口: text-eraser / python -m text_eraser。"""
    _require_web_deps()
    from text_eraser._webapp_impl import main as _impl_main

    _impl_main()


def __getattr__(name: str):
    """模块级懒加载: 支持 `uvicorn text_eraser.webapp:app`（首次访问才导入 fastapi）。"""
    if name == "app":
        try:
            from text_eraser._webapp_impl import app
        except ImportError:
            raise RuntimeError(_WEB_HINT) from None
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
