"""python -m text_eraser → 启动本地 TextEraser Web 界面 (http://127.0.0.1:8765/).

需先安装 web extra: pip install "text-eraser[web]" (缺失时会打印提示而非报错堆栈).
"""
from text_eraser.webapp import main

if __name__ == "__main__":
    main()
