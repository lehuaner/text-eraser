# Text Eraser

基于 **PatchMatch + DBNet** 的图片文字擦除工具，独立 Web 前端项目。

## 启动

**Windows（推荐双击）：**
```cmd
run.bat
```
然后浏览器打开 <http://127.0.0.1:8765/>

**Mac / Linux：**
```bash
./run.sh
```

**手动启动（任意平台）：**
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

需要 Python 3.10+，依赖见 `requirements.txt`。

## 项目结构

```
TextPatch/
├── app/main.py             # FastAPI 后端入口
├── core/
│   ├── patch_fill.py        # PatchMatch 内容识别填充（项目自带）
│   ├── text_select.py       # 文字检测 + 逐像素文字蒙版
│   ├── ml_text_select.py    # DBNet (PP-OCRv4 det) 推理
│   ├── eraser.py            # 一站式擦除流水线
│   └── models/det/*.onnx    # DBNet 模型 (4.5MB,首次启动会用缓存)
├── static/                  # 前端 (HTML/CSS/JS,无框架)
├── scripts/                 # 离线工具 (dryrun, debug, render)
├── data/                    # 示例图与测试输出
└── run.bat / run.sh         # 双击启动脚本
```

## 工作流

```
原图 RGB
  ↓ (DBNet)
文字框
  ↓ (Lab + Otsu + Close + 形态学)
逐像素文字蒙版
  ↓ (预外扩 4px)
扩边蒙版
  ↓ (sample_mask = 整图 − 扩边蒙版)
patch_fill.inpaint()        ← 强制只在非文字区取样
  ↓
force_color_match()         ← LAB 空间线性变换到周围 ring
  ↓ (轻量)
TELEA inpaint(3)            ← 边缘柔化兜底
  ↓
force_color_match()         ← 收尾
  ↓
擦除结果
```

## 算法来源

`core/patch_fill.py`、`core/text_select.py`、`core/ml_text_select.py` 从同环境的另一个项目（ExtractRole）集成进来，按本项目需求做了一处改动：`text_select.py` 与 `ml_text_select.py` 的 `from core.extractor import to_rgb_uint8` 改为 `from core.text_select import to_rgb_uint8`（在本项目内 `text_select.py` 自带 `to_rgb_uint8`），让本项目**完全不依赖**原项目。

其余文件（`core/eraser.py`、`app/main.py`、`static/*`、`scripts/*`）为本项目独有。
