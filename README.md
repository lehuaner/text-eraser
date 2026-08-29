# Text Eraser

[![CI](https://github.com/lehuaner/Text Eraser/actions/workflows/ci.yml/badge.svg)](https://github.com/lehuaner/Text Eraser/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/text-eraser)](https://pypi.org/project/text-eraser/)
[![Python](https://img.shields.io/pypi/pyversions/text-eraser)](https://pypi.org/project/text-eraser/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**图片文字擦除工具** — DBNet 文字检测 + PatchMatch 内容识别填充，附带去发光（去除绿光/辉光文字的光晕）与本地 Web 界面。
*Image text removal via DBNet text detection + PatchMatch-based content-aware fill, with glow removal and a local web UI.*

![demo](docs/assets/demo.png)

## 特性

- **一键擦除文字**：DBNet（PP-OCRv4 det ONNX，约 5MB，CPU 推理）定位文字框，框内 Otsu 生成逐像素字形蒙版，PatchMatch 算法用周围背景填充
- **去发光（deglow）**：绿光/辉光文字的光晕污染背景，普通填充会残留绿斑——v2 方案用 alpha 分解 + 调和场插值恢复干净背景
- **所见即所得**：「移动边缘」让展示蒙版与真实填充区完全一致，支持自动判定边缘外扩量
- **纯本地运行**：无 API、无上传，模型首次运行自动下载后完全离线
- **两种用法**：本地 Web 界面（拖图即擦）或 Python 库调用（`erase_text()` 一步出结果）

## 安装

```bash
pip install text-eraser
```

需要 Python 3.10+。DBNet 模型（约 5MB）在首次使用时自动从 HuggingFace 下载，之后离线可用。

> PyPI 发行名为 `text-eraser`（曾用名 `textpatch` 与既有包名过于相似未获准），Python 导入名是 `text_eraser`。

从源码运行：

```bash
git clone https://github.com/lehuaner/Text Eraser.git
cd Text Eraser
pip install -e .
```

## 快速上手

### Web 界面

```bash
text-eraser            # 或 python -m text_eraser
```

浏览器打开 <http://127.0.0.1:8765/>，拖入图片即可擦除；支持逐面板查看蒙版/去发光中间结果、调整参数、保留历史记录。

- 端口/地址：环境变量 `TEXT_ERASER_PORT`（默认 8765）、`TEXT_ERASER_HOST`（默认 127.0.0.1）
- 运行数据目录：`TEXT_ERASER_DATA_DIR`（仓库开发用 `data/`，pip 安装默认 `~/.text_eraser/data`）
- 模型缓存目录：`TEXT_ERASER_MODEL_DIR`（pip 安装默认 `~/.text_eraser/models`）

### Python 库调用

```python
from PIL import Image
from text_eraser import erase_text, to_rgb_uint8

rgb = to_rgb_uint8(Image.open("demo.png"))       # HxWx3 uint8 RGB
result, mask, meta = erase_text(rgb, return_mask=True)
Image.fromarray(result).save("out.png")
# mask: 255=被擦除的文字; meta: mask_pix / inpaint_seconds / edge_used 等
```

只用填充器（去水印/去任意内容，无需文字检测）：

```python
import numpy as np
from text_eraser import inpaint

hole = np.zeros(rgb.shape[:2], np.uint8)   # >0 = 要清除的区域
hole[10:40, 20:80] = 255
filled = inpaint(rgb, hole, sample_mask=255 - hole)
```

### 主要参数（`erase_text`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `edge` | 1 | 「移动边缘」：蒙版与填充区同步外扩(>0)/收缩(<0)像素 |
| `auto_edge` | True | 按文字色残留自动判定最小外扩（多数图 1，硬图自动到 2） |
| `q_off` | 55 | 蒙版紧密度 [30,70]，越高越贴字形 |
| `direction` | None | 纹理方向角度°（木纹/条带类背景） |
| `deglow_scheme` | "v2" | 去发光方案："v2" / "off"（无发光图自动零改动） |
| `max_side` | 960 | DBNet 推理最长边，调大可提升小字召回 |

完整函数级 API 见 [docs/ALGORITHM.md](docs/ALGORITHM.md)。

## 算法概览

```
原图 RGB
  → DBNet 文字框 → 框内 Otsu + 纯白补全 → 逐像素文字蒙版
  → 去发光 v2 (强绿信号门 → alpha 分解恢复背景 → 调和场插值)
  → 移动边缘 → PatchMatch 填充 (Criminisi 优先级 + 颜色自适应 + TELEA 兜底)
  → 擦除结果
```

- 设计细节与参数速查：[docs/ALGORITHM.md](docs/ALGORITHM.md)
- 去发光 v4 规格说明：[docs/DEGLOW_V4.md](docs/DEGLOW_V4.md)

## 已知限制

- **发光文字路径仍在迭代**：绿光/辉光场景经过多轮修复已大幅改善，但个别复杂背景（暖色多弧段等）仍可能有色差残留，持续按实际观感调整中
- 文字检测依赖 DBNet 召回；极端小字（<8px 高）或严重模糊的字体可能漏检
- 大图（4K+）CPU 推理约 0.1s/张，填充耗时与蒙版面积成正比

## 开发

```bash
pip install -e .[dev]
pytest                 # 合成图测试套件
```

```
Text Eraser/
├── text_eraser/          # 包本体 (检测/蒙版/填充/去发光/Web)
├── tests/              # 合成图测试 (CI 用, 自足不依赖样图)
├── docs/
│   ├── ALGORITHM.md    # 函数级算法与参数指南
│   ├── DEGLOW_V4.md    # 去发光 v4 规格说明
│   ├── assets/         # README 演示图
│   └── dev/            # 开发日志与专项修复报告
├── scripts/            # 离线诊断/回归脚本 (见下)
└── deglow/             # 去发光 v4 实验模块 (不随 pip 包发布)
```

`scripts/` 下还有一套基于真实样图的回归脚本（`regress_*.py`，样图不入库，需本地自备 `data/` 样图），用于算法调参时的逐位回归验证。

## 发布

发新版本无需本地构建、无需任何 API token——推送 tag 即自动发布到 PyPI（Trusted Publishing / OIDC）：

```bash
# 1. 升版本号: pyproject.toml 的 version + text_eraser/__init__.py 的 __version__
# 2. 提交后打 tag 推送
git tag v0.1.2 && git push origin main v0.1.2
```

首次使用前需在 PyPI 一次性登记发布者：项目管理页 → Settings → Publishing，
填 Owner `lehuaner` / Repository `text-eraser` / Workflow `publish.yml`（Environment 留空）。

## License

[MIT](LICENSE)
