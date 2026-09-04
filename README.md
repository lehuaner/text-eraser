# TextEraser

[![CI](https://github.com/lehuaner/text-eraser/actions/workflows/ci.yml/badge.svg)](https://github.com/lehuaner/text-eraser/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/text-eraser)](https://pypi.org/project/text-eraser/)
[![Python](https://img.shields.io/pypi/pyversions/text-eraser)](https://pypi.org/project/text-eraser/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**图片文字擦除工具** — DBNet 文字检测 + PatchMatch 内容识别填充，附带去发光（去除绿光/辉光文字的光晕）与本地 Web 界面。
*Image text removal via DBNet text detection + PatchMatch-based content-aware fill, with glow removal and a local web UI.*

![demo](docs/assets/demo.png)

## 特性

- **一键擦除文字**：DBNet（PP-OCRv4 det ONNX，约 5MB，CPU 推理）定位文字框，框内 Otsu 生成逐像素字形蒙版，PatchMatch 算法用周围背景填充
- **去发光（deglow）**：绿光/辉光文字的光晕污染背景，普通填充会残留绿斑——v2 方案用 alpha 分解 + 调和场插值恢复干净背景
- **一份算法核心（wasm）**：去发光/蒙版修复/填充全部编译为一个 `textcore.wasm`（245KB），后端经 wasmtime、浏览器经 WebAssembly 调**同一份字节码**，结果逐字节一致
- **引擎自由选择**：后端引擎（Python 库调用）或浏览器引擎（ESM 包，纯前端本地计算），Web 界面一键切换；深度自定义可直接取共享核算子自建管线
- **所见即所得**：「移动边缘」让展示蒙版与真实填充区完全一致，支持自动判定边缘外扩量
- **纯本地运行**：无 API、无上传，模型首次运行自动下载后完全离线
- **批量并发**：`erase_batch()` 多图并行（线程本地 wasm 实例隔离，实测近线性加速）

## 安装

```bash
# 库调用: cv2 (OpenCV) 由使用方环境提供, 先自选其一安装
pip install opencv-python        # 桌面/有 GUI; 服务器可选 opencv-python-headless
pip install text-eraser          # 纯填充/经典检测即装即用

# 需要 DBNet (ML) 文字检测再装可选依赖
pip install "text-eraser[ml]"

# 本地 Web 界面 (fastapi + uvicorn + DBNet 检测一次装全)
pip install "text-eraser[web,ml]"
```

> **⚠️ cv2 不要混装**：`opencv-python` 与 `opencv-python-headless` 是同一个 `cv2`
> 命名空间，同一环境里两个都装会互相覆盖 site-packages/cv2 文件（OpenCV 官方禁止）。
> 本包对二者任选其一均可，但**绝不会替你安装 cv2**——用哪种由你的环境决定。

需要 Python 3.10+。DBNet 模型（约 5MB）在首次使用时自动从 HuggingFace 下载，之后离线可用。

> **0.3.0 破坏性变更（Python 核心剔除）**：算法核心只保留 `textcore.wasm`（随包分发，
> `wasmtime` 成为硬依赖）。已删除：`glow_mode` 通道法及 auto/autov1.1/deglow_first/v4
> 等去发光变体（仅存 "v2"/"off"）、numpy PatchMatch 参考实现、`TEXTCORE_BACKEND=0`
> 调试开关。`deglow_green_thr/deglow_range/deglow_glo/deglow_protect` 参数保留但不再
> 生效（v2 不使用），旧调用方可平滑升级。Web 界面无感（原本就只有 v2/off）。

> **0.2.0 更名迁移**：Python 导入名由 `textpatch` 更名为 `text_eraser`（与 PyPI 包名
> `text-eraser` 对应），其余 API 完全不变，只需改导入前缀：
>
> ```diff
> - from textpatch import erase_text
> + from text_eraser import erase_text
> ```
>
> 同时环境变量 `TEXTPATCH_MODEL_DIR` 更名 `TEXTERASER_MODEL_DIR`（旧名自动兼容），
> 模型缓存目录由 `~/.textpatch/models/det` 迁移到 `~/.text_eraser/models/det`
> （首次运行检测到旧目录模型会自动复制并提示，不会静默重新下载）。

从源码运行：

```bash
git clone https://github.com/lehuaner/text-eraser.git
cd text-eraser
pip install -e ".[web,ml,dev]"
```

## 快速上手

### Web 界面

```bash
text-eraser            # 或 python -m text_eraser
```

浏览器打开 <http://127.0.0.1:8765/>，拖入图片即可擦除；支持逐面板查看蒙版/去发光中间结果、调整参数、保留历史记录。

- 端口/地址：环境变量 `TEXTERASER_PORT`（默认 8765）、`TEXTERASER_HOST`（默认 127.0.0.1）
- 运行数据目录：`TEXTERASER_DATA_DIR`（仓库开发用 `data/`，pip 安装默认 `~/.text_eraser/data`）
- 模型缓存目录：`TEXTERASER_MODEL_DIR`（pip 安装默认 `~/.text_eraser/models`）
- 以上均兼容 0.1.x 旧名 `TEXT_ERASER_*`，新名优先

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
| `fill_white` | True | 临近纯白补全：把紧邻蒙版的亮白/抗锯齿残留并入蒙版。白字/高亮字**必须保持 True**——否则 Otsu 漏检的笔画残段（如"台"底横、"周"顶横）既残留在结果里，又会留在 patchmatch 取样区内被复制进填充区，导致填充明显偏白、与背景色差大 |
| `fill_max_dist` | 12 | 孤立纯白段最大吞并距离（px，0=关闭）。小字/细笔画建议 8~15 |
| `max_side` | 960 | DBNet 推理最长边，调大可提升小字召回 |

> ⚠️ 调用方注意：若外部项目只需换背景色/填充色与原图一致，请不要关闭 `fill_white`、
> 不要把 `fill_max_dist` 设为 0，也不要绕过 `erase_text` 直接用 `detect_text_mask`
> 自建填充——蒙版修复（`_fill_nearby_white` / `_fill_bright_near_mask` /
> 亮核吸收）都在 `erase_text` 编排内，跳过即复现"漏检笔画污染填充"问题。

完整函数级 API 见 [docs/ALGORITHM.md](docs/ALGORITHM.md)；版本历史见 [CHANGELOG.md](CHANGELOG.md)。

### 引擎选择与自定义管线

算法核心只有一份 `textcore.wasm`，跑在**哪里**由调用者决定——Web 界面只是参考
实现（右上角切换「后端计算 / 本地浏览器计算」），库层面同样一等公民：

| 引擎 | 用法 | 适用场景 |
|---|---|---|
| 后端（Python 进程） | `from text_eraser import erase_text` | 服务端批处理、无头脚本 |
| 浏览器（纯前端） | `import { erase, eraseTextGlyphs, inpaint } from 'text-eraser-browser'`（见仓库 `browser/`，图片不出设备） | 隐私敏感场景、离线页面 |
| 自定义编排 | `from text_eraser import core` 取共享核算子 | 只要其中一步 / 自己串管线 |

深度自定义示例——只借去发光、填充自己做：

```python
from text_eraser import core

clean, core_mask, zone = core.deglow_full_green_v2(rgb, tmask, strength=1.0)
# 之后随意: 塞回自己的修复器 / 导出中间图 / 接入别的管线
```

完整算子清单（与浏览器 cv-bridge 一一对应）：`rgb2gray / threshold_otsu /
dilate / erode / morphology_ex / connected_components(_with_stats) /
edt_to_nearest_zero / resize_* / grow_color_tint / deglow_full_green_v2 /
patchmatch_inpaint_fill / smooth_telea_full / erase_text_glyphs`。

### 并发

```python
from text_eraser import erase_batch
results = erase_batch(list_of_rgbs, workers=4)   # 多图并行, 返回与输入同序
```

- 每个线程持有**独立的 wasm 核实例**（wasmtime Store 非线程安全，单例并发会踩
  内存——包内已按线程隔离，Web 服务多请求并发同样受益）
- wasmtime 的 FFI 调用释放 GIL，实测多图填充接近线性加速
- 单张图内部不做并行：逐框拆分填充会改变结果、破坏前后端逐字节一致（有意取舍）

## 算法概览

```
原图 RGB
  → DBNet 文字框 → 框内 Otsu + 纯白补全 → 逐像素文字蒙版      [Python, onnxruntime]
  → 去发光 v2 (强绿信号门 → alpha 分解恢复背景 → 调和场插值)   [textcore.wasm]
  → 蒙版修复 (并集 → 闭运算 → 亮核吸收 → zone 吸收)           [textcore.wasm]
  → 移动边缘 → PatchMatch 填充 (Criminisi 优先级 + 颜色自适应；平滑渐变背景
    自动切 TELEA 扩散)                                        [textcore.wasm]
  → 擦除结果
```

除 DBNet 推理与检测链前处理（cv2 float 算子，位级复刻不可达）外，整条算法
管线都在共享 wasm 核内执行——浏览器引擎跑的是同一份字节码，两端结果逐字节一致。

- 设计细节与参数速查：[docs/ALGORITHM.md](docs/ALGORITHM.md)
- 共享核跨端一致性验证：`shared/_verify_align/`（py/node/browser 三端 md5 对照）
- 去发光 v4 规格说明：[docs/DEGLOW_V4.md](docs/DEGLOW_V4.md)（0.2.x 实验方案，已从
  主管线移除，模块保留于 `deglow/` 供研究）

## 已知限制

- **发光文字路径仍在迭代**：绿光/辉光场景经过多轮修复已大幅改善，但个别复杂背景（暖色多弧段等）仍可能有色差残留，持续按实际观感调整中
- 文字检测依赖 DBNet 召回；极端小字（<8px 高）或严重模糊的字体可能漏检
- 大图（4K+）CPU 推理约 0.1s/张，填充耗时与蒙版面积成正比

## 开发

```bash
# cv2 仍由环境提供 (自选 opencv-python 或 opencv-python-headless 其一)
pip install -e ".[dev,web,ml]"
pytest                 # 合成图测试套件
```

```
TextEraser/
├── text_eraser/          # 包本体 (检测编排/蒙版/填充/Web) — 算法核心在 assets/textcore.wasm
│   └── core.py           # 共享核算子公开门面 (自定义管线入口)
├── shared/               # 共享算法核 Rust 源码 + 三端 binding + 跨端校验
├── browser/              # 浏览器引擎 ESM 包 (text-eraser-browser)
├── tests/              # 合成图测试 (CI 用, 自足不依赖样图)
├── docs/
│   ├── ALGORITHM.md    # 函数级算法与参数指南
│   ├── DEGLOW_V4.md    # 去发光 v4 规格说明 (0.2.x 实验方案存档)
│   ├── assets/         # README 演示图
│   └── dev/            # 开发日志与专项修复报告
├── scripts/            # 离线诊断/回归脚本 (见下)
└── deglow/             # 去发光 v4 实验模块 (不随 pip 包发布)
```

`scripts/` 保留核心验收与诊断工具：`parity_check.py`（后端 vs 浏览器 wasm 逐字节
parity 验收门，4 基准图）、`render.py` / `pixel_check.py`（全管线对照与逐像素检查）、
`dbnet_*.py`（DBNet 诊断）、`_cmp_*` / `_gen_dist.py`（跨端算子对照）、
`make_release_demo.py`（发布演示图）、`fix_ref.py`（仓库 ref 修复工具）、
`v4_*.py`（deglow v4 研究，需完整仓库）。

## 发布

发新版本无需本地构建、无需任何 API token——推送 tag 即自动发布到 PyPI（Trusted Publishing / OIDC）：

```bash
# 1. 升版本号: pyproject.toml 的 version + text_eraser/__init__.py 的 __version__
# 2. 提交后打 tag 推送
git tag v0.3.0 && git push origin main v0.3.0
```

首次使用前需在 PyPI 一次性登记发布者：项目管理页 → Settings → Publishing，
填 Owner `lehuaner` / Repository `text-eraser` / Workflow `publish.yml`（Environment 留空）。

## License

[MIT](LICENSE)
