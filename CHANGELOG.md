# Changelog

本文件记录 text-eraser 的显著变更。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [0.3.0] — 2026-09-04

相比 0.2.1 的架构级重写：**算法核心从「Python + cv2 多套实现」收敛为「一份 textcore.wasm 共享核」**，后端与浏览器调同一份字节码，结果逐字节一致。

### 架构重写

- **共享算法核 `textcore.wasm`（新增 `shared/`，Rust→WebAssembly，约 245KB）**：
  去发光（v2 减绿度）、蒙版修复（并集/闭运算/亮核吸收/zone 吸收）、PatchMatch 填充、
  平滑渐变 TELEA、形态学/连通域/灰度/Otsu/精确 EDT/resize 全部进核。
  Python(wasmtime)、Node、浏览器三端对同输入 md5 逐字节一致（harness：`shared/_verify_align/`）。
- **剔除 Python 核心算法（破坏性）**：numpy PatchMatch 参考实现、通道法去发光
  （auto / autov1.1 / deglow_first）、v4 实验方案入口、`TEXTCORE_BACKEND=0` 调试开关
  全部删除。核心加载失败快速抛 `CoreLoadError`，**不再静默降级**（静默降级会重新引入
  前后端分歧）。保留的 cv2 仅用于 DBNet 检测链（float 算子位级复刻不可达）与画笔栅格化。
- **wasm 随包分发**：`textcore.wasm` 打包进 wheel（`text_eraser/assets/`），
  pip 安装即用；`wasmtime>=25` 成为硬依赖。
  （0.2.x 的 wheel 不含共享核——核心算法仅存在于源码仓库。）

### 新增

- **引擎自由选择**（Web 界面只是 demo）：
  - 后端引擎：`from text_eraser import erase_text`（Python 进程内跑 wasm）；
  - 浏览器引擎：ESM 包 `text-eraser-browser`（新增 `browser/` 目录，Web Worker +
    onnxruntime-web + opencv.js，图片不出设备），`erase()` / `eraseTextGlyphs()` / `inpaint()`；
  - 深度自定义：`text_eraser/core` 公开门面直接暴露共享核算子
    （`deglow_full_green_v2` / `erase_text_glyphs` / `patchmatch_inpaint_fill` /
    `smooth_telea_full` / `grow_color_tint` / 形态学 / 连通域 / Otsu / EDT / resize…），
    调用者可自由编排自己的管线。
- **批量并发 `erase_batch()`**：多图并行；每线程持有独立 wasm 核实例
  （wasmtime Store 非线程安全），wasmtime FFI 释放 GIL，实测 2 线程约 1.7x 加速；
  Web 服务多请求并发同样受益。`auto_edge` 与 v2 管线共用一次 DBNet 检测
  （`tmask_hint`），大图省 1/3 检测耗时。
- **Web 界面「本地浏览器计算」模式**：纯前端跑完整管线（DBNet→去发光→填充），
  附加跨源隔离中间件 + 浏览器资源自愈构建（`_browser_assets` 自动准备
  opencv.js / onnxruntime-web / te-bundle）。
- **验证工具链**：`scripts/parity_check.py`（后端 vs 浏览器 wasm 逐字节 parity 验收门，
  4 基准图 RESULT/MASK diff=0）、`text_eraser/test_wasm_erase_smoke.py`、
  `text_eraser/test_shared_core_smoke.py`（wasm vs cv2 算子逐位对照）。

### 性能

- deglow wasm 形态学前缀和快路径（O(k·n)），369×231 图 410→240ms；
- patchmatch SSD 早退 + harmonic runs 向量化 + resize scratch 复用；
- detect 链优化：`edge_aware` / `soft_expand` / `grow_color_tint`（原 120 轮
  Python↔wasm 往返）移植进共享核，一次调用完成；`_cv` shim 把 dilate/erode/
  morphologyEx/连通域/灰度路由到 wasm，检测链与浏览器共享同一份算子。

### 修复

- JS binding `eraseTextGlyphs` 在 `tmask2=null` 时未清零缓冲——wasm `alloc` 不清零，
  残留数据被当作第二检测蒙版并集导致结果错误（`textcore.js` 与
  `textcore.browser.js` 同修；浏览器内建流程恒传非 null 故未暴露，自定义调用传 null 必踩）。
- `_webapp_impl` 历史记录处 `glow_mode` 残留引用（会 NameError）。

### 破坏性变更与迁移

| 0.2.x | 0.3.0 |
|---|---|
| `erase_text(..., glow_mode="auto")` | 参数删除（通道法已移除） |
| `deglow_scheme="channel"/"v4"` | 仅支持 `"v2"`（默认）/ `"off"`，其余抛 `ValueError` |
| `deglow_green_thr/range/glo/protect` | 保留签名但不再生效（v2 不使用） |
| `inpaint(should_cancel=...)` | no-op（wasm 填充不可中断），仅为 API 兼容保留 |
| 无 wasmtime 依赖 | `wasmtime>=25` 硬依赖（随 pip 自动安装） |
| `scripts/` 88 个脚本 | 精简至 20 个核心验收/诊断工具（移除 68 个引用已删核心的一次性脚本） |
| `docs/PYTHON_PIPELINE_REFERENCE.md` | 删除（移植期对照基准，所述代码已不存在） |

Web 界面（demo）用户无感：0.2.x 界面本就只有 v2/off 两档去发光。

## [0.2.1] — 2026-08-31

- 修复与维护性发布（纯 Python 核心架构的最后一个版本）。

## [0.2.0] — 2026-08-30

- 更名 `textpatch` → `text_eraser`（PyPI 包名 `text-eraser`），模型缓存目录迁移
  `~/.textpatch/models/det` → `~/.text_eraser/models/det`（旧目录自动兼容）。

## [0.1.0] — 2026-08-30

- 初始版本：DBNet 文字检测 + 框内 Otsu 蒙版 + PatchMatch 填充 + 去发光 v2 + 本地 Web 界面。
