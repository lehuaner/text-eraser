# shared — 前后端共用算法核（WASM）

> 目标：**一套算法，浏览器与后端各跑一份同一 `.wasm`，从此"改一处、两端自动一致"**，
> 不再需要手工对齐（我们此前用逐像素 IoU 校验维持的"双份手写"时代结束）。

## 架构

```
              textcore.wasm   ←── 唯一算法真源（Rust 编译，C-ABI 无胶水）
             /              \
   browser (WebAssembly)   backend (wasmtime)
   browser/src/textcore.js  shared/bindings/textcore.py
            \              /
            统一 API: distanceTransformEdt(mask, h, w) -> Float32Array
```

- 算子用 **Rust** 写一份，编译到 `wasm32-unknown-unknown`（C-ABI，不用 wasm-bindgen），
  因此同一份 `.wasm` 既能浏览器 `WebAssembly.instantiate` 加载，也能 Python `wasmtime` 加载。
- 两端各有一个**薄 binding**（`bindings/textcore.js` / `bindings/textcore.py`），
  把"分配线性内存 → 写输入 → 调算子 → 读输出 → 释放"的样板封装成统一 API。
- 任何算子只需在 Rust 里加一次、重编一次，两端立即获得**逐位相同**的实现。

## 已验证（决定性证据）

首个迁移算子 `distance_transform_edt`（精确欧氏距离变换，Felzenszwalb & Huttenlocher）：

| 检查 | 结果 |
|---|---|
| 同一 wasm 被 Node 与 Python 加载、相同输入 | 输出 **maxabs = 0.000000（逐位一致）** |
| wasm 输出 vs `scipy.ndimage.distance_transform_edt`（精确真值） | maxerr = 0.0000（=f32 舍入） |
| 语义 | 量到**文字像素**的距离，对应后端 `cv2.distanceTransform((cur==0), DIST_L2, 3)` |

> 注：cv2 的 `distanceTransform(DIST_L2,3/5)` 本身是**近似**（大距离误差可达 ~18px）；
> 我们的精确 EDT 是对的，且与后端旧 cv 版基线表现一致。

## 构建

```bash
cd shared
# 首次：rustup target add wasm32-unknown-unknown   （已装则跳过）
cargo build --target wasm32-unknown-unknown --release --offline
cp target/wasm32-unknown-unknown/release/textcore.wasm build/textcore.wasm
```

> ⚠️ 在本机沙箱里 **后台 `cargo build` 会挂死**（疑似 `| tail` 管道 + 后台任务调度问题）；
> 请**前台**直接跑上面的命令（已验证 ~3s 完成，std 缓存后更快）。

## 测试（跨端一致性）

```bash
cd shared/test
PY=.../Python313/python.exe ; NODE=.../node22/node.exe
$PY gen_masks.py                 # 生成 mask_synth.bin / mask_real.bin
$PY py-edt.py mask_synth.bin     # wasmtime 消费，写 .out_py.bin
$NODE node-edt.cjs mask_synth.bin # WebAssembly 消费，写 .out_node.bin
$PY compare.py mask_synth.bin     # 断言 node vs py 逐位一致

# 或用 binding 集成层（更贴近真实调用）
$PY  binding_test.py  mask_synth.bin
$NODE binding_test.cjs mask_synth.bin
```

## 待迁移算子（来自 browser/src/cv-bridge.js 的纯 JS 实现，已与 cv2 对齐）

按优先级，把下列算子逐个搬进 `src/lib.rs`，每加一个就补一个 binding + 跨端一致性测试：

1. `morphology`：dilate/erode + open/close，RECT/ELLIPSE 结构元，**anchor = floor(ksize/2)**
   （注意偶 ksize=2 的闭操作，a=(ksize-1)/2 会出非整数偏移 → no-op，必须用 floor）
2. `connectedComponents`：8 连通 flood-fill，保留 cv 约定（`stats[0]` 为背景占位）
3. `thresholdOtsu`：直方图 + 类间方差 argmax
4. `rgbToGray`：定点 `(r*4899+g*9617+b*1868+8192)>>14`
5. `resizeGrayU8`：INTER_CUBIC 可分离三次插值（a=-0.75）
6. `resizeFloat`：INTER_LINEAR 双线性

搬完后，让 `browser/src/cv-bridge.js` 与 `text_eraser/text_select.py` 都改调 `bindings`，
即可删除两端口径各自的实现，彻底消除漂移。

## 接入点（下一步）

- 浏览器：worker 通过 `fetch('/browser/.../textcore.wasm')` 加载，或用 `bindings/textcore.js` 的 API 替换 `cv-bridge.js` 里的纯 JS 算子。
- 后端：在 `text_select.py` / `patch_fill.py` 里 `from shared.bindings.textcore import get_core` 调用，`wasmtime` 单例复用。
