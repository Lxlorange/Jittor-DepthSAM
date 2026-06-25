# 复现日志

本文档记录 Jittor-DepthSAM 复现过程中的阶段性版本、关键修改、测试结论和环境问题处理。版本号按当前 git 提交历史顺序整理。

## v0.1 - 2026-06-21 - 初始化原始代码

对应提交：`eba3410 init origin code`

- 初始化项目仓库。
- 引入 DepthSAM 原始 PyTorch 代码作为后续 Jittor 迁移的结构参考。
- 保留论文、数据准备、训练、测试等后续复现所需目录。

## v0.2 - 2026-06-22 - 基础损失和公共模块迁移

对应提交：`ec02411 update losses and common`

- 开始迁移 Jittor 基础组件。
- 实现 `structure_loss`，对齐 PyTorch 中的加权 BCE + 加权 IoU 结构损失。
- 迁移 `LayerNorm2d`、`MLPBlock` 等公共模块。
- 初步验证基础张量形状和参数更新逻辑。

## v0.3 - 2026-06-22 - MyNet / GSFM 相关模块迁移

对应提交：`a6e65a1 MyNet`

- 开始迁移 `MyNet.py`。
- 对应论文中的 GSFM / SFRM 多尺度几何-语义融合模块。
- 迁移 `Attention_SD`、`FM`、`MEF`、`Decode`、`DWConv` 等结构。
- 其中 `Attention_SD` 涉及频域 FFT、空间注意力和跨域交互，是 Jittor 迁移中的主要难点之一。

## v0.4 - 2026-06-23 - PromptEncoder 迁移

对应提交：`197826b prompt_encoder`、`c706e8b prompt_encoder`

- 迁移 SAM 风格的 `PromptEncoder`。
- 保留 Meta SAM 原始版权说明。
- 完成位置编码 `PositionEmbeddingRandom`、mask prompt 下采样、无 prompt embedding 等逻辑迁移。
- 已通过基本 shape 测试：
  - `get_dense_pe()` 输出 `[1, 256, 256, 256]`
  - `sparse_embeddings` 输出 `[1, 0, 256]`
  - `dense_embeddings` 输出 `[1, 256, 256, 256]`

## v0.5 - 2026-06-23 - 同步 dev 分支

对应提交：`5e67896 Merge branch 'dev' ... into dev`

- 同步远端 dev 分支内容。
- 保持当前 Jittor 迁移分支与远端仓库一致。

## v0.6 - 2026-06-24 - DepthSAM Decoder 迁移

对应提交：`3a08c7d DepthSAM_decoder`

- 迁移 `DepthSAM_decoder.py`。
- 实现 SAM mask decoder 风格的 token 拼接、transformer 调用、mask token hypernetwork 和上采样预测逻辑。
- 当前该模块依赖 `TwoWayTransformer`、`PromptEncoder` 和图像 embedding / dense prompt 的 shape 对齐。

## v0.7 - 2026-06-24 - 目录结构重构

对应提交：`852be43 refactor dir`

- 将迁移代码向原始 DepthSAM 仓库结构靠拢。
- Jittor 版本逐步采用：
  - `segment_anything_training/modeling/`
  - `segment_anything_training/build_DepthSAM.py`
  - `tests/`
- 将早期平铺的 `modeling/` 试验结构迁移到更接近原仓库的目录下，便于和 PyTorch 实现逐文件对照。

## v0.8 - 2026-06-24 - Sam 模块串联

对应提交：`ccaf544 Sam moudel`

- 继续迁移 SAM / DepthSAM 串联模块。
- 补充 `build_DepthSAM.py`、`build_sam.py`、`DepthSAM_edge.py`、`transformer.py` 等文件。
- 增加 `tests/test_Sam.py`、`tests/test_common.py` 等测试入口。
- 当前重点是先用假输入和局部模块测试打通：
  - `MyNet.Decode`
  - `PromptEncoder`
  - `MaskDecoder`
  - SAM 风格 forward 链路

## v0.9 - 2026-06-24 - 修复并测试 FFT

对应提交：`49c9772 fix and test fft`

### 问题现象

执行 Jittor 版 MyNet/Sam 测试时，`Attention_SD` 的 FFT 路径触发 Jittor cuFFT 编译错误。最小复现脚本中，即使只执行：

```python
import jittor as jt
import jittor.nn as nn

jt.flags.use_cuda = 1
real = jt.randn((1, 8, 8))
z = nn.ComplexNumber(real)
y = z.fft2()
print(y.real.sync())
```

也会报错：

```text
std::array<int, 2> fft = {n1, n2};
error: incomplete type is not allowed
```

这说明问题不在 `MyNet.py` 的模型逻辑，也不在 `_fft2/_ifft2` 包装函数本身，而是 Jittor 当前环境中的 cuFFT JIT op 编译失败。

### 原因分析

Jittor 的复数 FFT 调用路径为：

```text
nn.ComplexNumber.fft2()
  -> nn._fft2(self.value, inverse=False)
  -> Jittor 生成 cufft_fft CUDA/C++ op
```

其中 `ComplexNumber` 将复数保存为：

```text
[B, H, W, 2]
```

最后一维 `2` 分别表示实部和虚部。

Jittor 生成的 cuFFT op 源码内部使用了：

```cpp
std::array<int, 2> fft = {n1, n2};
```

但生成文件或其模板没有正确包含：

```cpp
#include <array>
```

在当前环境组合下：

```text
Jittor 1.3.10.0
Python 3.9
g++ 12.4.0
Jittor CUDA 12.2
NVIDIA Driver CUDA 12.9
RTX 3050 Ti
```

`std::array` 没有被其他头文件间接引入，因此编译器认为它是不完整类型，导致 JIT 编译失败。

### 修改方案

采用本地环境补丁：在 Jittor 安装目录中对应的 cuFFT op `.cc` 模板文件(/home/orangelxl/miniconda3/envs/jittor_env/lib/python3.9/site-packages/jittor/extern/cuda/cufft/ops/cufft_fft_op.cc)中加入：

```cpp
#include <array>
```

修改后重新触发 Jittor 编译，FFT 相关测试通过。

该修改属于本地 Jittor 环境修复，不属于 DepthSAM 模型代码修改。后续在 README 中需要明确记录，保证复现实验可解释。

### 修复后测试结果

执行：

```bash
python -u tests/test_Sam.py
```

输出：

```text
Testing MyNet components...
test BasicConv2d got (2, 16, 16, 16), expect (2, 16, 16, 16)
test DWConv got (2, 32, 16, 16), expect (2, 32, 16, 16)
test Attention_SD got (2, 32, 16, 16), expect (2, 32, 16, 16)
test FM got (2, 32, 16, 16), expect (2, 32, 16, 16)
test MEF got (2, 32, 16, 16), expect (2, 32, 16, 16)
test Decode out got (2, 1, 32, 32), expect (2, 1, 32, 32)
test Decode br1 got (2, 32, 16, 16), expect (2, 32, 16, 16)
test conv1x1 got (2, 16, 16, 16), expect (2, 16, 16, 16)
test conv3x3 got (2, 16, 16, 16), expect (2, 16, 16, 16)
test conv1x1_bn_relu got (2, 16, 16, 16), expect (2, 16, 16, 16)
test conv3x3_bn_relu got (2, 16, 16, 16), expect (2, 16, 16, 16)
test_mynet passed
All tests passed!
```

该结果说明 Jittor 版 `MyNet.py` 中的 GSFM / SFRM 相关基础模块已经完成最小 shape 测试。


### 社区反馈

我认为这是一个特定版本下触发的bug，但又很容易修复，且无破坏性更新，因此我已经向Jittor仓库提交了[PR](https://github.com/Jittor/jittor/pull/18024)。
