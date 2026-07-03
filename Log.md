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

## v0.10 - 2026-07-03 - Jittor 显存瓶颈定位与方案 A

### 背景

Jittor 版本在完整训练链路中已经可以稳定运行，但在 `batchsize=2`、`trainsize=384` 下曾出现显存溢出。为了避免只凭经验猜测，使用 Jittor memory profiler 对 full / encoder / decoder 三条路径进行定位。

### Profiler 结论

- `encoder` profile 在 `batchsize=2` 下峰值约为 3.73GB，主要来自 `depth_anything_v2/dpt.py -> dinov2.py`，不是主要瓶颈。
- `decoder` profile 在 `batchsize=2` 下峰值约为 12.34GB，最大路径集中在 `segment_anything_training/modeling/MyNet.py`：
  - `Decode.execute -> self.fm1(...)`
  - `MEF.execute -> self.fm(...)`
  - `FM.execute -> self.asd(x, x)`
- 因此瓶颈不是 Depth Anything encoder，也不是 MOEAdapter，而是最高分辨率分支 `fm1` 内的 `Attention_SD`。该模块包含 FFT / iFFT / ComplexNumber real/imag/abs 等操作，Jittor 下反向图会保留较多中间变量，导致 batch 维度放大后显存峰值过高。

### 方案 A

方案 A 只对最高分辨率融合分支 `fm1` 使用轻量路径：

```python
self.fm1 = MEF(in1, in2, use_attention=False)
```

即 `fm1` 保留：

- `conv_n`
- `DWConv`
- 后续 `premp` mask 输出路径

但跳过该分支中的 `Attention_SD`。`fm2` 和 `fm3` 仍保留原始 `Attention_SD`，因此低分辨率和中分辨率的频域/空间交互仍然存在。

### 合理性

论文 Beyond Appearance: Camouflaged Object Detection via Geometric Structure 的核心动机是利用几何结构信息补充外观特征，DepthSAM 中的几何结构融合模块通过多尺度特征进行语义融合。当前修改没有移除 Depth Anything 几何先验、SAM mask decoder，也没有移除全部 GSFM/SFRM 风格融合，只是在 Jittor 显存峰值最高的最高分辨率分支上使用轻量近似。

这一改动的工程理由是：

- profiler 明确显示最高分辨率 `fm1 -> Attention_SD` 是主要显存峰值来源；
- `fm1` 处理的空间分辨率最高，FFT/complex 中间变量最容易放大；
- `fm2/fm3` 保留 attention 后，模型仍保留较低分辨率上的频域和跨域交互；
- 方案 A 使 batchsize 从 1/2 的不稳定边界提升到可用的 batchsize=2，并具备继续尝试 batchsize=3 的空间。

该方案不是严格等价实现，而是 Jittor 复现中的显存友好近似。因此后续必须用验证集指标证明其影响可接受。

### 当前性能观察

已有 benchmark 结果：

- 原 Jittor full train，`batchsize=1`：约 648ms/iter。
- 方案 A Jittor full train，`batchsize=2`：约 526ms/iter，约 3.80 FPS。
- 实际训练中，方案 A 后单卡一个 epoch 约 30 分钟，显著优于原始 Jittor 路径。

训练 loss 观察：

- Jittor 方案 A 前 6 个 epoch 的 mean loss 从 1.1107 降至 0.3820。
- Torch baseline 前 5 个 epoch 的 mean loss 从 1.1206 降至 0.4112。
- 两者 batchsize 不同，Jittor 为 2，Torch 为 4；日志采样频率也不同，因此不能只凭 train loss 判定精度等价。
- 但从趋势看，方案 A 没有造成训练 loss 崩溃，收敛方向与 Torch baseline 基本一致。

### 后续验证计划

为了证明方案 A 对性能无明显负面影响，需要至少比较以下内容：

1. 固定训练数据、输入尺寸、学习率和 epoch 数，分别记录 Torch baseline 与 Jittor 方案 A 的验证集 MAE。
2. 优先比较 CAMO / COD10K / NC4K / CHAMELEON 的 test MAE，而不是只比较 train loss。
3. 每 10 个 epoch 保存一次可视化结果，检查边界、细小伪装目标和低对比目标是否退化。
4. 如果方案 A 在最终 MAE 上明显落后，再尝试折中方案：只在 `fm1` 内先下采样后执行 `Attention_SD`，再上采样回原分辨率。
5. 如果方案 A 与 Torch baseline 的验证 MAE 接近，优先保留方案 A，以换取可完成 120 epoch 训练的吞吐和显存稳定性。

### total_step 修正

Jittor `Dataset` 的 `len(train_loader)` 返回的是样本数而不是 batch 数。此前 `total_step = len(train_loader)` 会导致：

- `batchsize=2` 时显示 total_step=4040，但实际每 epoch 约 2020 个 batch；
- tqdm 进度条和 CSV 中的 `total_step` 与 Torch baseline 不可直接对齐；
- epoch 最后一步不一定触发 `i == total_step` 的日志记录。

因此训练脚本改为：

```python
sample_count = getattr(train_loader, "total_len", len(train_loader))
total_step = math.ceil(sample_count / opt.batchsize)
```

修正后 `batchsize=2` 对应 `total_step=2020`，`batchsize=3` 对应 `total_step=1347`，`batchsize=4` 对应 `total_step=1010`，与 Torch DataLoader 的 step 语义一致。

### MOEAdapter experts 设置影响

当前 Jittor 代码中的 `MOEAdapter` 位于 `segment_anything_training/modeling/DepthSAM_edge.py`，用于包裹冻结后的 Depth Anything ViT block：

```python
prompted = x + output
return self.block(prompted)
```

其中 `gate` 根据 token 特征生成专家权重，`experts` 是多个小型 MLP adapter。它的作用不是替代 Depth Anything 主干，而是在冻结主干参数的前提下提供少量可训练的特征调整能力。换句话说，它更接近参数高效微调 adapter，而不是论文中 DepthSAM 几何结构建模的核心组件。

从当前 README 和代码结构看，论文核心是利用 Depth Anything 提供几何结构先验，并通过多尺度几何结构融合模块与 SAM 风格 mask decoder 结合；`MOEAdapter` 属于复现/训练策略中的工程扩展。原始 torch baseline 如果没有同样的 `DEPTHSAM_MOE_EXPERTS=1` 配置，那么 Jittor 与 torch 在 adapter 容量上并不完全对齐。

`export DEPTHSAM_MOE_EXPERTS=1` 的影响：

- 将多个 expert 退化为单 expert adapter，`gate` 此时基本不再承担“选择专家”的作用；
- 减少 adapter 参数量、优化器状态和反向图中 expert 分支的中间变量；
- 能降低 Jittor 显存压力和无效 expert 计算，适合单卡复现实验；
- 代价是 adapter 表达能力弱于默认多 expert 设置，严格来说不再与默认 torch 配置完全等价。

因此该设置应当在实验记录中明确标注为 Jittor practical run 配置，而不是论文原始结构。后续证明其影响可接受的方法是：

1. 至少保留一组短训对照：`DEPTHSAM_MOE_EXPERTS=1` 与默认 expert 数，在相同 batchsize 可行范围内比较前 5-10 epoch 的验证 MAE。
2. 如果默认 expert 数无法 batch2 运行，可用 batch1 默认 expert 作为结构对照，但只比较验证趋势，不直接比较吞吐。
3. 最终报告中同时记录：Jittor 为 `fm1 lite + MOE experts=1`，torch baseline 为原始设置或其实际运行设置。
4. 只要最终 CAMO / COD10K / NC4K / CHAMELEON 的 MAE 与 torch baseline 接近，且可视化结果没有明显边界和小目标退化，就可以认为该工程配置对复现结论无明显负面影响。
