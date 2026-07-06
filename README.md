# Jittor-DepthSAM

本项目使用 [Jittor](https://github.com/Jittor/jittor) 复现论文 **Beyond Appearance: Camouflaged Object Detection via Geometric Structure** 中提出的 DepthSAM 方法。仓库目标是尽量保持与原 PyTorch 实现的模块命名、目录结构、训练流程和数据组织方式一致，并记录 Jittor 迁移过程中的环境问题、训练日志、性能日志和可视化结果。

## 论文信息

- **Title:** Beyond Appearance: Camouflaged Object Detection via Geometric Structure
- **Conference:** CVPR 2026
- **Authors:** Han et al.
- **Task:** Camouflaged Object Detection (COD)
- **Link:** [PDF / CVF OpenAccess](https://openaccess.thecvf.com/content/CVPR2026/papers/Han_Beyond_Appearance_Camouflaged_Object_Detection_via_Geometric_Structure_CVPR_2026_paper.pdf)

## 环境配置

基础环境：

```text
Python: 3.9
Jittor: >= 1.3.7.5
CUDA: 需要可用 CUDA 环境
```

最终 120 epoch 训练和测试使用的 AutoDL 单卡环境记录如下：

```text
Python: 3.9.25
Jittor: 1.3.11.0
Compiler: g++ 11.4.0
Jittor CUDA Toolkit: 12.2.140
CUDA arch: sm_89
GPU memory: 24GB 级别单卡
```

创建环境：

```bash
conda create -n jittor_env python=3.9 -y
conda activate jittor_env
pip install -U pip
pip install jittor numpy pillow opencv-python tqdm matplotlib
```

如果运行 `Attention_SD` 中的 FFT 路径，必须启用 CUDA。早期本地环境曾遇到 Jittor cuFFT JIT 编译错误，错误原因是 Jittor 生成的 cuFFT op 源码缺少 `#include <array>`。本地修复方式是在 Jittor 安装目录的如下文件中补充头文件：

```text
.../site-packages/jittor/extern/cuda/cufft/ops/cufft_fft_op.cc
```

该问题和模型逻辑无关，详细记录见 `Log.md` 的 `v0.9 - 修复并测试 FFT`。

## 代码结构

```text
segment_anything_training/
  build_DepthSAM.py                  # 构建 Jittor 版 DepthSAM
  modeling/
    DepthSAM_edge.py                 # SAM 主体、Depth Anything 主干加载、MOEAdapter
    MyNet.py                         # GSFM/SFRM 相关融合与解码模块
    prompt_encoder.py                # SAM PromptEncoder 迁移
    mask_decoder.py                  # SAM MaskDecoder 迁移
    transformer.py                   # TwoWayTransformer 迁移
depth_anything_v2/                   # Depth Anything V2 的 Jittor 迁移
data_cod.py                          # 测试集读取器
utils/dataset_rgb_strategy2.py        # 训练集读取器，读取 RGB / GT / depth
utils/experiment_monitor.py           # 训练、测试、loss 曲线、可视化日志
train.py                             # 训练入口
test.py                              # 测试入口
tests/                               # 模块级 shape / pipeline / benchmark 测试
scripts/prepare_torch_report_assets.py   # PyTorch 对齐结果整理
scripts/prepare_jittor_report_assets.py  # Jittor 对齐结果整理
report/image/                        # 报告用 loss、MAE、FPS、可视化图表
Log.md                               # 迁移过程、环境问题、性能观察日志
```

## 数据准备

本项目使用 COD-D 目录结构，保持和原 PyTorch 仓库一致。数据集先下载到 `Data_all/COD-D/`，再使用原始仓库的数据准备工具和 Depth Anything 的 `depth_anything_v2_vitl.pth` 为所有图像生成深度图。生成结果放在每个 split 对应的 `depth/` 文件夹下。

本仓库当前不内置单独的深度图生成脚本；训练和测试脚本默认消费已经生成好的 `depth/` 目录。若从零复现，需要先按原 Depth Anything / 原 DepthSAM 仓库的工具为 `Train_depth` 和 `Test_depth` 下各数据集生成深度图，再运行本仓库训练或测试。

当前仓库中的数据目录如下：

```text
Data_all/COD-D/
  Train_depth/
    Imgs/      # 4040 张训练图像
    GT/        # 4040 张训练标注
    depth/     # 4040 张 Depth Anything 生成深度图
    Edge/
  Test_depth/
    CAMO/
      Imgs/    # 250
      GT/      # 250
      depth/   # 250
    CHAMELEON/
      Imgs/    # 76
      GT/      # 76
      depth/   # 76
    COD10K/
      Imgs/    # 2026
      GT/      # 2026
      depth/   # 2026
    NC4K/
      Imgs/    # 4121
      GT/      # 4121
      depth/   # 4121
```

数据读取逻辑：

- 训练：`train.py -> get_loader(...) -> utils/dataset_rgb_strategy2.py::SalObjDataset`，每个样本返回 `image, gt, depth`。
- 测试：`test.py -> data_cod.py::test_dataset`，每个样本返回 `image, gt, depth, name, image_for_post`。
- 图像输入按 ImageNet 均值方差归一化，GT 和 depth 都以单通道 `[0, 1]` 形式读取。

预训练权重说明：

```text
checkpoints/depth_anything_v2_vitl.pth    # 从 Depth Anything 项目下载的 PyTorch 权重
checkpoints/Model_COD_gen.pth             # 原 PyTorch DepthSAM / COD 权重
```

当前 Jittor 代码在 `segment_anything_training/modeling/DepthSAM_edge.py` 中加载的是：

```text
checkpoints/depth_anything_v2_vitl.npz
```

因此如果只下载了 `.pth`，需要先把 Depth Anything 的 PyTorch `state_dict` 转成 Jittor 可直接读取的 `.npz`，或补充对应转换脚本后再运行训练 / 测试。`train.py` 训练得到的 Jittor 模型权重会保存为 `checkpoints/Model_<epoch>_gen.npz`。

## 训练脚本

默认训练入口：

```bash
python -u train.py \
  --epoch 200 \
  --batchsize 2 \
  --trainsize 512 \
  --lr_gen 5e-5 \
  --log_interval 50 \
  --sync_interval 200
```

单卡显存有限时可以使用更小输入尺寸或更少 epoch 做对齐实验：

```bash
python -u train.py \
  --epoch 10 \
  --batchsize 2 \
  --trainsize 384 \
  --lr_gen 5e-5 \
  --log_interval 20
```

如果使用单 expert adapter 降低显存压力：

```bash
export DEPTHSAM_MOE_EXPERTS=1
python -u train.py --epoch 10 --batchsize 2 --trainsize 384
```

本次最终 Jittor 训练实际使用：

```bash
export DEPTHSAM_MOE_EXPERTS=1
export DEPTHSAM_AUX_ALPHA=0.01
python -u train.py \
  --epoch 120 \
  --batchsize 2 \
  --trainsize 384 \
  --lr_gen 5e-5 \
  --log_interval 50 \
  --sync_interval 100 \
  --aux_weight 0.01
```

对应训练日志：

```text
runs/jittor/runs/20260704-131004_jittor_train/
  config.json
  train_log.csv
  eval_log.csv
  loss_curve.png
  summary.json
```

训练脚本行为：

- 损失函数为 `structure_loss`，对齐 PyTorch 版本中的加权 BCE + 加权 IoU。
- 训练数据固定读取 `Data_all/COD-D/Train_depth/Imgs/`、`GT/`、`depth/`。
- `total_step` 使用 `ceil(sample_count / batchsize)`，避免 Jittor `Dataset.__len__` 返回样本数导致 step 统计和 PyTorch DataLoader 不一致。
- 每个 epoch 结束后保存 `checkpoints/Model_<epoch>_gen.npz`；当前代码中 `epoch >= 10` 后会触发保存和 CAMO 测试。
- `utils/experiment_monitor.py` 自动记录训练日志、测试日志、loss 曲线和预测可视化。

训练日志产物：

```text
runs/<时间戳>_jittor_train/
  config.json
  train_log.csv       # time, epoch, step, total_step, loss, lr, gpu_mem_mb, gpu_util
  eval_log.csv        # dataset, name, mae, skipped, gpu_mem_mb, gpu_util
  loss_curve.png      # loss 曲线
  summary.json        # 运行耗时、峰值显存等摘要
  visuals/<dataset>/  # 原图 / 预测 / GT 横向拼接可视化
```

## 测试脚本

默认测试入口：

```bash
python -u test.py --model ./checkpoints --trainsize 512
```

也可以指定某个 Jittor checkpoint：

```bash
python -u test.py --model ./checkpoints/Model_10_gen.npz --trainsize 512
```

当前 `test.py` 默认测试 `CAMO`，预测图保存到：

```text
test_maps/CAMO/
runs/<时间戳>_jittor_test/
  config.json
  eval_log.csv
  summary.json
  visuals/CAMO/
```

如需测试 `CHAMELEON`、`COD10K`、`NC4K`，需要把 `test.py` 中的：

```python
test_datasets = ['CAMO']
```

扩展为：

```python
test_datasets = ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']
```

本次最终实验采用 `checkpoints/Model_120_gen.npz`，分两次完成四个测试集：

```text
runs/jittor/runs/20260706-065723_jittor_test/  # CAMO
runs/jittor/runs/20260706-070117_jittor_test/  # CHAMELEON / COD10K / NC4K
```

对应命令形式为：

```bash
python -u test.py --model checkpoints/Model_120_gen.npz --trainsize 384 --aux_weight 0.01
```

## 模块级测试

当前仓库包含若干 Jittor 模块级测试，用于和 PyTorch 实现做 shape / forward 链路对齐：

```bash
python -u tests/test_common.py
python -u tests/test_prompt_encoder.py
python -u tests/test_Sam.py
python -u tests/test_DepthSAM.py
python -u tests/test_depth.py
python -u tests/test_train_pipeline.py
```

已记录的关键 shape 对齐结果见 `Log.md`：

- `PromptEncoder.get_dense_pe()` 输出 `[1, 256, 256, 256]`。
- 无 prompt 输入下，`sparse_embeddings` 输出 `[1, 0, 256]`，`dense_embeddings` 输出 `[1, 256, 256, 256]`。
- `MyNet.Decode` 输出 `(2, 1, 32, 32)`，`br1` 输出 `(2, 32, 16, 16)`。
- `Attention_SD`、`FM`、`MEF`、`DWConv` 等基础模块已通过最小 shape 测试。

## 与 PyTorch 实现的对齐记录

迁移原则：

- 尽量保留原 PyTorch 仓库中的类名、函数名和文件职责，便于逐文件对照。
- `DepthAnythingV2 -> DINOv2 -> DPTHead -> EdgeDepthSAM -> Decode / MaskDecoder` 是当前 Jittor 版本的主链路。
- `Depth Anything` 主干默认冻结，`MOEAdapter` 作为少量可训练 adapter 包裹 ViT blocks。
- 因 Jittor 显存压力，当前工程版本对最高分辨率 `fm1` 分支采用轻量路径：`MEF(..., use_attention=False)`；`fm2` 和 `fm3` 仍保留 `Attention_SD`。

当前对齐日志摘要：

| 项目 | PyTorch / 原实现 | Jittor 当前实现 | 记录位置 |
| --- | --- | --- | --- |
| 损失函数 | weighted BCE + weighted IoU | `train.py::structure_loss` | `Log.md v0.2` |
| PromptEncoder | SAM 风格 prompt 编码 | `segment_anything_training/modeling/prompt_encoder.py` | `Log.md v0.4` |
| MaskDecoder / Transformer | SAM mask decoder 链路 | `mask_decoder.py`、`transformer.py` | `Log.md v0.6` |
| GSFM / SFRM | `MyNet.py` 多尺度融合 | Jittor 迁移，`fm1` 使用轻量近似 | `Log.md v0.3`、`v0.10` |
| Depth Anything | `depth_anything_v2_vitl.pth` | 需要 `.npz` 权重供 Jittor 加载 | `DepthSAM_edge.py` |
| 训练 step 语义 | DataLoader batch 数 | `ceil(total_len / batchsize)` | `Log.md v0.10` |

训练 loss 对齐观察：

| 实验 | 配置 | 观察 |
| --- | --- | --- |
| Torch baseline | batchsize=4，前 5 epoch | mean loss 从 `1.1206` 降至 `0.4112` |
| Jittor 方案 A | batchsize=2，前 6 epoch | mean loss 从 `1.1107` 降至 `0.3820` |

以上结果说明 Jittor 方案 A 的训练 loss 没有崩溃，收敛趋势和 PyTorch baseline 基本一致。但两者 batchsize 和日志采样频率不同，不能只凭训练 loss 证明最终精度完全等价，后续仍应以 CAMO / COD10K / NC4K / CHAMELEON 的 MAE 和可视化结果为准。

最终 120 epoch 权重的 MAE 对齐结果如下：

| Dataset | PyTorch MAE | Jittor MAE | Gap |
| --- | ---: | ---: | ---: |
| CAMO | 0.02999 | 0.03155 | +0.00156 |
| CHAMELEON | 0.01876 | 0.01932 | +0.00056 |
| COD10K | 0.01598 | 0.01687 | +0.00089 |
| NC4K | 0.02238 | 0.02297 | +0.00059 |

报告图表资产：

```text
report/image/torch/torch_mae_table.csv
report/image/jittor/jittor_mae_table.csv
report/image/jittor/loss_alignment.png
report/image/jittor/qualitative_framework_comparison.png
```

## 性能日志

可使用 benchmark 脚本记录 Jittor 前向 / 训练吞吐：

```bash
python -u tests/test_benchmark_jittor_fps.py \
  --mode train \
  --batchsize 2 \
  --trainsize 384 \
  --iters 30 \
  --profile full \
  --output runs/benchmark_jittor_full_train_b2_384.json
```

也可以只测 encoder 或 decoder：

```bash
python -u tests/test_benchmark_jittor_fps.py --profile encoder --mode train --batchsize 2 --trainsize 384
python -u tests/test_benchmark_jittor_fps.py --profile decoder --mode train --batchsize 2 --trainsize 384
```

已有性能观察记录：

| 实验 | 配置 | 结果 |
| --- | --- | --- |
| Jittor forward | batchsize=1, trainsize=384 | `10.03 FPS`, `99.70 ms/iter` |
| Jittor train | batchsize=1, trainsize=384 | `2.21 FPS`, `451.60 ms/iter` |
| Jittor train | batchsize=2, trainsize=384 | `3.54 FPS`, `565.68 ms/iter` |
| Jittor encoder profile | batchsize=2 | 峰值约 `3.73 GB` |
| Jittor decoder profile | batchsize=2 | 峰值约 `12.34 GB`，主要瓶颈在 `MyNet.py` 的高分辨率 `fm1 -> Attention_SD` |
| 实际训练 | batchsize=2 | 单卡一个 epoch 约 `30 min` |

PyTorch 对齐 benchmark 记录在：

```text
runs/torch/runs/torch_benchmark_120/
report/image/torch/torch_benchmark_table.csv
report/image/jittor/jittor_benchmark_table.csv
```

详细分析见 `Log.md v0.10 - Jittor 显存瓶颈定位与方案 A`。

## 结果与可视化记录

当前仓库保留了历史预测结果目录：

```text
runs/production/test_maps_rebuttal_512/
  CAMO/       # 250 张预测图
  CHAMELEON/  # 76 张预测图
  COD10K/     # 522 张预测图
```

训练和测试时，`ExperimentMonitor.save_prediction_panel(...)` 会额外保存三联图：

```text
runs/<时间戳>_jittor_train/visuals/<dataset>/<name>.png
runs/<时间戳>_jittor_test/visuals/<dataset>/<name>.png
```

三联图顺序为：

```text
原图 | 预测 mask | GT
```

loss 曲线保存为：

```text
runs/<时间戳>_jittor_train/loss_curve.png
```

报告中使用的统一坐标 loss 对齐图保存为：

```text
report/image/jittor/loss_alignment.png
```

评测指标按样本记录在：

```text
runs/<时间戳>_jittor_test/eval_log.csv
```

其中 `mae` 的计算方式为：

```python
sample_mae = np.sum(np.abs(pred - gt)) / (H * W)
```

最终报告用可视化对齐图由脚本重新整理，格式为：

```text
Origin | GT | PyTorch | Jittor
```

每行一个数据集，覆盖 CAMO / CHAMELEON / COD10K / NC4K：

```text
report/image/jittor/qualitative_framework_comparison.png
```

生成报告图表：

```bash
python scripts/prepare_torch_report_assets.py
python scripts/prepare_jittor_report_assets.py
```

## 当前限制和后续补充项

- 当前 `test.py` 默认只测试 CAMO，完整四数据集测试需要扩展 `test_datasets`。本次最终实验已完成四数据集测试，结果见上方 MAE 表。
- 当前 Jittor 测试链路推理输入为 `trainsize x trainsize`，若直接保存 `test_maps`，预测图会是方形；报告可视化已用同名样本重新整理为统一尺寸对比图。若要保存原始比例 prediction map，需要在测试后处理时把 `res` resize 回原始 GT 尺寸。
- 当前 Jittor 实现为显存友好的工程复现版本，`fm1` 分支和 `DEPTHSAM_MOE_EXPERTS=1` 配置需要在最终报告中明确标注。
- `.pth -> .npz` 权重转换脚本尚未纳入仓库；若从零复现，需要先补充该脚本或提交已转换的 `depth_anything_v2_vitl.npz`。
- 若计算资源有限，建议固定少量训练数据、相同输入尺寸和相同 epoch 数，分别运行 PyTorch baseline 与 Jittor 版本，再比较训练 loss、CAMO MAE 和可视化三联图。

## 推荐复现实验顺序

1. 准备 COD-D 数据集，并为 `Train_depth` 和各测试集生成 `depth/`。
2. 准备 `checkpoints/depth_anything_v2_vitl.npz`。
3. 运行模块级测试，确认 Jittor 环境、FFT 和 shape 链路正常。
4. 运行 `tests/test_benchmark_jittor_fps.py`，记录当前机器吞吐和显存。
5. 用 `--epoch 10 --trainsize 384 --batchsize 2` 做小规模训练对齐。
6. 查看 `runs/<时间戳>_jittor_train/train_log.csv`、`loss_curve.png` 和 `visuals/`。
7. 运行 `test.py` 生成 CAMO MAE 和预测图。
8. 资源允许时扩展到 CAMO / CHAMELEON / COD10K / NC4K 四个测试集。
9. 运行 `scripts/prepare_jittor_report_assets.py` 和 `scripts/prepare_torch_report_assets.py`，生成 README / PPT 中使用的 loss、MAE、FPS 和可视化对齐图。
