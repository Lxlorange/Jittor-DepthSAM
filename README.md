# Jittor-DepthSAM

本项目使用 [Jittor](https://github.com/Jittor/jittor) 复现论文 **Beyond Appearance: Camouflaged Object Detection via Geometric Structure** 中提出的 DepthSAM 方法。

## 论文信息

- **Title:** Beyond Appearance: Camouflaged Object Detection via Geometric Structure
- **Conference:** CVPR 2026
- **Authors:** Han et al.
- **Task:** Camouflaged Object Detection (COD)
- **Link:** [PDF / CVF OpenAccess](https://openaccess.thecvf.com/content/CVPR2026/papers/Han_Beyond_Appearance_Camouflaged_Object_Detection_via_Geometric_Structure_CVPR_2026_paper.pdf)

## 环境配置

当前开发和测试环境：

```text
OS: WSL2 / Linux
Python: 3.9
Jittor: 1.3.10.0
CUDA Driver: 12.9
Jittor CUDA Toolkit: 12.2
Compiler: g++ 12.4.0
GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
```

创建环境：

```bash
conda create -n jittor_env python=3.9 -y
conda activate jittor_env
pip install -U pip
pip install jittor numpy pillow opencv-python tqdm
```


## 数据准备脚本

待补充。当前数据目录保持与 PyTorch 原仓库一致：

```text
Data_all/COD-D/
  Train_depth/
    Imgs/
    GT/
    depth/
  Test_depth/
    CAMO/
      Imgs/
      GT/
      depth/
    COD10K/
    NC4K/
    CHAMELEON/
```

## 训练脚本

待补充。

## 测试脚本

当前可运行的局部测试：

```bash
python -u tests/test_common.py
python -u tests/test_prompt_encoder.py
python -u tests/test_Sam.py
```

## 对齐实验与性能实验

待补充。后续需要记录：

- PyTorch baseline 环境与日志
- Jittor 小样本训练日志
- loss 曲线
- 预测图可视化
- PyTorch / Jittor 模块级 shape 对齐
- 小数据集性能对齐结果
