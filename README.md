# Jittor-DepthSAM

> 本项目旨在通过新兴深度学习框架Jittor复现DepthSAM这一COD模型。

## 论文信息

- **Title:** Beyond Appearance: Camouflaged Object Detection via Geometric Structure
- **Conference:** CVPR 2026
- **Authors:** Han et al.
- **Link:** [PDF/CVF](https://openaccess.thecvf.com/content/CVPR2026/papers/Han_Beyond_Appearance_Camouflaged_Object_Detection_via_Geometric_Structure_CVPR_2026_paper.pdf)

## 环境配置



## 数据准备脚本

## 训练脚本

## 测试脚本

## 对齐实验与性能实验


---

# DepthSAM Jittor Reproduction Checklist

## Day 1
- [ ] GitHub repository initialized
- [ ] Official PyTorch DepthSAM code downloaded
- [ ] Pre-trained weights downloaded
- [ ] Dataset structure confirmed (COD10K/CAMO/NC4K)
- [ ] Run PyTorch test on one image, save prediction
- [ ] Log environment information (Python, PyTorch, CUDA, GPU)

## Day 2
- [ ] PyTorch small subset training run, save log, loss curve, predictions

## Day 3
- [ ] Jittor dataset loader implemented
- [ ] Loss functions implemented
- [ ] Metrics implemented
- [ ] Minimal training loop tested

## Day 4-6
- [ ] Minimal DepthSAM forward pass implemented
- [ ] GSFM + SFRM implemented
- [ ] SMEA adapter implemented

## Day 7
- [ ] DAv2/DINOv2 backbone integration attempted
- [ ] Torch->Jittor weight conversion verified

## Day 8-9
- [ ] Complete forward + training pipeline
- [ ] Small-sample Jittor training done
- [ ] Predictions saved

## Day 10-11
- [ ] Evaluation metrics computed
- [ ] Loss curves plotted
- [ ] Qualitative comparison with PyTorch predictions
- [ ] README updated with results

## Day 12
- [ ] PPT draft created
- [ ] Figures, logs, and architecture diagrams included

## Day 13
- [ ] Video recorded (algorithm + training + testing + implementation details)

## Day 14
- [ ] Final repository check
- [ ] README, PPT, video verified
- [ ] GitHub push and submission