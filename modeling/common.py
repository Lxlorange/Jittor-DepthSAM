# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jittor as jt
import jittor.nn as nn
from typing import Type

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)

    def execute(self, x: jt.Var) -> jt.Var:
        return self.lin2(nn.gelu(self.lin1(x)))

# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        # Jittor中不需要Parameter接口，参见https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/jittor.nn.html#jittor.nn.Parameter
        self.weight = jt.ones(num_channels)
        self.bias = jt.zeros(num_channels)
        self.eps = eps

    def execute(self, x: jt.Var) -> jt.Var:
        u = x.mean(1, keepdims=True)
        s = (x - u).pow(2).mean(1, keepdims=True)
        x = (x - u) / (s + self.eps).sqrt()
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


    

if __name__ == '__main__':
    # x = jt.randn((1,1,256,768))
    # dim = 768
    # mlp_ratio = 4.0

    # mlp_block = MLPBlock(embedding_dim=dim, mlp_dim=int(dim*mlp_ratio))
    # output = mlp_block(x)
    # print(output.shape)
    
    ln = LayerNorm2d(num_channels=256)
    optimizer = nn.SGD(ln.parameters(), lr=0.01)
    print([p.name() for p in ln.parameters()])
    # 预期输出包含 'weight' 和 'bias'
    x = jt.randn((2, 256, 14, 14))
    y = ln(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)   # 应为 (2, 256, 14, 14)
    
    # 每个 (B,H,W) 位置上的 C 个通道应均值为 0，方差为 1
    # 注意：因为 weight=1, bias=0，所以输出就是标准归一化结果
    mean_per_pos = y.mean(1, keepdims=True)
    var_per_pos  = y.pow(2).mean(1, keepdims=True)  # 均值已为0，直接平方平均
    
    print("Mean (should ~0):", mean_per_pos.mean().item())
    print("Var  (should ~1):", var_per_pos.mean().item())
    
    # 验证梯度是否正常回传
    old_weight = ln.weight.copy()
    old_bias = ln.bias.copy()

    loss = y.sum()
    # 一步更新（内部包含 backward + step）
    optimizer.step(loss)

    # 检查参数是否被更新（如果梯度正常回传，参数会变化）
    print("weight updated:", not jt.all_equal(ln.weight, old_weight))
    print("bias   updated:", not jt.all_equal(ln.bias, old_bias))