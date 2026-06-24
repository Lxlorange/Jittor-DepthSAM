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

