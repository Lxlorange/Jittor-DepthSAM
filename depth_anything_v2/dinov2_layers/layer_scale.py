# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L103-L110

from typing import Union

import jittor as jt
from jittor import Var
from jittor import nn


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Var] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * jt.ones(dim))

    def execute(self, x: Var) -> Var:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
