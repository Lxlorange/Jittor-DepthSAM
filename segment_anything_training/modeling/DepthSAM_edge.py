# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# import cv2
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as ops
import numpy as np
import jittor as jt
from jittor import nn, Var
from typing import Any, Dict, List, Tuple, Union
from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .MyNet import Decode


def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)


def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv1x1(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

class MOEAdapter(nn.Module):
    def __init__(self, blk, num_experts=8, top_k=2) -> None:
        super(MOEAdapter, self).__init__()
        self.block = blk
        self.num_experts = num_experts
        self.top_k = top_k

        dim = blk.attn.qkv.in_features

        # 门控网络
        self.gate = nn.Linear(dim, num_experts)

        # 多个专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 32),
                nn.GELU(),
                nn.Linear(32, dim),
                nn.GELU(),
            ) for _ in range(num_experts)
        ])

    def execute(self, x):
        # x shape: (B, H, W, C)
        B, N, C = x.shape
        x_flat = x.reshape(B * N, C)  # 展平空间维度

        # 计算门控权重
        gate_logits = self.gate(x_flat)  # (B*H*W, num_experts)
        gate_weights = nn.softmax(gate_logits, dim=-1)

        # Dense routing avoids Jittor top-k/where crashes and keeps every expert trainable.
        expert_outputs = []
        for e_idx in range(self.num_experts):
            expert_outputs.append(self.experts[e_idx](x_flat))
        expert_outputs = jt.stack(expert_outputs, dim=1)

        output = jt.sum(expert_outputs * gate_weights.unsqueeze(-1), dims=[1])
        output = output.reshape(B, N, C)
        prompted = x + output
        return self.block(prompted)

class EdgeDepthSAM(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        # 改成npz了，再转成Var
        model_path = f'checkpoints/depth_anything_v2_vitl.npz'

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        print(f"Loading checkpoint from {model_path}")

        npz = np.load(model_path)
        state_dict = {k: jt.array(npz[k]) for k in npz.files}
        self.image_encoder.load_state_dict(state_dict)

        for param in self.image_encoder.parameters():
            param.requires_grad = False


        blocks = []
        for block in self.image_encoder.pretrained.blocks:
            blocks.append(
                MOEAdapter(block)
            )
        self.image_encoder.pretrained.blocks = nn.Sequential(
            *blocks
        )
        for block in self.image_encoder.pretrained.blocks:
            for param in block.gate.parameters():
                param.requires_grad = True
            for param in block.experts.parameters():
                param.requires_grad = True

        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        self.decoder = Decode(256,256,256,256)

        self.register_buffer("pixel_mean", jt.Var(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", jt.Var(pixel_std).view(-1, 1, 1), False)
        self.pixel_mean.requires_grad = False
        self.pixel_std.requires_grad = False
        for param in self.prompt_encoder.point_embeddings.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.not_a_point_embed.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.no_mask_embed.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.pe_layer_2.parameters():
            param.requires_grad = False

    def _set_prompt_encoder_size(self, embedding):
        image_embedding_size = tuple(embedding.shape[-2:])
        self.prompt_encoder.image_embedding_size = image_embedding_size
        self.prompt_encoder.mask_input_size = (
            image_embedding_size[0] * 4,
            image_embedding_size[1] * 4,
        )

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def execute(
        self,
        batched_input: List[Dict[str, Any]],x
    ):
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a jt Var in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (jt.Var) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (jt.Var) Batched labels for point prompts,
                with shape BxN.
              'boxes': (jt.Var) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (jt.Var) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (jt.Var) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (jt.Var) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (jt.Var) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """

        x = nn.interpolate(x, scale_factor=14 / 16, mode='bilinear', align_corners=True)

        depth,features = self.image_encoder(x)

        out1,out_1 = self.decoder(features[3], features[2], features[1], features[0])
        outputs = []
        for image_record, curr_embedding,out11 in zip(batched_input, out_1,out1):
            self._set_prompt_encoder_size(curr_embedding)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=image_record.get("boxes", None),
                masks=out11.unsqueeze(0),
            )

            low_res_mask, iou = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
            )
            outputs.append(
                {
                    "mask": low_res_mask,
                    "low_res_logits": low_res_mask,
                }
            )
        masks = jt.concat([x["mask"] for x in outputs], dim=0)
        return masks


    def postprocess_masks(
        self,
        masks: jt.Var,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> jt.Var:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (jt.Var): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (jt.Var): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = nn.interpolate(masks, original_size, mode="bilinear")

        return masks

    def preprocess(self, x: jt.Var) -> jt.Var:
        """Normalize pixel values and pad to a square input."""
        x = (x - self.pixel_mean) / self.pixel_std
        x = x.unsqueeze(0)
        return x
