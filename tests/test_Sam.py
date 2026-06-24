import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import jittor as jt
import jittor.nn as nn
from functools import partial
from segment_anything_training.build_sam import sam_model_registry
from segment_anything_training.modeling import ImageEncoderViT,PatchEmbed,Block,Attention,window_partition,window_unpartition,get_rel_pos,PromptEncoder,PositionEmbeddingRandom,MaskDecoder,MLP,TwoWayTransformer

from segment_anything_training.modeling.MyNet import (
    Attention_SD, FM, MEF, Decode, DWConv, BasicConv2d,
    conv1x1, conv3x3, conv1x1_bn_relu, conv3x3_bn_relu
)


def check(name, x, shape):
    if isinstance(x, (tuple, list)):
        shapes = tuple(tuple(v.shape) if hasattr(v, 'shape') else str(type(v)) for v in x)
        print(f"test {name} got {shapes}, expect {shape}")
        assert shapes == shape, f"test {name} got {shapes}, expect {shape}"
    else:
        actual = tuple(x.shape)
        print(f"test {name} got {actual}, expect {shape}")
        assert actual == shape, f"test {name} got {actual}, expect {shape}"


def check_len(name, x, length):
    print(f"test {name} got len {len(x)}, expect {length}")
    assert len(x) == length, f"test {name} got len {len(x)}, expect {length}"

# 分别测试
def test_registry():
    assert isinstance(sam_model_registry, dict)
    assert "default" in sam_model_registry
    assert "vit_h" in sam_model_registry
    assert "vit_l" in sam_model_registry
    assert "vit_b" in sam_model_registry
    assert callable(sam_model_registry["vit_b"])
    print("test_registry passed\n")


def test_image_encoder():
    print("Testing ImageEncoderViT & submodules...")

    # 按vit_b参数构造
    encoder = ImageEncoderViT(
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(jt.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        out_chans=256,
    )

    x = jt.randn((2, 3, 1024, 1024))
    out, interm = encoder(x)

    check("image_encoder output", out, (2, 256, 64, 64))
    check_len("image_encoder interm", interm, 12)
    for i, emb in enumerate(interm):
        check(f"image_encoder interm[{i}]", emb, (2, 64, 64, 768))

    pe = PatchEmbed(kernel_size=(16, 16), stride=(16, 16), in_chans=3, embed_dim=768)
    pe_out = pe(x)
    check("patch_embed", pe_out, (2, 64, 64, 768))

    block_global = Block(
        dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(jt.nn.LayerNorm, eps=1e-6),
        window_size=0,
        input_size=(64, 64),
    )
    bg_out = block_global(jt.randn((2, 64, 64, 768)))
    check("block_global", bg_out, (2, 64, 64, 768))

    block_win = Block(
        dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(jt.nn.LayerNorm, eps=1e-6),
        window_size=14,
        input_size=(64, 64),
    )
    bw_out = block_win(jt.randn((2, 64, 64, 768)))
    check("block_window", bw_out, (2, 64, 64, 768))

    attn = Attention(dim=768, num_heads=12, qkv_bias=True, use_rel_pos=False)
    attn_out = attn(jt.randn((2, 16, 16, 768)))
    check("attention_no_rel", attn_out, (2, 16, 16, 768))

    attn_rel = Attention(dim=768, num_heads=12, qkv_bias=True, use_rel_pos=True, input_size=(16, 16))
    attn_rel_out = attn_rel(jt.randn((2, 16, 16, 768)))
    check("attention_rel_pos", attn_rel_out, (2, 16, 16, 768))

    wp_in = jt.randn((2, 64, 64, 768))
    windows, pad_hw = window_partition(wp_in, 14)
    check("window_partition", windows, (50, 14, 14, 768))  # 2*5*5=50
    wp_back = window_unpartition(windows, 14, pad_hw, (64, 64))
    check("window_unpartition", wp_back, (2, 64, 64, 768))

    rel_pos_small = jt.randn((15, 64))
    rp_out = get_rel_pos(16, 16, rel_pos_small)
    check("get_rel_pos_interp", rp_out, (16, 16, 64))

    print("test_image_encoder passed\n")


def test_prompt_encoder():
    print("Testing PromptEncoder...")

    # 按vit_b参数
    pe = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(1024, 1024),
        mask_in_chans=16,
    )

    # 1) get_dense_pe
    dense_pe = pe.get_dense_pe()
    check("get_dense_pe", dense_pe, (1, 256, 64, 64))

    # 2) 无输入
    sparse, dense = pe.execute(points=None, boxes=None, masks=None)
    check("prompt_no_input_sparse", sparse, (1, 0, 256))
    check("prompt_no_input_dense", dense, (1, 256, 64, 64))

    # 3) points only (会 padding 一个点)
    points = jt.randn((2, 3, 2))
    labels = jt.ones((2, 3))
    sparse_p, dense_p = pe.execute(points=(points, labels), boxes=None, masks=None)
    check("prompt_points_sparse", sparse_p, (2, 4, 256))   # 3+1 pad
    check("prompt_points_dense", dense_p, (2, 256, 64, 64))

    # 4) boxes only
    boxes = jt.randn((2, 4))
    sparse_b, dense_b = pe.execute(points=None, boxes=boxes, masks=None)
    check("prompt_boxes_sparse", sparse_b, (2, 2, 256))    # 每个 box 2 corners
    check("prompt_boxes_dense", dense_b, (2, 256, 64, 64))

    # 5) masks only
    masks = jt.randn((2, 1, 256, 256))
    sparse_m, dense_m = pe.execute(points=None, boxes=None, masks=masks)
    check("prompt_masks_sparse", sparse_m, (2, 0, 256))
    check("prompt_masks_dense", dense_m, (2, 256, 64, 64))

    # 6) combined (points + boxes + masks)
    sparse_c, dense_c = pe.execute(points=(points, labels), boxes=boxes, masks=masks)
    check("prompt_combined_sparse", sparse_c, (2, 5, 256))  # 3 points + 2 box corners
    check("prompt_combined_dense", dense_c, (2, 256, 64, 64))

    # 7) PositionEmbeddingRandom 独立形状
    per = PositionEmbeddingRandom(num_pos_feats=128)
    per_out = per.execute((64, 64))
    check("pos_embed_random", per_out, (256, 64, 64))

    print("test_prompt_encoder passed\n")


def test_mask_decoder():
    print("Testing MaskDecoder...")

    transformer = TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8)
    decoder = MaskDecoder(
        transformer_dim=256,
        transformer=transformer,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )

    # SAM 原始流程：image_embeddings 通常是 (1, C, H, W)，然后按 prompt batch 重复
    img_emb = jt.randn((1, 256, 64, 64))
    img_pe = jt.randn((1, 256, 64, 64))
    sparse = jt.randn((2, 5, 256))
    dense = jt.randn((2, 256, 64, 64))

    # 1) multimask_output=True
    masks, iou = decoder.execute(
        image_embeddings=img_emb,
        image_pe=img_pe,
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=True,
    )
    check("mask_decoder multimask", masks, (2, 3, 256, 256))
    check("mask_decoder iou_pred", iou, (2, 3))

    # 2) multimask_output=False
    masks_s, iou_s = decoder.execute(
        image_embeddings=img_emb,
        image_pe=img_pe,
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    check("mask_decoder single_mask", masks_s, (2, 1, 256, 256))
    check("mask_decoder single_iou", iou_s, (2, 1))

    # 3) MLP 独立测试
    mlp = MLP(input_dim=256, hidden_dim=128, output_dim=64, num_layers=3)
    mlp_out = mlp(jt.randn((2, 256)))
    check("mlp", mlp_out, (2, 64))

    print("test_mask_decoder passed\n")


def test_mynet():
    print("Testing MyNet components...")

    # 1) BasicConv2d
    bc = BasicConv2d(32, 16, kernel_size=3, padding=1)
    bc_out = bc(jt.randn((2, 32, 16, 16)))
    check("BasicConv2d", bc_out, (2, 16, 16, 16))

    # 2) DWConv
    dw = DWConv(dim=32)
    dw_out = dw(jt.randn((2, 32, 16, 16)))
    check("DWConv", dw_out, (2, 32, 16, 16))

    jt.flags.use_cuda = 1
    attn_sd = Attention_SD(dim=32, num_heads=2)
    x_rgb = jt.randn((2, 32, 16, 16))
    x_depth = jt.randn((2, 32, 16, 16))
    attn_out = attn_sd(x_rgb, x_depth)
    check("Attention_SD", attn_out, (2, 32, 16, 16))

    # 4) FM
    fm = FM(dim=64, oup=32)
    fm_out = fm(jt.randn((2, 64, 16, 16)))
    check("FM", fm_out, (2, 32, 16, 16))

    # 5) MEF
    mef = MEF(in1=32, in2=64)
    mef_out = mef(jt.randn((2, 32, 16, 16)), jt.randn((2, 64, 8, 8)))
    check("MEF", mef_out, (2, 32, 16, 16))

    # 6) Decode
    decode = Decode(in1=32, in2=32, in3=32, in4=32)
    out, br1 = decode(
        jt.randn((2, 32, 16, 16)),
        jt.randn((2, 32, 8, 8)),
        jt.randn((2, 32, 4, 4)),
        jt.randn((2, 32, 2, 2)),
    )
    check("Decode out", out, (2, 1, 32, 32))
    check("Decode br1", br1, (2, 32, 16, 16))

    # 7) conv helpers
    c1 = conv1x1(32, 16)
    check("conv1x1", c1(jt.randn((2, 32, 16, 16))), (2, 16, 16, 16))

    c3 = conv3x3(32, 16)
    check("conv3x3", c3(jt.randn((2, 32, 16, 16))), (2, 16, 16, 16))

    c1r = conv1x1_bn_relu(32, 16)
    check("conv1x1_bn_relu", c1r(jt.randn((2, 32, 16, 16))), (2, 16, 16, 16))

    c3r = conv3x3_bn_relu(32, 16)
    check("conv3x3_bn_relu", c3r(jt.randn((2, 32, 16, 16))), (2, 16, 16, 16))

    print("test_mynet passed\n")


def test_build_sam_style_integration():
    print("Testing build_sam-style integration flow...")

    # 用 build_sam_vit_b 的参数分别构造各组件，并做端到端前向
    encoder = ImageEncoderViT(
        depth=12, embed_dim=768, img_size=1024, mlp_ratio=4,
        norm_layer=partial(jt.nn.LayerNorm, eps=1e-6),
        num_heads=12, patch_size=16, qkv_bias=True,
        use_rel_pos=True, global_attn_indexes=[2, 5, 8, 11],
        window_size=14, out_chans=256,
    )
    prompt_encoder = PromptEncoder(
        embed_dim=256, image_embedding_size=(64, 64),
        input_image_size=(1024, 1024), mask_in_chans=16,
    )
    transformer = TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8)
    mask_decoder = MaskDecoder(
        num_multimask_outputs=3, transformer=transformer,
        transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256,
    )

    # 模拟一次完整推理
    img = jt.randn((1, 3, 1024, 1024))
    img_emb, _ = encoder(img)
    check("integration img_emb", img_emb, (1, 256, 64, 64))

    sparse, dense = prompt_encoder.execute(points=None, boxes=None, masks=None)
    masks, iou = mask_decoder.execute(
        image_embeddings=img_emb,
        image_pe=prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=True,
    )
    check("integration masks", masks, (1, 3, 256, 256))
    check("integration iou", iou, (1, 3))

    print("test_build_sam_style_integration passed\n")



if __name__ == "__main__":
    # test_registry()
    # test_image_encoder()
    # test_prompt_encoder()
    # test_mask_decoder()
    test_mynet()
    # test_build_sam_style_integration()
    print("All tests passed!")