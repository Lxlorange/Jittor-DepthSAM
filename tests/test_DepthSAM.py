import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import jittor as jt
import jittor.nn as nn
from segment_anything_training.modeling import TwoWayTransformer,MaskDecoder,PromptEncoder

def check(name, x, shape):
    if isinstance(x, (tuple, list)):
        shapes = tuple(tuple(v.shape) if hasattr(v, 'shape') else str(type(v)) for v in x)
        print(f"test {name} got {shapes}, expect {shape}")
        assert shapes == shape, f"test {name} got {shapes}, expect {shape}"
    else:
        actual = tuple(x.shape)
        print(f"test {name} got {actual}, expect {shape}")
        assert actual == shape, f"test {name} got {actual}, expect {shape}"

def test_transformer():
    print("Test transformer...")
    net = TwoWayTransformer(
        depth=2,
        embedding_dim=256,
        num_heads=8,
        mlp_dim=2048
    )
    src = jt.randn((1,256,16,16))
    pos = jt.randn((1,256,16,16))
    tokens = jt.randn((1,5,256))

    hs,out = net(src,pos,tokens)
    check("hs",hs,(1,5,256))
    check("out",out,(1,256,256))
    print("test_transformer passes\n")

def test_depthsam_decoder():
    print("Test DepthSAM_decoder...")
    net = TwoWayTransformer(
        depth=2,
        embedding_dim=256,
        num_heads=8,
        mlp_dim=2048
    )

    decoder = MaskDecoder(
        transformer=net,
        transformer_dim=256
    )
    masks,iou = decoder(
        image_embeddings=jt.randn((1,256,16,16)),
        image_pe=jt.randn((1,256,16,16)),
        sparse_prompt_embeddings = jt.randn((1,0,256)),
        dense_prompt_embeddings=jt.randn((1,256,16,16)),
    )
    check("masks",masks,(1,1,32,32))
    check("iou",iou,(1,1,32,32))
    print("test_depthsam_decoder passes\n")

def test_P_M():
    pe = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(16, 16),
        input_image_size=(32, 32),
        mask_in_chans=16,
    )

    decoder = MaskDecoder(
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=256,
    )

    mask_prompt = jt.randn((1, 1, 32, 32))
    sparse, dense = pe(points=None, boxes=None, masks=mask_prompt)
    image_embeddings = jt.randn((1, 256, 16, 16))
    image_pe = pe.get_dense_pe()
    masks, iou = decoder(image_embeddings, image_pe, sparse, dense)

    check("sparse",sparse,(1,0,256))
    check("dense",dense,(1,256,16,16))
    check("image_pe",image_pe,(1,256,16,16))
    check("masks",masks,(1,1,32,32))
    print("test_depthsam_decoder passes\n")


if __name__ == "__main__":
    # test_transformer()
    # test_depthsam_decoder()
    test_P_M()

    print("All tests passed!")