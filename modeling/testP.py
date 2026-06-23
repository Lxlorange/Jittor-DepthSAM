from prompt_encoder import *

jt.flags.use_cuda = 1

def check(name,x,shape):
    print(f"test {name} got {x.shape}, expect {shape}")
    assert tuple(x.shape) == shape, f"test {name} got {x.shape}, expect {shape}"

if __name__ == "__main__":

    image_size = 512
    pe = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(image_size // 2, image_size //2),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )

    dense_pe = pe.get_dense_pe()
    check("dense_pe_512", dense_pe, (1, 256, 256, 256))

    sparse, dense = pe(points=None, boxes=None, masks=None)
    check("sparse_no_mask", sparse, (1, 0, 256))
    check("dense_no_mask", dense, (1, 256, 256, 256))

    mask = jt.randn((1, 1, 512, 512))
    sparse, dense = pe(points=None, boxes=None, masks=mask)
    check("sparse_mask_b1", sparse, (1, 0, 256))
    check("dense_mask_b1", dense, (1, 256, 256, 256))

    mask = jt.randn((2, 1, 512, 512))
    sparse, dense = pe(points=None, boxes=None, masks=mask)
    check("sparse_mask_b2", sparse, (2, 0, 256))
    check("dense_mask_b2", dense, (2, 256, 256, 256))

    image_embeddings = jt.randn((1, 256, 256, 256))
    mask = jt.randn((1, 1, 512, 512))

    sparse, dense = pe(points=None, boxes=None, masks=mask)
    image_pe = pe.get_dense_pe()

    check("image_embeddings", image_embeddings, (1, 256, 256, 256))
    check("dense_embeddings", dense, (1, 256, 256, 256))
    check("image_pe", image_pe, (1, 256, 256, 256))

    src = image_embeddings + dense
    check("decoder_src", src, (1, 256, 256, 256))

    pos = PositionEmbeddingRandom(128)
    coords = jt.array([[[0.0, 0.0], [256.0, 256.0], [511.0, 511.0]]])
    emb = pos.forward_with_coords(coords, (512, 512))
    check("coords_embedding", emb, (1, 3, 256))


