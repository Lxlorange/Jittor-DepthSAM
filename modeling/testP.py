from prompt_encoder import *
if __name__ == "__main__":

    pe = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(256, 256),
        input_image_size=(512, 512),
        mask_in_chans=16,
    )

    dense_pe = pe.get_dense_pe()
    print(dense_pe.shape)
    assert dense_pe.shape == (1, 256, 256, 256)

    # sparse, dense = pe(points=None, boxes=None, masks=None)


    mask_prompt = jt.randn(1, 1, 512, 512)

    image_embeddings = jt.randn(1,256,256,256)
    image_pe = pe.get_dense_pe()

    sparse, dense = pe(points=None, boxes=None, masks=mask_prompt)
    print(sparse.shape)
    print(dense.shape)

    assert image_embeddings.shape == dense.shape
    assert image_pe.shape == image_embeddings.shape

    src = image_embeddings + dense
    print(src.shape)
