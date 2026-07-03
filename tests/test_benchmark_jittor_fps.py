import argparse
import json
import os
import time

import jittor as jt

from segment_anything_training.build_DepthSAM import build_sam_DepthSAM
from train import structure_loss, trainable_parameters
from utils.jittor_runtime import configure_jittor_runtime,print_runtime_hints

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["forward", "train"], default="train")
    parser.add_argument("--batchsize", type=int, default=2)
    parser.add_argument("--trainsize", type=int, default=384)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--profile", choices=["full", "encoder", "decoder"], default="full")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def make_batched_input(images):
    batched_input = []
    for b_i in range(images.shape[0]):
        input_image = images[b_i]
        batched_input.append(
            {
                "image": input_image,
                "original_size": (input_image.shape[1], input_image.shape[2]),
            }
        )
    return batched_input


def sync():
    jt.sync_all()
    jt.gc()


def peak_memory_mb():
    try:
        used = jt.flags.stat_allocator_total_alloc_byte
        return round(float(used) / 1024 / 1024, 2)
    except Exception:
        return None


def run_step(model, optimizer, images, gts, mode, profile):
    batched_input = make_batched_input(images)

    if profile == "encoder":
        pred = model.image_encoder(images)[1][0]
        if mode == "train":
            loss = pred.mean()
            optimizer.step(loss)
            return loss
        return pred

    if profile == "decoder":
        _, features = model.image_encoder(images)
        out1, out_1 = model.decoder(features[3], features[2], features[1], features[0])
        pred = out1
        if mode == "train":
            if pred.shape != gts.shape:
                gts = jt.rand(pred.shape)
            loss = structure_loss(pred, gts)
            optimizer.step(loss)
            return loss
        return pred
    
    if mode == "forward":
        with jt.no_grad():
            pred = model(batched_input, images)
        return pred
    
    pred = model(batched_input, images)
    loss = structure_loss(pred, gts)
    optimizer.step(loss)
    return loss


def main():
    args = get_args()
    configure_jittor_runtime()
    print_runtime_hints()

    model = build_sam_DepthSAM(image_size=args.trainsize)
    model.train() if args.mode == "train" else model.eval()
    optimizer = None
    if args.mode == "train":
        params = trainable_parameters(model)
        assert len(params) > 0, "No trainable parameters found! Check requires_grad settings."
        optimizer = jt.optim.Adam(params, args.lr)

    images = jt.randn((args.batchsize, 3, args.trainsize, args.trainsize))
    gts = jt.rand((args.batchsize, 1, args.trainsize, args.trainsize))
    sync()

    for _ in range(args.warmup):
        run_step(model, optimizer, images, gts, args.mode, args.profile)
        sync()

    start = time.perf_counter()
    for _ in range(args.iters):
        run_step(model, optimizer, images, gts, args.mode, args.profile)
        sync()
    elapsed = time.perf_counter() - start

    samples = args.batchsize * args.iters
    result = {
        "framework": "jittor",
        "mode": args.mode,
        "batchsize": args.batchsize,
        "trainsize": args.trainsize,
        "warmup": args.warmup,
        "iters": args.iters,
        "profile": args.profile,
        "elapsed_sec": elapsed,
        "fps": samples / elapsed,
        "ms_per_iter": elapsed * 1000 / args.iters,
        "peak_memory_mb": peak_memory_mb(),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
