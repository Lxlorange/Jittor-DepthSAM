import argparse
import json
import time

import torch
import torch.nn.functional as F

from segment_anything_training.build_DepthSAM import build_sam_DepthSAM


def structure_loss(pred, mask, eps=1e-8):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = torch.sum(weit * wbce, dim=(2, 3)) / (torch.sum(weit, dim=(2, 3)) + eps)

    pred = torch.sigmoid(pred)
    inter = torch.sum((pred * mask) * weit, dim=(2, 3))
    union = torch.sum((pred + mask) * weit, dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1 + eps)
    return (wbce + wiou).mean()


def moe_aux_loss(model, alpha=0.01):
    total_aux_loss = 0.0
    for module in model.modules():
        if hasattr(module, "current_aux_loss"):
            total_aux_loss += module.current_aux_loss
    return alpha * total_aux_loss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["forward", "train"], default="train")
    parser.add_argument("--batchsize", type=int, default=2)
    parser.add_argument("--trainsize", type=int, default=384)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def make_batched_input(images, device):
    imgs = images.permute(0, 2, 3, 1).detach().cpu().numpy()
    batched_input = []
    for b_i in range(len(imgs)):
        input_image = (
            torch.as_tensor((imgs[b_i]).astype(dtype="uint8"), device=device)
            .permute(2, 0, 1)
            .contiguous()
        )
        batched_input.append(
            {
                "image": input_image,
                "original_size": imgs[b_i].shape[:2],
            }
        )
    return batched_input


def run_step(model, optimizer, images, gts, mode):
    batched_input = make_batched_input(images, model.device)
    if mode == "forward":
        with torch.no_grad():
            return model(batched_input, images)

    pred = model(batched_input, images)
    loss = structure_loss(pred, gts) + moe_aux_loss(model)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss


def main():
    args = get_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for comparable FPS benchmarking.")

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")

    model = build_sam_DepthSAM(image_size=args.trainsize).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    images = torch.randn(args.batchsize, 3, args.trainsize, args.trainsize, device=device)
    gts = torch.rand(args.batchsize, 1, args.trainsize, args.trainsize, device=device)
    torch.cuda.synchronize()

    for _ in range(args.warmup):
        run_step(model, optimizer, images, gts, args.mode)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(args.iters):
        run_step(model, optimizer, images, gts, args.mode)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    samples = args.batchsize * args.iters
    result = {
        "framework": "torch",
        "mode": args.mode,
        "batchsize": args.batchsize,
        "trainsize": args.trainsize,
        "warmup": args.warmup,
        "iters": args.iters,
        "elapsed_sec": elapsed,
        "fps": samples / elapsed,
        "ms_per_iter": elapsed * 1000 / args.iters,
        "peak_memory_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
