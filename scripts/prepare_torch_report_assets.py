import csv
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TORCH_ROOT = ROOT / "runs" / "torch"
TRAIN_RUN = TORCH_ROOT / "runs" / "merged_torch_train_120"
TEST_RUN = TORCH_ROOT / "runs" / "20260704-165209_torch_test"
BENCH_RUN = TORCH_ROOT / "runs" / "torch_benchmark_120"
OUT_DIR = ROOT / "report" / "image" / "torch"


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def read_csv_dicts(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def plot_loss_curve():
    rows = read_csv_dicts(TRAIN_RUN / "train_log.csv")
    xs = []
    ys = []
    for idx, row in enumerate(rows, start=1):
        try:
            xs.append(idx)
            ys.append(float(row["loss"]))
        except Exception:
            continue

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4.2))
    plt.plot(xs, ys, linewidth=0.8, color="#1c4682")
    plt.xlabel("Logged step")
    plt.ylabel("Training loss")
    plt.title("PyTorch Training Loss (120 epochs)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "torch_loss_curve.png", dpi=180)
    plt.close()


def export_mae_table():
    with open(TEST_RUN / "test_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    order = ["CAMO", "CHAMELEON", "COD10K", "NC4K"]
    rows = []
    for dataset in order:
        item = results[dataset]
        rows.append([dataset, item["count"], f'{item["mae"]:.5f}'])

    write_csv(OUT_DIR / "torch_mae_table.csv", ["Dataset", "Count", "MAE"], rows)
    with open(OUT_DIR / "torch_mae_table.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\nDataset & Count & MAE \\\\\n\\midrule\n")
        for dataset, count, mae in rows:
            f.write(f"{dataset} & {count} & {mae} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def export_benchmark_table():
    rows = []
    for path in sorted(BENCH_RUN.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            item = json.load(f)
        rows.append([
            item["mode"],
            item["batchsize"],
            f'{item["fps"]:.2f}',
            f'{item["ms_per_iter"]:.2f}',
            f'{item["peak_memory_mb"]:.2f}',
        ])

    write_csv(
        OUT_DIR / "torch_benchmark_table.csv",
        ["Mode", "Batch", "FPS", "ms/iter", "Peak memory MB"],
        rows,
    )
    with open(OUT_DIR / "torch_benchmark_table.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\nMode & Batch & FPS & ms/iter & Peak Mem(MB) \\\\\n\\midrule\n")
        for mode, batch, fps, ms, mem in rows:
            f.write(f"{mode} & {batch} & {fps} & {ms} & {mem} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def plot_benchmark_bar():
    items = []
    for path in sorted(BENCH_RUN.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            item = json.load(f)
        items.append((f'{item["mode"]}-b{item["batchsize"]}', float(item["fps"])))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [x[0] for x in items]
    values = [x[1] for x in items]
    plt.figure(figsize=(7.5, 4))
    plt.bar(labels, values, color="#1c4682")
    plt.ylabel("FPS")
    plt.title("PyTorch FPS Benchmark")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "torch_fps_benchmark.png", dpi=180)
    plt.close()


def make_contact_sheet(dataset, max_images=2):
    try:
        from PIL import Image
    except Exception:
        return

    src_dir = TEST_RUN / "visuals" / dataset
    files = [
        p for p in sorted(src_dir.glob("*.png"))
        if p.is_file() and ".ipynb_checkpoints" not in str(p)
    ][:max_images]
    if not files:
        return

    panels = []
    target_width = 1050
    for path in files:
        img = Image.open(path).convert("RGB")
        scale = target_width / img.width
        target_height = max(1, int(img.height * scale))
        img = img.resize((target_width, target_height))
        panels.append((path, img))
        shutil.copy2(path, OUT_DIR / f"torch_{dataset}_{path.name}")

    gap = 14
    label_h = 28
    width = target_width
    height = sum(img.height + label_h for _, img in panels) + gap * (len(panels) - 1)
    sheet = Image.new("RGB", (width, height), "white")

    y = 0
    for path, img in panels:
        sheet.paste(img, (0, y + label_h))
        y += img.height + label_h + gap
    sheet.save(OUT_DIR / f"torch_{dataset}_panels.png")


def main():
    ensure_dir(OUT_DIR)
    plot_loss_curve()
    export_mae_table()
    export_benchmark_table()
    plot_benchmark_bar()
    for dataset in ["CAMO", "CHAMELEON", "COD10K", "NC4K"]:
        make_contact_sheet(dataset)
    print(OUT_DIR)


if __name__ == "__main__":
    main()
