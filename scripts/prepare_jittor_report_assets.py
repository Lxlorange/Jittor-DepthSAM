import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TORCH_TRAIN = ROOT / "runs" / "torch" / "runs" / "merged_torch_train_120" / "train_log.csv"
JITTOR_TRAIN = ROOT / "runs" / "jittor" / "runs" / "20260704-131004_jittor_train" / "train_log.csv"
JITTOR_TESTS = [
    ROOT / "runs" / "jittor" / "runs" / "20260706-065723_jittor_test" / "eval_log.csv",
    ROOT / "runs" / "jittor" / "runs" / "20260706-070117_jittor_test" / "eval_log.csv",
]
JITTOR_VISUAL_ROOTS = [
    ROOT / "runs" / "jittor" / "runs" / "20260706-065723_jittor_test" / "visuals",
    ROOT / "runs" / "jittor" / "runs" / "20260706-070117_jittor_test" / "visuals",
]
TORCH_VISUAL_ROOT = ROOT / "runs" / "torch" / "runs" / "20260704-165209_torch_test" / "visuals"
OUT_DIR = ROOT / "report" / "image" / "jittor"


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def read_csv_dicts(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_loss(path):
    xs, ys = [], []
    for row in read_csv_dicts(path):
        try:
            epoch = float(row["epoch"])
            step = float(row["step"])
            total = max(float(row["total_step"]), 1.0)
            xs.append(epoch - 1.0 + step / total)
            ys.append(float(row["loss"]))
        except Exception:
            continue
    return xs, ys


def epoch_mean(xs, ys):
    buckets = {}
    for x, y in zip(xs, ys):
        epoch = int(x) + 1
        buckets.setdefault(epoch, []).append(y)
    out_x, out_y = [], []
    for epoch in sorted(buckets):
        vals = buckets[epoch]
        out_x.append(epoch)
        out_y.append(sum(vals) / len(vals))
    return out_x, out_y


def plot_loss_alignment():
    torch_x, torch_y = read_loss(TORCH_TRAIN)
    jittor_x, jittor_y = read_loss(JITTOR_TRAIN)
    torch_mx, torch_my = epoch_mean(torch_x, torch_y)
    jittor_mx, jittor_my = epoch_mean(jittor_x, jittor_y)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.6), sharex=True, sharey=True)
    items = [
        (axes[0], "PyTorch", torch_x, torch_y, torch_mx, torch_my, "#1c4682"),
        (axes[1], "Jittor", jittor_x, jittor_y, jittor_mx, jittor_my, "#b45309"),
    ]
    for ax, title, xs, ys, mx, my, color in items:
        ax.plot(xs, ys, linewidth=0.35, alpha=0.22, color=color, label="logged loss")
        ax.plot(mx, my, linewidth=2.0, color=color, label="epoch mean")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.22)
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 1.8)
    axes[0].set_ylabel("Loss")
    axes[1].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("Training Loss Alignment (same epoch/loss scale)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "loss_alignment.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def export_mae_table():
    datasets = {}
    for path in JITTOR_TESTS:
        for row in read_csv_dicts(path):
            if row.get("skipped") != "0":
                continue
            try:
                datasets.setdefault(row["dataset"], []).append(float(row["mae"]))
            except Exception:
                continue

    order = ["CAMO", "CHAMELEON", "COD10K", "NC4K"]
    rows = []
    for dataset in order:
        vals = datasets.get(dataset, [])
        if vals:
            rows.append([dataset, len(vals), f"{sum(vals) / len(vals):.5f}"])

    with open(OUT_DIR / "jittor_mae_table.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Count", "MAE"])
        writer.writerows(rows)

    with open(OUT_DIR / "jittor_mae_table.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\nDataset & Count & MAE \\\\\n\\midrule\n")
        for dataset, count, mae in rows:
            f.write(f"{dataset} & {count} & {mae} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def visual_root_for(dataset):
    for root in JITTOR_VISUAL_ROOTS:
        path = root / dataset
        if path.exists():
            return path
    return None


def make_contact_sheet(dataset, max_images=2):
    try:
        from PIL import Image
    except Exception:
        return

    src_dir = visual_root_for(dataset)
    if src_dir is None:
        return
    files = [p for p in sorted(src_dir.glob("*.png")) if p.is_file()][:max_images]
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
        shutil.copy2(path, OUT_DIR / f"jittor_{dataset}_{path.name}")

    gap = 14
    label_h = 28
    width = target_width
    height = sum(img.height + label_h for _, img in panels) + gap * (len(panels) - 1)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for _, img in panels:
        sheet.paste(img, (0, y + label_h))
        y += img.height + label_h + gap
    sheet.save(OUT_DIR / f"jittor_{dataset}_panels.png")


def split_panel(path):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    part_w = w // 3
    return [
        img.crop((0, 0, part_w, h)),
        img.crop((part_w, 0, part_w * 2, h)),
        img.crop((part_w * 2, 0, w, h)),
    ]


def make_framework_comparison():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return

    picks = [
        ("CAMO", "camourflage_00012.png", "20260706-065723_jittor_test"),
        ("CHAMELEON", "animal-1.png", "20260706-070117_jittor_test"),
        ("COD10K", "COD10K-CAM-1-Aquatic-1-BatFish-2.png", "20260706-070117_jittor_test"),
        ("NC4K", "1002.png", "20260706-070117_jittor_test"),
    ]
    try:
        font = ImageFont.truetype("arial.ttf", 17)
        small_font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    rows = []
    target_row_h = 118
    for dataset, filename, jittor_run in picks:
        torch_path = TORCH_VISUAL_ROOT / dataset / filename
        jittor_path = ROOT / "runs" / "jittor" / "runs" / jittor_run / "visuals" / dataset / filename
        if not torch_path.exists() or not jittor_path.exists():
            continue

        torch_origin, torch_pred, torch_gt = split_panel(torch_path)
        _, jittor_pred, _ = split_panel(jittor_path)

        cell_w = max(1, int(torch_origin.width * target_row_h / torch_origin.height))
        cell = (cell_w, target_row_h)
        parts = []
        for img in [torch_origin, torch_gt, torch_pred, jittor_pred]:
            parts.append(img.resize(cell, Image.Resampling.LANCZOS))
        rows.append((dataset, cell, parts))

    if not rows:
        return

    label_w = 112
    header_h = 28
    row_gap = 8
    max_cell_w = max(cell[0] for _, cell, _ in rows)
    width = label_w + 4 * max_cell_w
    height = header_h + sum(cell[1] for _, cell, _ in rows) + row_gap * (len(rows) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    headers = ["Origin", "GT", "PyTorch", "Jittor"]
    for col, text in enumerate(headers):
        x = label_w + col * max_cell_w + max_cell_w // 2
        draw.text((x, 6), text, fill=(30, 30, 30), font=font, anchor="ma")

    y = header_h
    for dataset, cell, parts in rows:
        draw.text((8, y + cell[1] // 2), dataset, fill=(20, 20, 20), font=small_font, anchor="lm")
        row_x0 = label_w + (max_cell_w - cell[0]) // 2
        for col, img in enumerate(parts):
            x = row_x0 + col * max_cell_w
            canvas.paste(img, (x, y))
        y += cell[1] + row_gap

    canvas.save(OUT_DIR / "qualitative_framework_comparison.png")


def main():
    ensure_dir(OUT_DIR)
    plot_loss_alignment()
    export_mae_table()
    for dataset in ["CAMO", "CHAMELEON", "COD10K", "NC4K"]:
        make_contact_sheet(dataset)
    make_framework_comparison()
    print(OUT_DIR)


if __name__ == "__main__":
    main()
