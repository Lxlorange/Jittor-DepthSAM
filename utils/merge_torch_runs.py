import argparse
import csv
import json
import os
import shutil
from datetime import datetime


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def write_csv_rows(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(rows)


def dedupe_train_rows(header, rows):
    try:
        epoch_idx = header.index("epoch")
        step_idx = header.index("step")
    except ValueError:
        return rows
    merged = {}
    order = []
    for row in rows:
        if len(row) <= max(epoch_idx, step_idx):
            continue
        key = (row[epoch_idx], row[step_idx])
        if key not in merged:
            order.append(key)
        merged[key] = row

    def sort_key(key):
        try:
            return int(key[0]), int(key[1])
        except ValueError:
            return key

    return [merged[key] for key in sorted(order, key=sort_key)]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def merge_csv(target_dir, filenames):
    merged = {}
    for name in filenames:
        path = os.path.join(target_dir, name)
        if not os.path.exists(path):
            continue
        header, rows = read_csv_rows(path)
        if name not in merged:
            merged[name] = (header, [])
        merged[name][1].extend(rows)

    for name, (header, rows) in merged.items():
        write_csv_rows(os.path.join(target_dir, name), header, rows)


def plot_loss_curve_from_train_csv(target_dir):
    path = os.path.join(target_dir, "train_log.csv")
    if not os.path.exists(path):
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    steps = []
    losses = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) <= 1:
        return
    header = rows[0]
    try:
        step_idx = header.index("elapsed_sec")
        loss_idx = header.index("loss")
    except ValueError:
        return
    for row in rows[1:]:
        if len(row) <= max(step_idx, loss_idx):
            continue
        try:
            steps.append(float(row[step_idx]))
            losses.append(float(row[loss_idx]))
        except ValueError:
            continue
    if not steps:
        return
    plt.figure()
    plt.plot(steps, losses)
    plt.xlabel("elapsed_sec")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, "loss_curve.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run1", required=True, help="first run directory")
    parser.add_argument("--run2", required=True, help="second run directory")
    parser.add_argument("--output", default="", help="merged output directory")
    args = parser.parse_args()

    run1 = os.path.abspath(args.run1)
    run2 = os.path.abspath(args.run2)
    output = os.path.abspath(args.output) if args.output else os.path.join(
        os.path.dirname(run1),
        f"merged_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    os.makedirs(output, exist_ok=True)

    for name in ["train_log.csv", "eval_log.csv"]:
        rows = []
        header = None
        for run in [run1, run2]:
            path = os.path.join(run, name)
            if os.path.exists(path):
                h, r = read_csv_rows(path)
                if header is None:
                    header = h
                rows.extend(r)
        if header is not None:
            if name == "train_log.csv":
                rows = dedupe_train_rows(header, rows)
            write_csv_rows(os.path.join(output, name), header, rows)
    plot_loss_curve_from_train_csv(output)

    configs = []
    for run in [run1, run2]:
        path = os.path.join(run, "config.json")
        if os.path.exists(path):
            configs.append(load_json(path))
    if configs:
        dump_json(os.path.join(output, "config_merged.json"), {
            "runs": configs,
            "merged_at": datetime.now().isoformat(timespec="seconds"),
        })

    summaries = []
    for run in [run1, run2]:
        path = os.path.join(run, "summary.json")
        if os.path.exists(path):
            summaries.append(load_json(path))
    if summaries:
        dump_json(os.path.join(output, "summary_merged.json"), {
            "runs": summaries,
            "merged_at": datetime.now().isoformat(timespec="seconds"),
        })

    for run in [run1, run2]:
        vis_dir = os.path.join(run, "visuals")
        if not os.path.isdir(vis_dir):
            continue
        for dataset in os.listdir(vis_dir):
            src = os.path.join(vis_dir, dataset)
            dst = os.path.join(output, "visuals", dataset)
            if os.path.isdir(src):
                os.makedirs(dst, exist_ok=True)
                for name in os.listdir(src):
                    src_file = os.path.join(src, name)
                    dst_file = os.path.join(dst, f"{os.path.basename(run)}_{name}")
                    if os.path.isfile(src_file):
                        shutil.copy2(src_file, dst_file)

    print(output)


if __name__ == "__main__":
    main()
