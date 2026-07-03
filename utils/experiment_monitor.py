import csv
import json
import os
import subprocess
import time
from datetime import datetime

import numpy as np


class ExperimentMonitor:
    def __init__(self, name, root="./runs", config=None):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.name = name
        self.run_dir = os.path.join(root, f"{timestamp}_{name}")
        self.visual_dir = os.path.join(self.run_dir, "visuals")
        os.makedirs(self.visual_dir, exist_ok=True)

        self.start_time = time.time()
        self.loss_steps = []
        self.loss_values = []
        self.peak_gpu_mem_mb = 0
        self.peak_gpu_util = 0
        self.gpu_sample_interval = max(1, int(os.environ.get("MONITOR_GPU_INTERVAL", "50")))
        self._last_gpu = {"mem_used_mb": "", "mem_total_mb": "", "util": ""}

        if config is not None:
            self.write_json("config.json", config)

        self.train_csv = os.path.join(self.run_dir, "train_log.csv")
        self.eval_csv = os.path.join(self.run_dir, "eval_log.csv")
        self._init_csv(
            self.train_csv,
            ["time", "elapsed_sec", "epoch", "step", "total_step", "loss", "lr", "gpu_mem_mb", "gpu_util"],
        )
        self._init_csv(
            self.eval_csv,
            ["time", "elapsed_sec", "dataset", "name", "mae", "skipped", "gpu_mem_mb", "gpu_util"],
        )

    def _init_csv(self, path, header):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    def _append_csv(self, path, row):
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def write_json(self, filename, data):
        with open(os.path.join(self.run_dir, filename), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def elapsed(self):
        return time.time() - self.start_time

    def gpu_snapshot(self):
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            first = out.strip().splitlines()[0]
            mem_used, mem_total, util = [int(x.strip()) for x in first.split(",")]
            self.peak_gpu_mem_mb = max(self.peak_gpu_mem_mb, mem_used)
            self.peak_gpu_util = max(self.peak_gpu_util, util)
            return {"mem_used_mb": mem_used, "mem_total_mb": mem_total, "util": util}
        except Exception:
            return {"mem_used_mb": "", "mem_total_mb": "", "util": ""}

    def log_train_step(self, epoch, step, total_step, loss, lr):
        if step == 1 or step == total_step or step % self.gpu_sample_interval == 0:
            self._last_gpu = self.gpu_snapshot()
        gpu = self._last_gpu
        loss_value = float(loss)
        global_step = (epoch - 1) * total_step + step
        self.loss_steps.append(global_step)
        self.loss_values.append(loss_value)
        self._append_csv(
            self.train_csv,
            [
                datetime.now().isoformat(timespec="seconds"),
                round(self.elapsed(), 3),
                epoch,
                step,
                total_step,
                loss_value,
                lr,
                gpu["mem_used_mb"],
                gpu["util"],
            ],
        )

    def log_eval_sample(self, dataset, name, mae=None, skipped=False):
        gpu = self.gpu_snapshot()
        self._append_csv(
            self.eval_csv,
            [
                datetime.now().isoformat(timespec="seconds"),
                round(self.elapsed(), 3),
                dataset,
                name,
                "" if mae is None else float(mae),
                int(skipped),
                gpu["mem_used_mb"],
                gpu["util"],
            ],
        )

    def save_prediction_panel(self, dataset, name, image, pred, gt):
        try:
            import cv2
        except Exception:
            return

        out_dir = os.path.join(self.visual_dir, dataset)
        os.makedirs(out_dir, exist_ok=True)

        image = np.asarray(image)
        pred = np.asarray(pred)
        gt = np.asarray(gt)

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        pred_img = (pred * 255).clip(0, 255).astype(np.uint8)
        gt_img = (gt * 255).clip(0, 255).astype(np.uint8)
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w = pred_img.shape[:2]
        image = cv2.resize(image, (w, h))
        panel = np.concatenate([image, pred_img, gt_img], axis=1)
        out_path = os.path.join(out_dir, name)
        if not cv2.imwrite(out_path, panel):
            print(f"Warning: failed to write prediction panel: {out_path}")

    def save_loss_curve(self):
        if not self.loss_steps:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(self.loss_steps, self.loss_values)
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.tight_layout()
            plt.savefig(os.path.join(self.run_dir, "loss_curve.png"), dpi=150)
            plt.close()
        except Exception:
            pass

    def finish(self, extra=None):
        self.save_loss_curve()
        summary = {
            "name": self.name,
            "run_dir": self.run_dir,
            "elapsed_sec": round(self.elapsed(), 3),
            "peak_gpu_mem_mb": self.peak_gpu_mem_mb,
            "peak_gpu_util": self.peak_gpu_util,
        }
        if extra:
            summary.update(extra)
        self.write_json("summary.json", summary)
        return summary
