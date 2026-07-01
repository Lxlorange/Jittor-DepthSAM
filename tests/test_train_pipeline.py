import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import jittor as jt

from data_cod import test_dataset
from train import get_loader, structure_loss
from utils.dataset_rgb_strategy2 import SalObjDataset
from utils.utils import AvgMeter


def _write_rgb(path, size=(40, 36), value=128):
    arr = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _write_gray(path, size=(40, 36), value=255):
    arr = np.full((size[1], size[0]), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _prepare_cod_tree(root):
    train_img = root / "Train_depth" / "Imgs"
    train_gt = root / "Train_depth" / "GT"
    train_depth = root / "Train_depth" / "depth"
    test_img = root / "Test_depth" / "CAMO" / "Imgs"
    test_gt = root / "Test_depth" / "CAMO" / "GT"
    test_depth = root / "Test_depth" / "CAMO" / "depth"
    for path in [train_img, train_gt, train_depth, test_img, test_gt, test_depth]:
        path.mkdir(parents=True, exist_ok=True)

    for i in range(2):
        name = f"sample_{i}.png"
        _write_rgb(train_img / name, value=80 + i)
        _write_gray(train_gt / name, value=255)
        _write_gray(train_depth / name, value=120)
        _write_rgb(test_img / name, value=90 + i)
        _write_gray(test_gt / name, value=255)
        _write_gray(test_depth / name, value=120)

    return train_img, train_gt, train_depth, test_img, test_gt, test_depth


def check(name, got, expected):
    actual = tuple(got.shape)
    print(f"test {name} got {actual}, expect {expected}")
    assert actual == expected, f"{name}: got {actual}, expect {expected}"


def test_train_dataset_and_loss():
    root = Path("tests_tmp_train_pipeline")
    if root.exists():
        shutil.rmtree(root)
    try:
        train_img, train_gt, train_depth, test_img, test_gt, test_depth = _prepare_cod_tree(root)

        dataset = SalObjDataset(
            str(train_img) + os.sep,
            str(train_gt) + os.sep,
            str(train_depth) + os.sep,
            trainsize=32,
        )
        image, gt, depth = dataset[0]
        check("dataset image", image, (3, 32, 32))
        check("dataset gt", gt, (1, 32, 32))
        check("dataset depth", depth, (1, 32, 32))

        loader, sampler = get_loader(
            str(train_img) + os.sep,
            str(train_gt) + os.sep,
            str(train_depth) + os.sep,
            batchsize=2,
            trainsize=32,
        )
        assert sampler is None
        images, gts, depths = next(iter(loader))
        check("loader images", images, (2, 3, 32, 32))
        check("loader gts", gts, (2, 1, 32, 32))
        check("loader depths", depths, (2, 1, 32, 32))

        pred = jt.zeros_like(gts)
        loss = structure_loss(pred, gts)
        assert tuple(loss.shape) == (), f"loss should be scalar, got {loss.shape}"

        meter = AvgMeter()
        meter.update(loss, n=2)
        shown = meter.show()
        assert hasattr(shown, "shape"), f"AvgMeter.show should return a Jittor value, got {type(shown)}"

        eval_loader = test_dataset(
            str(test_img) + os.sep,
            str(test_gt) + os.sep,
            str(test_depth) + os.sep,
            testsize=32,
        )
        eval_image, eval_gt, eval_depth, name, image_for_post = eval_loader.load_data()
        check("eval image", eval_image, (1, 3, 32, 32))
        check("eval depth", eval_depth, (1, 1, 32, 32))
        assert eval_gt.size == (40, 36)
        assert name.endswith(".png")
        assert image_for_post.shape == (36, 40, 3)
    finally:
        if root.exists():
            shutil.rmtree(root)


if __name__ == "__main__":
    test_train_dataset_and_loss()
    print("All train pipeline tests passed!")
