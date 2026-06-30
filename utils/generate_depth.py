import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet


def build_model(ckpt_path, device):
    model = DepthAnythingV2()
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model

def build_transform(input_size):
    return Compose([
        Resize(
            width=input_size,
            height=input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

@torch.no_grad()
def gen_depth(model, transform, image_path, device):
    raw_bgr = cv2.imread(str(image_path))
    if raw_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    h, w = raw_bgr.shape[:2]
    image = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB) / 255.0
    image = transform({"image": image})["image"]
    image = torch.from_numpy(image).unsqueeze(0).to(device)

    depth, _ = model(image)
    # print(depth.shape)

    if depth.ndim == 4:
        depth = depth[:, 0]
    depth = F.interpolate(depth[:, None], size=(h, w), mode="bilinear", align_corners=True)[0, 0]

    depth = depth.detach().float().cpu().numpy()
    depth_min, depth_max = depth.min(), depth.max()
    if depth_max - depth_min < 1e-6:
        depth_u16 = np.zeros_like(depth, dtype=np.uint16)
    else:
        depth_u16 = ((depth - depth_min) / (depth_max - depth_min) * 65535.0).astype(np.uint16)

    return depth_u16


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("checkpoints/depth_anything_v2_vitl.pth",device)
    input_size = 518
    transform = build_transform(input_size)

    data_set = [
        {"img":Path("Data_all/COD-D/Train_depth/Imgs"),"out":Path("Data_all/COD-D/Train_depth/depth")},
        {"img":Path("Data_all/COD-D/Test_depth/CAMO/Imgs"),"out":Path("Data_all/COD-D/Test_depth/CAMO/depth")},
        {"img":Path("Data_all/COD-D/Test_depth/CHAMELEON/Imgs"),"out":Path("Data_all/COD-D/Test_depth/CHAMELEON/depth")},
        {"img":Path("Data_all/COD-D/Test_depth/COD10K/Imgs"),"out":Path("Data_all/COD-D/Test_depth/COD10K/depth")},
        {"img":Path("Data_all/COD-D/Test_depth/NC4K/Imgs"),"out":Path("Data_all/COD-D/Test_depth/NC4K/depth")}
    ]

    for ds in data_set:
        img_dir = ds["img"]
        out_dir = ds["out"]
        out_dir.mkdir(parents=True, exist_ok=True)

        image_paths = []
        for suffix in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG", "*.JPEG"]:
            image_paths.extend(img_dir.glob(suffix))
        image_paths.sort()

        out_num = 0
        for image in tqdm(image_paths):
            out_path = out_dir / f"{image.stem}.png"
            if out_path.exists() and out_path.stat().st_size > 0:
                continue
            out_num += 1
            depth = gen_depth(model, transform, image, device)
            cv2.imwrite(str(out_path),depth)

        print(f"done {out_num} images, total {len(image_paths)} in {ds}")


if __name__ == "__main__":
    main()
