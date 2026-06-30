from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.dinov2_layers import PatchEmbed
import jittor as jt
import numpy as np
if __name__ == "__main__":
    # d = DepthAnythingV2()
    # d(jt.randn((1,3,518,518)))
    # raw_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # d.infer_image(raw_image)
    p = PatchEmbed()
    p(jt.randn((1,3,518,518)))