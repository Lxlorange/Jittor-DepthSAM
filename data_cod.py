
import os

import numpy as np
from PIL import Image


# test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, depth_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp') or f.endswith('.png') or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = image.resize((self.testsize, self.testsize), Image.BILINEAR)
        image = np.array(image, dtype=np.float32)
        image = np.ascontiguousarray(image.transpose(2, 0, 1)) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        image = (image - mean) / std
        image = np.expand_dims(image, axis=0)   # (1, 3, H, W)

        gt = self.binary_loader(self.gts[self.index])
        gt_size = gt.size                         # 先保存PIL size，给image_for_post用
        gt = np.array(gt.resize((self.testsize, self.testsize), Image.NEAREST), dtype=np.float32)
        gt = np.ascontiguousarray(gt[np.newaxis, ...]) / 255.0   # (1, H, W)

        depth = self.binary_loader(self.depths[self.index])
        depth = depth.resize((self.testsize, self.testsize), Image.BILINEAR)
        depth = np.array(depth, dtype=np.float32)
        depth = np.ascontiguousarray(depth[np.newaxis, ...]) / 255.0
        depth = np.expand_dims(depth, axis=0)   # (1, 1, H, W)

        name = self.images[self.index].split('/')[-1]
        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt_size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size
        return image, gt, depth, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

