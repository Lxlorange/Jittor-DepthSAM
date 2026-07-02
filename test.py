import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import jittor as jt
import jittor.nn as nn
import numpy as np
import cv2
from segment_anything_training.build_DepthSAM import build_sam_DepthSAM
from data_cod import test_dataset
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='epoch number')
    parser.add_argument('--lr_gen', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=512, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
    parser.add_argument('-beta1_gen', type=float, default=0.5, help='beta of Adam for generator')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
    parser.add_argument('--feat_channel', type=int, default=64, help='reduced channel of saliency feat')
    parser.add_argument('--gpu', type=int, default='0', help='reduced channel of saliency feat')
    return parser.parse_args()
opt = get_args()

print('USE GPU', opt.gpu)
jt.flags.use_cuda = 1

def test():

    dataset_path = './Data_all/COD-D/Test_depth/'
    test_datasets = ['CAMO']
    generator = build_sam_DepthSAM(image_size=opt.trainsize)
    npz = np.load('./checkpoints/Model_1_gen.npz')
    data = {key: jt.array(npz[key]) for key in npz.files}
    if list(data.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in data.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        generator.load_state_dict(new_state_dict)
    else:
        generator.load_state_dict(data)

    generator.train()
    for dataset in test_datasets:
        save_path = './test_maps/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        image_root = dataset_path + dataset + '/Imgs/'
        gt_root = dataset_path + dataset + '/GT/'
        g_root = dataset_path + dataset + '/depth/'
        test_loader = test_dataset(image_root, gt_root, g_root, opt.trainsize)

        mae_sum = 0
        test_count = 0
        for i in tqdm(range(test_loader.size)):
            # if i >= 10:
            #     break
            image, gt, depth, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            depth_np = np.asarray(depth, np.float32) if not isinstance(depth, np.ndarray) else depth
            print(f"[{name}] depth min={depth_np.min():.4f}, max={depth_np.max():.4f}, "
                f"has zero={(depth_np==0).sum()}, has nan={np.isnan(depth_np).any()}")

            image = jt.array(image)
            depth = jt.array(depth)
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            if len(depth.shape) == 3:
                depth = depth.unsqueeze(0)

            batched_input = []
            for b_i in range(image.shape[0]):
                input_image = image[b_i]
                batched_input.append(
                    {
                        'image': input_image,
                        'original_size': (input_image.shape[1], input_image.shape[2]),
                    }
                )

            res = generator(batched_input, image)
            target_h, target_w = gt.shape[-2], gt.shape[-1]
            if res.shape[-2] != target_h or res.shape[-1] != target_w:
                res = nn.upsample(res, size=(target_h, target_w), mode='bilinear', align_corners=False)
            res = res.sigmoid().numpy().squeeze()
            if np.isnan(res).any():
                print(f"Warning: NaN in output for {name}, skipping")
                continue
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            out_img = (res * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(save_path + name, out_img)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[-2] * gt.shape[-1])
            test_count += 1

        mae = mae_sum / test_count
        print(dataset, 'mae is : ', mae)

        
if __name__ == '__main__':
    test()

