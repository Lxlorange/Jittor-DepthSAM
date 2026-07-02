import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import cv2
from segment_anything_training.build_DepthSAM import build_sam_DepthSAM
from data_cod import test_dataset
from utils.experiment_monitor import ExperimentMonitor
from tqdm import tqdm

def get_loader(image_root, gt_root, trainsize):
    dataset = test_dataset(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=0,
                                  pin_memory=True, )
    return data_loader


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
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

def test():

    dataset_path = './Data_all/COD-D/Test_depth/'
    test_datasets = ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']
    # test_datasets = ['CAMO']
    monitor = ExperimentMonitor(
        "torch_test",
        config={
            "framework": "torch",
            "trainsize": opt.trainsize,
            "model": "./checkpoints/Model_200_gen.pth",
            "test_root": dataset_path,
            "test_datasets": test_datasets,
            "note": "single-card full-dataset test",
        },
    )
    print("开始初始化模型，优化器...")
    generator = build_sam_DepthSAM(image_size=opt.trainsize)
    data = torch.load('./checkpoints/Model_200_gen.pth', map_location='cpu')
    if list(data.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in data.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        generator.load_state_dict(new_state_dict)
    else:
        generator.load_state_dict(data)

    if torch.cuda.is_available():
        try:
            # 尝试初始化 CUDA
            torch.cuda.init()
            device = torch.device('cuda:0')
            generator = generator.to(device)
            print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        except RuntimeError as e:
            print(f"CUDA 初始化失败: {e}")
            print("切换到 CPU 模式")
            device = torch.device('cpu')
            generator = generator.to(device)
    else:
        print("CUDA 不可用，使用 CPU")
        device = torch.device('cpu')
        generator = generator.to(device)

    # generator.cuda()
    generator.eval()
    for dataset in test_datasets:
        save_path = 'test_maps_rebuttal/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        image_root = dataset_path + dataset + '/Imgs/'
        gt_root = dataset_path + dataset + '/GT/'
        g_root = dataset_path + dataset + '/depth/'
        test_loader = test_dataset(image_root, gt_root,g_root, opt.trainsize)

        with torch.no_grad():
            mae_sum = 0
            test_count = 0
            for i in tqdm(range(test_loader.size)):
                image, gt,depth, name, img_for_post = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.to(device)

                imgs = image.permute(0, 2, 3, 1).cpu().numpy()
                batched_input = []
                for b_i in range(len(imgs)):
                    dict_input = dict()
                    input_image = (torch.as_tensor((imgs[b_i]).astype(dtype=np.uint8), device=generator.device)
                                   .permute(2, 0, 1).contiguous())
                    dict_input['image'] = input_image
                    dict_input['original_size'] = imgs[b_i].shape[:2]
                    batched_input.append(dict_input)

                res = generator(batched_input, image)

                res= F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                if np.isnan(res).any():
                    print(f"Warning: NaN in output for {name}, skipping")
                    monitor.log_eval_sample(dataset, name, skipped=True)
                    continue
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                out_img = (res * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(save_path + name, out_img)
                sample_mae = np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
                mae_sum += sample_mae
                test_count += 1
                monitor.log_eval_sample(dataset, name, sample_mae)
                monitor.save_prediction_panel(dataset, name, img_for_post, res, gt)
            mae = mae_sum / max(test_count, 1)
            print(dataset, 'mae is : ', mae)

    summary = monitor.finish({"mode": "test"})
    print("Experiment log saved to:", summary["run_dir"])

if __name__ == '__main__':
    test()
