import argparse
import math
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ.setdefault("cpu_mem_limit", "-1")
os.environ.setdefault("device_mem_limit", "-1")
import cv2
import jittor as jt
import datetime
from segment_anything_training.build_DepthSAM import build_sam_DepthSAM
from utils.dataset_rgb_strategy2 import SalObjDataset
from utils.utils import AvgMeter
from utils.experiment_monitor import ExperimentMonitor
from utils.jittor_runtime import (
    configure_jittor_runtime,
    maybe_print_memory_profile,
    optional_memory_profile,
    print_runtime_hints,
    sync_gc,
)
import jittor.nn as nn
# import jittor.distributed as dist
import numpy as np
from data_cod import test_dataset
from tqdm import tqdm



def get_loader(image_root, gt_root, depth_root, batchsize, trainsize, distributed=False):
    dataset = SalObjDataset(image_root, gt_root, depth_root, trainsize)
    if distributed:
        # API uncertainty: Jittor distributed data sharding is version dependent.
        raise NotImplementedError("distributed Jittor data loading is not wired in this port")
    else:
        sampler = None
        shuffle = True

    dataset.set_attrs(batch_size=batchsize, shuffle=shuffle, num_workers=0, keep_numpy_array=True)
    return dataset, sampler


def structure_loss(pred, mask, eps=1e-8):
    weit = 1 + 5 * jt.abs(nn.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    max_val = jt.clamp(-pred, min_v=0)
    log_term = max_val + ((-max_val).exp() + (-pred - max_val).exp()).log()
    wbce = (1 - mask) * pred + log_term
    wbce = jt.sum(weit * wbce, dims=[2, 3]) / (jt.sum(weit, dims=[2, 3]) + eps)

    pred = jt.sigmoid(pred)
    inter = jt.sum((pred * mask) * weit, dims=[2, 3])
    union = jt.sum((pred + mask) * weit, dims=[2, 3])
    wiou = 1 - (inter + 1) / (union - inter + 1 + eps)

    return (wbce + wiou).mean()


def normalize_map(res):
    if not np.isfinite(res).all():
        return None
    res_min = res.min()
    res_max = res.max()
    denom = res_max - res_min
    if not np.isfinite(denom) or denom < 1e-8:
        return np.zeros_like(res, dtype=np.float32)
    return (res - res_min) / denom


def is_dist_avail_and_initialized():
    """Jittor 中通过 jt.mpi 是否为 None 判断是否处于分布式环境"""
    return jt.mpi is not None

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return jt.rank

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return jt.world_size

def barrier():
    """Jittor 没有 dist.barrier()，可用 sync_all 做粗略同步"""
    if jt.mpi is not None:
        jt.sync_all()

def init_distributed_mode(args):
    # 情况1：标准 MPI 环境（mpirun / mpiexec 启动）
    if jt.mpi is not None:
        args.rank = jt.rank
        args.world_size = jt.world_size
        
        # 计算 local_rank（当前节点内的 GPU 编号）
        if 'LOCAL_RANK' in os.environ:
            args.gpu = int(os.environ['LOCAL_RANK'])
        else:
            args.gpu = jt.rank % jt.get_device_count()
        
        jt.flags.use_cuda = 1
        jt.set_device(args.gpu)   # 或 jt.cuda.set_device(args.gpu)
        
        distributed = True
        print(f"Distributed training initialized: rank {args.rank}/{args.world_size}, gpu {args.gpu}")

    # 情况2：SLURM 集群环境
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ.get('SLURM_NTASKS', 1))
        args.gpu = args.rank % jt.get_device_count()
        
        jt.flags.use_cuda = 1
        jt.set_device(args.gpu)
        
        distributed = True
        print(f"SLURM distributed training: rank {args.rank}, gpu {args.gpu}")

    # 情况3：单卡 / 非分布式
    else:
        print("Not using distributed mode")
        distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = 0

    return distributed, args.gpu


def save_state_dict_npz(state_dict, path):
    np_state = {}
    for key, value in state_dict.items():
        if hasattr(value, "numpy"):
            np_state[key] = value.numpy()
        else:
            np_state[key] = np.asarray(value)
    np.savez(path, **np_state)
    del np_state
    sync_gc()


def load_state_dict_npz(path):
    data = np.load(path)
    return {key: jt.array(data[key]) for key in data.files}


def trainable_parameters(model):
    return [param for param in model.parameters() if getattr(param, "requires_grad", True)]




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--lr_gen', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=512, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=200, help='every n epochs decay learning rate')
    parser.add_argument('-beta1_gen', type=float, default=0.5, help='beta of Adam for generator')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
    parser.add_argument('--feat_channel', type=int, default=64, help='reduced channel of saliency feat')
    parser.add_argument('--gpu', type=int, default='0', help='reduced channel of saliency feat')
    parser.add_argument('--log_interval', type=int, default=50, help='steps between scalar loss logging')
    parser.add_argument('--sync_interval', type=int, default=200, help='steps between explicit Jittor sync/gc calls')
    return parser.parse_args()


def train():
    opt = get_args()
    configure_jittor_runtime()
    print_runtime_hints()
    profile_memory = os.environ.get("JT_PROFILE_MEMORY", "0") == "1"

    image_cod_root = "./Data_all/COD-D/Train_depth/Imgs/"
    gt_cod_root = "./Data_all/COD-D/Train_depth/GT/"
    depth_cod_root = "./Data_all/COD-D/Train_depth/depth/"

    train_loader, cod_sampler = get_loader(image_cod_root, gt_cod_root, depth_cod_root, batchsize=opt.batchsize,
                                           trainsize=opt.trainsize, distributed=False)

    save_path = './checkpoints/'
    monitor = ExperimentMonitor(
        "jittor_train",
        config={
            "framework": "jittor",
            "epoch": opt.epoch,
            "lr_gen": opt.lr_gen,
            "batchsize": opt.batchsize,
            "trainsize": opt.trainsize,
            "log_interval": opt.log_interval,
            "sync_interval": opt.sync_interval,
            "train_root": image_cod_root,
            "test_root": "./Data_all/COD-D/Test_depth/",
            "note": "single-card full-dataset run; recommended as about 2/3 of the original 300-epoch schedule",
        },
    )

    print("开始初始化模型，优化器...")
    generator = build_sam_DepthSAM(image_size=opt.trainsize)
    generator_optimizer = jt.optim.Adam(trainable_parameters(generator), opt.lr_gen)

    sample_count = getattr(train_loader, "total_len", len(train_loader))
    total_step = math.ceil(sample_count / opt.batchsize)
    last_w_path = None
    print("Start Training...")
    for epoch in range(1, opt.epoch + 1):
        generator.train()
        loss_record = AvgMeter()
        current_lr = getattr(generator_optimizer, 'lr', opt.lr_gen)
        print('Epoch [{:03d}/{:03d}] Learning Rate: {}'.format(epoch, opt.epoch, current_lr))
        last_loss_value = None
        last_loss1_value = None

        train_loader_iter = tqdm(
            train_loader,
            total=total_step,
            desc='Train Epoch {:03d}/{:03d}'.format(epoch, opt.epoch),
            ncols=120,
        )

        for i, (images, gts, depth) in enumerate(train_loader_iter, start=1):
            # 从 numpy 统一转成 jittor Var
            images = jt.array(images)
            gts = jt.array(gts)
            depth = jt.array(depth)
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            if len(gts.shape) == 3:
                gts = gts.unsqueeze(0)
            if len(depth.shape) == 3:
                depth = depth.unsqueeze(0)
            
            B = images.shape[0]
            batched_input = []
            for b_i in range(B):
                dict_input = dict()
                input_image = images[b_i]  # 直接取，已经是 (C, H, W)
                dict_input['image'] = input_image
                dict_input['original_size'] = (input_image.shape[1], input_image.shape[2])
                batched_input.append(dict_input)

            do_memory_profile = profile_memory and epoch == 1 and i == 1
            with optional_memory_profile(do_memory_profile):
                s1 = generator(batched_input, images)
                loss1 = structure_loss(s1, gts)
                loss = loss1

                generator_optimizer.step(loss)
                maybe_print_memory_profile(do_memory_profile)

            should_log = i == 1 or i % opt.log_interval == 0 or i == total_step
            if should_log:
                loss_value = float(loss.item())
                loss1_value = float(loss1.item())
                last_loss_value = loss_value
                last_loss1_value = loss1_value
                loss_record.update(loss_value, opt.batchsize)
                monitor.log_train_step(
                    epoch,
                    i,
                    total_step,
                    loss_value,
                    current_lr,
                )
                train_loader_iter.set_postfix(
                    loss='{:.4f}'.format(loss_value),
                    avg='{:.4f}'.format(float(loss_record.show())),
                    lr=current_lr,
                )
            if should_log:
                tqdm.write('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Pre Loss: {:.4f}, Pre1 Loss: {:.4f}'.
                           format(datetime.datetime.now(), epoch, opt.epoch, i, total_step,
                                  float(loss_record.show()), last_loss1_value))
            if i % opt.sync_interval == 0 or i == total_step:
                sync_gc()
            del images, gts, depth, batched_input, s1, loss1, loss

        print('{} Epoch [{:03d}/{:03d}] Finished, Avg Loss: {:.4f}'.
              format(datetime.datetime.now(), epoch, opt.epoch, float(loss_record.show())))
        sync_gc()

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if epoch >= 10 or epoch % opt.epoch == 0:
            w_path = save_path + 'Model_' + str(epoch) + '_gen.npz'
            save_state_dict_npz(generator.state_dict(), w_path)
            sync_gc()
            test_cod(w_path, generator, monitor)
            sync_gc()
            if last_w_path is not None and last_w_path != w_path and os.path.exists(last_w_path):
                os.remove(last_w_path)
            last_w_path = w_path

    summary = monitor.finish({"checkpoint_dir": save_path})
    print("Experiment log saved to:", summary["run_dir"])

best_mae = 10000
best_epoch = 0

def test_cod(w_path, generator=None, monitor=None):
    opt = get_args()
    configure_jittor_runtime()
    global best_mae, best_epoch

    test_path = './Data_all/COD-D/Test_depth/'
    test_datasets = ['CAMO']


    save_path = './checkpoints/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if generator is None:
        generator = build_sam_DepthSAM(image_size=opt.trainsize)
        data = load_state_dict_npz(w_path)
        if list(data.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in data.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            generator.load_state_dict(new_state_dict)
        else:
            generator.load_state_dict(data)

    # generator.eval() #不知道为啥会让res变Nan
    generator.train()
    sync_gc()
    for dataset in test_datasets:
        save_path = './test_maps/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = test_path + dataset + '/Imgs/'
        gt_root = test_path + dataset + '/GT/'
        d_root = test_path + dataset + '/depth/'
        test_loader = test_dataset(image_root, gt_root, d_root, opt.trainsize)

        mae_sum = 0
        test_count = 0

        for i in range(test_loader.size):  # 250
            image, gt, depth, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = jt.array(image)
            depth = jt.array(depth)
            batched_input = []
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            if len(depth.shape) == 3:
                depth = depth.unsqueeze(0)
            for b_i in tqdm(range(image.shape[0])):
                input_image = image[b_i]
                batched_input.append(
                    {
                        'image': input_image,
                        'original_size': (input_image.shape[1], input_image.shape[2]),
                    }
                )

            res = generator(batched_input, image)
            res = nn.upsample(res, size=gt.shape[-2:], mode='bilinear', align_corners=False)
            res = res.sigmoid().numpy().squeeze()
            res = normalize_map(res)
            if res is None:
                print(f"Warning: non-finite output for {name}, skipping")
                if monitor is not None:
                    monitor.log_eval_sample(dataset, name, skipped=True)
                del image, gt, depth, batched_input, res
                sync_gc()
                continue
            out_img = (res * 255).clip(0, 255).astype(np.uint8)
            if not cv2.imwrite(save_path + name, out_img):
                print(f"Warning: failed to write prediction for {name}")
            sample_mae = np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[-2] * gt.shape[-1])
            mae_sum += sample_mae
            test_count += 1
            if monitor is not None:
                monitor.log_eval_sample(dataset, name, sample_mae)
                monitor.save_prediction_panel(dataset, name, image_for_post, res, gt)
            del image, gt, depth, batched_input, res, out_img
            sync_gc()

        mae = mae_sum / max(test_count, 1)

        print(dataset, 'Res mae is : ', mae)

if __name__ == '__main__':
    train()
