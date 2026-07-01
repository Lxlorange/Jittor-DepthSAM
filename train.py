import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import jittor as jt
import datetime
from segment_anything_training.build_DepthSAM import build_sam_DepthSAM
from utils.dataset_rgb_strategy2 import SalObjDataset
from utils.utils import AvgMeter
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


def structure_loss(pred, mask):
    weit = 1 + 5 * jt.abs(nn.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    max_val = jt.clamp(-pred, min_v=0)
    log_term = max_val + ((-max_val).exp() + (-pred - max_val).exp()).log()
    wbce = (1 - mask) * pred + log_term
    wbce = jt.sum(weit * wbce, dims=[2, 3]) / jt.sum(weit, dims=[2, 3])

    pred = jt.sigmoid(pred)
    inter = jt.sum((pred * mask) * weit, dims=[2, 3])
    union = jt.sum((pred + mask) * weit, dims=[2, 3])
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


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


class MOEAdapter(nn.Module):
    def __init__(self, blk, num_experts=8, top_k=4) -> None:
        super(MOEAdapter, self).__init__()
        self.block = blk
        self.num_experts = num_experts
        self.top_k = top_k

        self.register_buffer("current_aux_loss", jt.tensor(0.0))

        dim = blk.attn.qkv.in_features
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 32),
                nn.GELU(),
                nn.Linear(32, dim),
                nn.GELU(),
            ) for _ in range(num_experts)
        ])

    def execute(self, x):
        # x shape: (B, H, W, C)
        B, N, C = x.shape
        x_flat = x.reshape(B * N, C)

        gate_logits = self.gate(x_flat)
        gate_weights = nn.softmax(gate_logits, dim=-1)

        if self.training:
            avg_prob_per_expert = gate_weights.mean(dim=0)
            _, top_k_indices = jt.topk(gate_weights, self.top_k, dim=-1)
            expert_mask = nn.one_hot(top_k_indices, self.num_experts).sum(dim=1)
            tokens_per_expert = expert_mask.float().sum(dim=0)
            frac_tokens_per_expert = tokens_per_expert / (B * N)
            load_balance_loss = (avg_prob_per_expert * frac_tokens_per_expert).sum()
            self.current_aux_loss = load_balance_loss * self.num_experts
        else:
            self.current_aux_loss = jt.zeros((), dtype=x.dtype)

        top_k_weights, top_k_indices = jt.topk(gate_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)  # 归一化

        output = jt.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_weight = top_k_weights[:, i].unsqueeze(-1)

            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)
                if bool(mask.any().item()):
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e_idx](expert_input)
                    output[mask] += expert_weight[mask] * expert_output

        output = output.reshape(B, N, C)
        prompted = x + output
        net = self.block(prompted)

        return net




Contrastive_Loss = None


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", jt.array(temperature))
        self.register_buffer("negatives_mask", (1 - jt.init.eye(batch_size * 2)).float())

    def execute(self, emb_i, emb_j):
        z_i = emb_i / (emb_i.sqr().sum(dim=1, keepdims=True).sqrt() + 1e-12)
        z_j = emb_j / (emb_j.sqr().sum(dim=1, keepdims=True).sqrt() + 1e-12)
        representations = jt.concat([z_i, z_j], dim=0)
        similarity_matrix = jt.matmul(representations, representations.transpose(0, 1))
        # API uncertainty: jt.diag offset support differs by version, so use explicit indexing.
        idx = jt.arange(self.batch_size)
        positives = jt.concat(
            [
                similarity_matrix[idx, idx + self.batch_size],
                similarity_matrix[idx + self.batch_size, idx],
            ],
            dim=0,
        )
        nominator = jt.exp(positives / self.temperature)
        denominator = self.negatives_mask * jt.exp(similarity_matrix / self.temperature)
        loss_partial = -jt.log(nominator / jt.sum(denominator, dim=1))
        return jt.sum(loss_partial) / (2 * self.batch_size)


Contrastive_Loss = ContrastiveLoss(batch_size=1)


class MOELossCollector:
    def __init__(self, alpha=0.01):

        self.alpha = alpha

    def get_total_loss(self, model: nn.Module) -> jt.Var:

        total_aux_loss = jt.zeros(())

        for module in model.modules():
            if isinstance(module, MOEAdapter):
                total_aux_loss += module.current_aux_loss

        total_loss = self.alpha * total_aux_loss

        return total_loss


moe_loss_collector = MOELossCollector(alpha=0.01)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300, help='epoch number')
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
    return parser.parse_args()


def train():
    opt = get_args()
    jt.flags.use_cuda = 1

    image_cod_root = "./Data_all/COD-D/Train_depth/Imgs/"
    gt_cod_root = "./Data_all/COD-D/Train_depth/GT/"
    depth_cod_root = "./Data_all/COD-D/Train_depth/depth/"

    train_loader, cod_sampler = get_loader(image_cod_root, gt_cod_root, depth_cod_root, batchsize=opt.batchsize,
                                           trainsize=opt.trainsize, distributed=False)

    save_path = './checkpoints/'

    print("开始初始化模型，优化器...")
    generator = build_sam_DepthSAM(image_size=opt.trainsize)
    generator_optimizer = jt.optim.Adam(generator.parameters(), opt.lr_gen)

    total_step = len(train_loader)
    print("Start Training...")
    for epoch in range(1, opt.epoch + 1):
        generator.train()
        loss_record = AvgMeter()
        print('Learning Rate: {}'.format(getattr(generator_optimizer, 'lr', opt.lr_gen)))

        train_loader_iter = iter(train_loader)

        for i, (images, gts, depth) in enumerate(train_loader_iter):
            if i >= 10:
                break
            print(f"Step {i}: images type={type(images)}, shape={images.shape if hasattr(images, 'shape') else 'N/A'}")
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

            s1 = generator(batched_input, images)
            total_loss = moe_loss_collector.get_total_loss(generator)
            loss1 = structure_loss(s1, gts)
            loss = loss1 + total_loss

            generator_optimizer.step(loss)

            loss_record.update(loss, opt.batchsize)
            if i % 200 == 0 or i == total_step:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Pre Loss: {:.4f}, Pre1 Loss: {:.4f}'.
                      format(datetime.datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show(), float(loss1.item())))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if epoch >= 10 or epoch % opt.epoch == 0:
            jt.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '_gen.pth')
            w_path = save_path + 'Model_' + str(epoch) + '_gen.pth'
            test_cod(w_path)

best_mae = 10000
best_epoch = 0

def test_cod(w_path):
    opt = get_args()
    global best_mae, best_epoch

    test_path = './Data_all/COD-D/Test_depth/'
    test_datasets = ['CAMO']


    save_path = './checkpoints/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    generator = build_sam_DepthSAM(image_size=opt.trainsize)

    data = jt.load(w_path)
    if list(data.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in data.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        generator.load_state_dict(new_state_dict)
    else:
        generator.load_state_dict(data)

    generator.eval()
    for dataset in test_datasets:
        save_path = './test_maps/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = test_path + dataset + '/Imgs/'
        gt_root = test_path + dataset + '/GT/'
        d_root = test_path + dataset + '/depth/'
        test_loader = test_dataset(image_root, gt_root, d_root, opt.trainsize)

        mae_sum = 0

        for i in range(test_loader.size):  # 250
            image, gt, depth, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            imgs = image.permute(0, 2, 3, 1).numpy()
            batched_input = []
            for b_i in tqdm(range(len(imgs))):
                dict_input = dict()
                input_image = jt.array((imgs[b_i]).astype(dtype=np.uint8)).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            res = generator(batched_input, image)
            res = nn.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size

        print(dataset, 'Res mae is : ', mae)

if __name__ == '__main__':
    train()
