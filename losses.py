
import jittor as jt
import jittor.nn as nn

def structure_loss(pred, mask):
    weit = 1 + 5 * jt.abs(nn.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    # wbce = nn.binary_cross_entropy_with_logits(pred,mask) # 返回标量，下面空间求和就没意义了
    # 这里手动实现一下交叉熵
    max_val = jt.clamp(-pred, min_v=0)
    log_term = max_val + ((-max_val).exp() + (-pred - max_val).exp()).log()

    wbce = (1 - mask) * pred + log_term
    wbce = jt.sum(weit * wbce, dims=[2, 3]) / jt.sum(weit, dims=[2, 3])

    pred = jt.sigmoid(pred)
    inter = jt.sum(((pred * mask) * weit),dims=[2,3])
    union = jt.sum(((pred + mask) * weit),dims=[2,3])
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


