   
from segment_anything_training.modeling.common import *

if __name__ == '__main__':
    
    # x = jt.randn((1,1,256,768))
    # dim = 768
    # mlp_ratio = 4.0

    # mlp_block = MLPBlock(embedding_dim=dim, mlp_dim=int(dim*mlp_ratio))
    # output = mlp_block(x)
    # print(output.shape)
    
    ln = LayerNorm2d(num_channels=256)
    optimizer = nn.SGD(ln.parameters(), lr=0.01)
    print([p.name() for p in ln.parameters()])
    # 预期输出包含 'weight' 和 'bias'
    x = jt.randn((2, 256, 14, 14))
    y = ln(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)   # 应为 (2, 256, 14, 14)
    
    # 每个 (B,H,W) 位置上的 C 个通道应均值为 0，方差为 1
    # 注意：因为 weight=1, bias=0，所以输出就是标准归一化结果
    mean_per_pos = y.mean(1, keepdims=True)
    var_per_pos  = y.pow(2).mean(1, keepdims=True)  # 均值已为0，直接平方平均
    
    print("Mean (should ~0):", mean_per_pos.mean().item())
    print("Var  (should ~1):", var_per_pos.mean().item())
    
    # 验证梯度是否正常回传
    old_weight = ln.weight.copy()
    old_bias = ln.bias.copy()

    loss = y.sum()
    # 一步更新（内部包含 backward + step）
    optimizer.step(loss)

    # 检查参数是否被更新（如果梯度正常回传，参数会变化）
    print("weight updated:", not jt.all_equal(ln.weight, old_weight))
    print("bias   updated:", not jt.all_equal(ln.bias, old_bias))