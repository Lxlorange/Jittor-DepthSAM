import jittor as jt
import jittor.nn as nn
from segment_anything_training.modeling.common import LayerNorm2d
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm(out_planes),
        nn.ReLU(),
    )

def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)


def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv1x1(in_planes, out_planes, stride),
        nn.BatchNorm(out_planes),
        nn.ReLU(),
    )


def custom_complex_normalization(input_var_real:jt.Var, input_var_imag:jt.Var, dim=-1)-> nn.ComplexNumber:
    # 传参改成虚部实部分开了，后续调用这个函数要注意
    norm_real = nn.softmax(input_var_real, dim=dim)
    norm_imag = nn.softmax(input_var_imag, dim=dim)
    normalized_var = nn.ComplexNumber(norm_real, norm_imag)
    return normalized_var


# 从这里开始，fft2是nn.ComplexNumber的函数，所以Var类变量不能直接调用，因此重写一些包装函数。同理，也不使用rearrange了，这样比较好理解
def _complex(x):
    if isinstance(x, nn.ComplexNumber):
        return x
    return nn.ComplexNumber(x, jt.zeros_like(x))


def _complex_abs(x):
    return _complex(x).norm()


def _complex_real(x):
    return _complex(x).real


def _complex_mul_real(x, y):
    x = _complex(x)
    return nn.ComplexNumber(x.real * y, x.imag * y)


def _fft2(x):
    if jt.flags.use_cuda != 1:
        raise RuntimeError("Attention_SD uses Jittor ComplexNumber.fft2(), which requires CUDA. Set jt.flags.use_cuda = 1 before running this module.")
    x = _complex(x)
    shape = x.real.shape
    flat = nn.ComplexNumber(
        x.real.reshape((-1, shape[-2], shape[-1])),
        x.imag.reshape((-1, shape[-2], shape[-1])),
    )
    out = flat.fft2()
    return nn.ComplexNumber(out.real.reshape(shape), out.imag.reshape(shape))


def _ifft2(x):
    if jt.flags.use_cuda != 1:
        raise RuntimeError("Attention_SD uses Jittor ComplexNumber.ifft2(), which requires CUDA. Set jt.flags.use_cuda = 1 before running this module.")
    x = _complex(x)
    shape = x.real.shape
    flat = nn.ComplexNumber(
        x.real.reshape((-1, shape[-2], shape[-1])),
        x.imag.reshape((-1, shape[-2], shape[-1])),
    )
    out = flat.ifft2()
    return nn.ComplexNumber(out.real.reshape(shape), out.imag.reshape(shape))

# b (head c) h w -> b head c (h w)
def _split_heads(x, head):
    b, c, h, w = x.shape
    assert c % head == 0
    return x.reshape((b, head, c // head, h * w))

# b head c (h w) -> b (head c) h w
def _merge_heads(x, head, h, w):
    b = x.shape[0]
    c_per_head = x.shape[2]
    return x.reshape((b, head * c_per_head, h, w))

def _normalize(x, dim=-1, eps=1e-12):
    if isinstance(x, nn.ComplexNumber):
        denom = (x.norm() * x.norm()).sum(dim, keepdims=True).sqrt() + eps
        return nn.ComplexNumber(x.real / denom, x.imag / denom)

    denom = (x * x).sum(dim, keepdims=True).sqrt() + eps
    return x / denom


def custom_complex_normalization(input_tensor, dim=-1):
    input_tensor = _complex(input_tensor)
    norm_real = nn.softmax(input_tensor.real, dim=dim)
    norm_imag = nn.softmax(input_tensor.imag, dim=dim)
    return nn.ComplexNumber(norm_real, norm_imag)

# 到这里为止

# 对应论文 SFRM
class Attention_SD(nn.Module):
    def __init__(self, dim, num_heads=2):
        super(Attention_SD, self).__init__()
        self.num_heads = num_heads

        # rgb分支频域
        self.qr = conv1x1(dim,dim)  
        self.kr = conv1x1(dim,dim)
        self.vr = conv1x1(dim,dim)

        # depth分支空间
        self.qm = conv1x1(dim,dim)
        self.km = conv1x1(dim,dim)
        self.vm = conv1x1(dim,dim)

        self.temperature = jt.ones((num_heads, 1, 1))
        self.temperatured = jt.ones((num_heads, 1, 1))

        self.project_out = conv3x3_bn_relu(dim * 4, dim)

        self.weight = nn.Sequential(
            nn.Conv2d(dim, dim // 16, 1, bias=True),
            nn.BatchNorm(dim // 16),
            nn.ReLU(),
            nn.Conv2d(dim // 16, dim, 1, bias=True),
            nn.Sigmoid())
        self.project_outr = nn.Conv2d(dim * 2, dim, kernel_size=1)

        self.weightd = nn.Sequential(
            nn.Conv2d(dim, dim // 16, 1, bias=True),
            nn.BatchNorm(dim // 16),
            nn.ReLU(),
            nn.Conv2d(dim // 16, dim, 1, bias=True),
            nn.Sigmoid())
        self.project_outd = nn.Conv2d(dim * 2, dim, kernel_size=1)


    def execute(self, x, d):
        b, c, h, w = x.shape
        q_s = self.qr(x)
        k_s = self.kr(x)
        v_s = self.vr(x)
        q_s = _fft2(q_s.float32())
        k_s = _fft2(k_s.float32())
        v_s = _fft2(v_s.float32())
        q_s = _split_heads(q_s, self.num_heads)
        k_s = _split_heads(k_s, self.num_heads)
        v_s = _split_heads(v_s, self.num_heads)
        q_s = _normalize(q_s, dim=-1)
        k_s = _normalize(k_s, dim=-1)
        attn_s = _complex_mul_real(q_s @ k_s.transpose((0, 1, 3, 2)), self.temperature)
        attn_s = custom_complex_normalization(attn_s, dim=-1)
        outr0 = _complex_abs(_ifft2(attn_s @ v_s))
        attn_s = _complex_abs(_ifft2(attn_s))


        dq_s = self.qm(d)
        dk_s = self.km(d)
        dv_s = self.vm(d)
        dq_s = _split_heads(dq_s, self.num_heads)
        dk_s = _split_heads(dk_s, self.num_heads)
        dv_s = _split_heads(dv_s, self.num_heads)
        dq_s = _normalize(dq_s, dim=-1)
        dk_s = _normalize(dk_s, dim=-1)
        dattn_s = (dq_s @ dk_s.transpose((0, 1, 3, 2))) * self.temperatured
        dattn_s = nn.softmax(dattn_s, dim=-1)
        outd0 = dattn_s @ dv_s
        dattn_s = _fft2(dattn_s.float32())

        outr = _complex_abs(_ifft2(dattn_s @ v_s))
        outd = attn_s @ dv_s

        outd = _merge_heads(outd, self.num_heads, h, w)
        outr = _merge_heads(outr, self.num_heads, h, w)
        outd0 = _merge_heads(outd0, self.num_heads, h, w)
        outr0 = _merge_heads(outr0, self.num_heads, h, w)


        x_fft = _fft2(x.float32())
        out_f_lr = _complex_abs(_ifft2(_complex_mul_real(x_fft, self.weight(_complex_real(x_fft)))))
        outr = self.project_outr(jt.concat((outr, out_f_lr), dim=1))

        d_fft = _fft2(d.float32())
        out_f_ld = _complex_abs(_ifft2(_complex_mul_real(d_fft, self.weightd(_complex_real(d_fft)))))
        outd = self.project_outd(jt.concat((outd, out_f_ld), dim=1))

        out = self.project_out(jt.concat((outr, outr0, outd, outd0), dim=1))


        return out

class FM(nn.Module):
    def __init__(self, dim,oup):
        super(FM, self).__init__()
        self.dim = oup
        self.conv_n = BasicConv2d(dim, self.dim, kernel_size=1, stride=1)
        self.asd = Attention_SD(self.dim)
        self.ddd = DWConv(self.dim)

    def execute(self,x):
        x = self.conv_n(x)
        out = self.asd(x,x)
        out = self.ddd(out)
        return out

class MEF(nn.Module):
    def __init__(self, in1,in2):
        super(MEF, self).__init__()
        self.fm = FM(in1 + in1,in1)
        self.fm1 = nn.Sequential(
            nn.ConvTranspose2d(in2, in1, kernel_size=2, stride=2),
            LayerNorm2d(in1),
            nn.GELU()
        )
    def execute(self, in1, in2=None):
        in2 = self.fm1(in2)
        x = jt.concat((in1, in2), dim=1)
        out = self.fm(x)
        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm(out_planes)
        self.relu = nn.ReLU()

    def execute(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Decode(nn.Module):
    def __init__(self, in1,in2,in3,in4):
        super(Decode, self).__init__()
        # self.in3 = in3
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.premp = nn.Sequential(
            nn.ConvTranspose2d(in1, in1//2, kernel_size=2, stride=2),
            LayerNorm2d(in1//2),
            nn.GELU(),
            nn.Conv2d(in_channels=in1//2, out_channels=1, kernel_size=3, padding=1, bias=True),
        )


        self.fm1 = MEF(in1,in2)
        self.fm2 = MEF(in2,in3)
        self.fm3 = MEF(in3,in4)
    def execute(self,x1,x2,x3,x4):

        br3 = self.fm3(x3, x4)
        br2 = self.fm2(x2, br3)
        br1 = self.fm1(x1, br2)
        out = self.premp(br1)
        return out, br1


class DWConv(nn.Module):
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dim = dim//4
        self.dw2 = nn.Conv2d(self.dim, self.dim, kernel_size=7, padding=3, groups=self.dim)
        self.dw1 = nn.Conv2d(dim, self.dim, kernel_size=5, padding=2, groups=self.dim)
        self.dw3 = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1, groups=self.dim)
        self.dw4 = nn.Conv2d(self.dim, self.dim, kernel_size=1)

        self.conv_end = conv1x1_bn_relu(dim,dim)
    def execute(self, x: jt.Var) -> jt.Var:
        x1 = self.dw1(x)
        x2 = self.dw2(x1) + x1
        x3 = self.dw3(x2) + x2
        x4 = self.dw4(x3) + x3

        x = self.conv_end(jt.concat((x1, x2, x3, x4), dim=1))
        return x

# if __name__ == "__main__":
#     jt.flags.use_cuda = 1

#     def check(name, value, expected_shape=None):
#         if isinstance(value, tuple):
#             shapes = tuple(v.shape for v in value)
#             print(f"{name}: {shapes}")
#             if expected_shape is not None:
#                 assert shapes == expected_shape, f"{name}: expected {expected_shape}, got {shapes}"
#         else:
#             print(f"{name}: {value.shape}")
#             if expected_shape is not None:
#                 assert value.shape == expected_shape, f"{name}: expected {expected_shape}, got {value.shape}"

#     x = jt.randn((1, 32, 16, 16))
#     d = jt.randn((1, 32, 16, 16))

#     z = _complex(x)
#     check("_complex_abs", _complex_abs(z), (1, 32, 16, 16))
#     check("_fft2/_ifft2", _complex_abs(_ifft2(_fft2(x))), (1, 32, 16, 16))
#     check("_split_heads", _split_heads(x, 2), (1, 2, 16, 256))
#     check("_merge_heads", _merge_heads(_split_heads(x, 2), 2, 16, 16), (1, 32, 16, 16))

#     check("conv1x1", conv1x1(32, 16)(x), (1, 16, 16, 16))
#     check("conv1x1_bn_relu", conv1x1_bn_relu(32, 16)(x), (1, 16, 16, 16))
#     check("conv3x3", conv3x3(32, 16)(x), (1, 16, 16, 16))
#     check("conv3x3_bn_relu", conv3x3_bn_relu(32, 16)(x), (1, 16, 16, 16))

#     check("BasicConv2d", BasicConv2d(32, 16, kernel_size=1)(x), (1, 16, 16, 16))
#     check("Attention_SD", Attention_SD(dim=32, num_heads=2)(x, d), (1, 32, 16, 16))
#     check("DWConv", DWConv(dim=32)(x), (1, 32, 16, 16))
#     check("FM", FM(dim=64, oup=32)(jt.randn((1, 64, 16, 16))), (1, 32, 16, 16))
#     check("MEF", MEF(in1=32, in2=64)(x, jt.randn((1, 64, 8, 8))), (1, 32, 16, 16))

#     decode = Decode(32, 32, 32, 32)
#     decode_out = decode(
#         jt.randn((1, 32, 16, 16)),
#         jt.randn((1, 32, 8, 8)),
#         jt.randn((1, 32, 4, 4)),
#         jt.randn((1, 32, 2, 2)),
#     )
#     check("Decode", decode_out, ((1, 1, 32, 32), (1, 32, 16, 16)))

#     print("MyNet smoke tests passed.")
