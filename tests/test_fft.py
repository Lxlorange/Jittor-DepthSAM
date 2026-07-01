import jittor as jt
import jittor.nn as nn

jt.flags.use_cuda = 1

# real = jt.randn((1, 8, 8))
# z = nn.ComplexNumber(real)

# print("before fft")
# y = z.fft2()
# print("after graph build")

# print(y.real.shape)
# print(y.imag.shape)

# print(y.real.sync())


import numpy as np
jt.flags.use_cuda = 1
x = np.zeros((1, 3, 128, 128), dtype=np.float32)
y = jt.Var(x)
print(y.shape)