import jittor as jt
import jittor.nn as nn

jt.flags.use_cuda = 1

real = jt.randn((1, 8, 8))
z = nn.ComplexNumber(real)

print("before fft")
y = z.fft2()
print("after graph build")

print(y.real.shape)
print(y.imag.shape)

print(y.real.sync())