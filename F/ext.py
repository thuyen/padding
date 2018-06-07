import torch
from torch.autograd import Function
import padding

class Padding(Function):
    def __init__(self, pad):
        self.pad = pad

    def forward(self, x):
        if x.is_cuda():
            return padding.padh_cpu_forward(x, self.pad)
        else:
            return padding.padh_gpu_forward(x, self.pad)

    def backward(self, grad_output):
        if grad_output.is_cuda():
            return padding.padh_gpu_backward(x, self.pad)
        else:
            return padding.padh_gpu_backward(x, self.pad)

def pad(x, p):
    op = Padding(p)
    return op(x)


