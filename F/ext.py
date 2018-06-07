import torch
from torch.autograd import Function
import padding

class PaddingFunction(Function):
    @staticmethod
    def forward(ctx, x, pad_h=1, pad_w=0, flag=False):
        ctx.constant = pad_h, pad_w, flag
        if not x.is_contiguous():
            x = x.contiguous()
        if x.is_cuda:
            out = padding.padh_gpu_forward(x, pad_h, pad_w, flag)
        else:
            out = padding.padh_cpu_forward(x, pad_h, pad_w, flag)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        pad_h, pad_w, flag = ctx.constant
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        if grad_output.is_cuda:
            out = padding.padh_gpu_backward(grad_output, pad_h, pad_w, flag)
        else:
            out = padding.padh_cpu_backward(grad_output, pad_h, pad_w, flag)
        return out, None, None

pad = PaddingFunction.apply
