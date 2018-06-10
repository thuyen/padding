import torch
import torch.nn as nn
from torch.autograd import Function
import padding._C as _ext


class PaddingFunction(Function):
    @staticmethod
    def forward(ctx, x, pad_h=1, pad_w=0, flag=False):
        ctx.constant = pad_h, pad_w, flag
        if not x.is_contiguous():
            x = x.contiguous()
        if x.is_cuda:
            out = _ext.padh_gpu_forward(x, pad_h, pad_w, flag)
        else:
            out = _ext.padh_cpu_forward(x, pad_h, pad_w, flag)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        pad_h, pad_w, flag = ctx.constant
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        if grad_output.is_cuda:
            out = _ext.padh_gpu_backward(grad_output, pad_h, pad_w, flag)
        else:
            out = _ext.padh_cpu_backward(grad_output, pad_h, pad_w, flag)
        return out, None, None, None


pad = PaddingFunction.apply
class Padding(nn.Module):
    def __init__(self, pad_h=1, pad_w=0, onesided=False):
        super(Padding, self).__init__()
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.onesided = onesided

    def forward(self, x):
        return pad(x, self.pad_h, self.pad_w, self.onesided)


class CropFunction(Function):
    @staticmethod
    def forward(ctx, x, r, pooled_h=1, pooled_w=1):
        height, width = x.size(2), x.size(3)
        ctx.constant = height, width
        ctx.save_for_backward(r)
        if not x.is_contiguous():
            x = x.contiguous()
        if x.is_cuda:
            out = _ext.crop_gpu_forward(x, r, pooled_w, pooled_h)
        else:
            out = _ext.crop_cpu_forward(x, r, pooled_w, pooled_h)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        height, width = ctx.constant
        r, = ctx.saved_tensors
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        if grad_output.is_cuda:
            out = _ext.crop_gpu_backward(grad_output, r, height, width)
        else:
            out = _ext.crop_cpu_backward(grad_output, r, height, width)
        return out, None, None, None


