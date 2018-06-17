import torch
import torch.nn as nn
from torch.autograd import Function
import padding._C as _C


class PaddingFunction(Function):
    @staticmethod
    def forward(ctx, x, pad_h=1, pad_w=0, flag=False):
        ctx.constant = pad_h, pad_w, flag
        if not x.is_contiguous():
            x = x.contiguous()
        if x.is_cuda:
            out = _C.padh_gpu_forward(x, pad_h, pad_w, flag)
        else:
            out = _C.padh_cpu_forward(x, pad_h, pad_w, flag)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        pad_h, pad_w, flag = ctx.constant
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        if grad_output.is_cuda:
            out = _C.padh_gpu_backward(grad_output, pad_h, pad_w, flag)
        else:
            out = _C.padh_cpu_backward(grad_output, pad_h, pad_w, flag)
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
    def forward(ctx, x, r, pooled_h=1, pooled_w=1, first=True):
        off = 1 if first else 0
        height, width = x.size(2-off), x.size(3-off)
        ctx.constant = height, width, first
        ctx.save_for_backward(r)
        if not x.is_contiguous():
            x = x.contiguous()
        if x.is_cuda:
            out = _C.crop_gpu_forward(x, r, pooled_h, pooled_w, first)
        else:
            out = _C.crop_cpu_forward(x, r, pooled_h, pooled_w, first)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        height, width, first = ctx.constant
        r, = ctx.saved_tensors
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        if grad_output.is_cuda:
            out = _C.crop_gpu_backward(grad_output, r, height, width, first)
        else:
            out = _C.crop_cpu_backward(grad_output, r, height, width, first)
        return out, None, None, None, None


crop = CropFunction.apply


class Conv2DFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, groups, padh, padw, onesided):
        ctx.constant = padh, padw, onesided, stride, groups
        ctx.save_for_backward(x)
        ctx.save_for_backward(weight)

        if not x.is_contiguous():
            x = x.contiguous()
        if x.is_cuda:
            out = _C.conv2d_gpu_forward(
                    x, weight, bias,
                    padh, padw, onesided,
                    stride, groups)
        else:
            raise ValueError('CPU OP is not supported')

        return out

    @staticmethod
    def backward(ctx, grad_output):
        padh, padw, onesided, stride, groups = ctx.constant
        x, weight = ctx.saved_tensors
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        if grad_output.is_cuda:
            #grad_input, grad_weight
            ret = _C.conv2d_gpu_backward(
                    grad_output, x, weight,
                    padh, padw, onesided,
                    stride, groups)
        else:
            raise ValueError('CPU OP is not supported')

        ret = ret + (None, )*5
        return ret


circular_conv2d = Conv2DFunction.apply


class Svf2DFunction(Function):
    @staticmethod
    def forward(ctx, x, r, weight, pooled_height, pooled_width, first):

        off = 1 if first else 0
        height, width = x.size(2-off), x.size(3-off)

        ctx.constnat = height, width, pooled_height, pooled_widht, first
        ctx.save_for_backward(x)
        ctx.save_for_backward(r)
        ctx.save_for_backward(weight)

        if not x.is_contiguous():
            x = x.contiguous()
        if x.is_cuda:
            out = _C.svf2d_gpu_forward(
                    x, r, weight, pooled_height, pooled_width, first)
        else:
            raise ValueError('CPU OP is not supported')

        return out

    @staticmethod
    def backward(ctx, grad_output):
        height, width, pooled_height, pooled_widht, first = ctx.constant
        x, r, weight = ctx.saved_tensors
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        if grad_output.is_cuda:
            #grad_input, grad_weight
            ret = _C.svf2d_gpu_backward(
                    grad_output,
                    x, r, weight,
                    height, width, pooled_height, pooled_width, first)

        else:
            raise ValueError('CPU OP is not supported')

        ret = ret + (None, )*5
        return ret


circular_svf2d = Svf2DFunction.apply
