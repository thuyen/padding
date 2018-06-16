#include <torch/torch.h>

at::Tensor padh_gpu_forward(
    const at::Tensor &X, // 3d image hwc
    const int pad, const int padw, const bool b
    );

at::Tensor padh_gpu_backward(
    const at::Tensor &X, // 3d image hwc
    const int pad, const int padw, const bool b
    );

at::Tensor crop_gpu_forward(
    const at::Tensor &X, // nchw
    const at::Tensor &R, // k2
    const int pooled_height, const int pooled_width, const bool first
    );

at::Tensor crop_gpu_backward(
    const at::Tensor &X, // nkchw
    const at::Tensor &R, // k2
    const int height, const int width, const bool first
    );


at::Tensor conv2d_gpu_forward(
    const at::Tensor &X, // nchw
    const int padh, const int padw, const bool onesided,
    const at::Tensor weight, const at::Tensor bias,
    at::IntList stride, int64_t groups
    ) {

  auto& ctx = at::globalContext();

  at::Tensor padded = padh_gpu_forward(X, padh, padw, onesided);
  return at::cudnn_convolution(
      padded, weight, bias,
      {0, 0}, stride, {1, 1}, groups,
      ctx.benchmarkCuDNN(), ctx.deterministicCuDNN());

  //cudnn_convolution(self, weight, bias,
  //    padding, stride, dilation, groups, benchmark, deterministic)
}


std::tuple<at::Tensor, at::Tensor, at::Tensor> conv2d_gpu_backward(
    const at::Tensor &input, // nchw
    const at::Tensor &grad_output, // nchw
    const int padh, const int padw, const bool onesided,
    const at::Tensor weight,
    at::IntList stride, int64_t groups
    ) {
  //grad_input, grad_weight, grad_bias =
  auto& ctx = at::globalContext();
  auto ret =
    at::cudnn_convolution_backward(input, grad_output, weight,
        {0, 0}, stride, {1, 1}, groups,
        ctx.benchmarkCuDNN(),
        ctx.deterministicCuDNN(),
        {true, true, true});
  at::Tensor grad_input  = std::get<0>(ret);
  at::Tensor grad_weight = std::get<1>(ret);
  at::Tensor grad_bias   = std::get<2>(ret);
  grad_input = padh_gpu_forward(grad_input, padh, padw, onesided);
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}


at::Tensor svf2d_gpu_forward(
    const at::Tensor &X, // nchw
    const at::Tensor &R, // k2
    const int pooled_height, const int pooled_width, const bool first,
    const at::Tensor &weight
    ) {
  auto cropped = crop_gpu_forward(X, R, pooled_height, pooled_width, first);
  cropped = cropped.view({X.size(0) * R.size(0), 1, -1});
  return at::bmm(cropped, weight);
}


std::tuple<at::Tensor, at::Tensor> svf2d_gpu_backward(
    const at::Tensor &grad_output, // nchw
    const at::Tensor &X, // nchw
    const at::Tensor &R, // k2
    const int height, const int width,
    const int pooled_height, const int pooled_width, const bool first,
    const at::Tensor &weight
    ) {
  auto grad_input  =  grad_output.bmm(weight.transpose(1, 2));
  auto grad_weight =  X.transpose(1, 2).bmm(grad_output);
  grad_input = grad_input.view({X.size(0), R.size(0), pooled_height, pooled_width});
  grad_input = crop_gpu_backward(grad_input, R, height, width, first);
  return std::make_tuple(grad_input, grad_weight);
}
