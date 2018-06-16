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
    const int pad, const int padw, const bool onesided,
    const at::Tensor weight, const at::Tensor bias,
    at::IntList stride, int64_t groups
    ) {

  auto& ctx = at::globalContext();

  padded = padh_gpu_forward(X, padh, padw, onsided);
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
    const int pad, const int padw, const bool onesided,
    const at::Tensor weight, const at::Tensor bias,
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
  //cudnn_convolution_backward(self, grad_output, weight,
  //    padding, stride, dilation, groups, benchmark, deterministic, std::array<bool,3> output_mask)
  //
  at::Tensor grad_input  = std::get<0>(ret);
  at::Tensor grad_weight = std::get<1>(ret);
  at::Tensor grad_bias   = std::get<2>(ret);
  grad_input = padh_gpu_forward(grad_input, padh, padw, false);
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}


at::Tensor svf2d_gpu_forward(
    const at::Tensor &X, // nchw
    const at::Tensor &R, // k2
    const int pooled_height, const int pooled_width, const bool first,
    const at::Tensor &weight
    ) {
  auto cropped = crop_gpu_forward(X, R, pooled_height, pooled_width, false);
  return at::matmul(cropped, weight);
}


at::Tensor svf2d_gpu_backward(
    const at::Tensor &grad_output // nchw
    const at::Tensor &X, // nchw
    const at::Tensor &R, // k2
    const int pooled_height, const int pooled_width, const bool first,
    const at::Tensor &weight
    ) {
  auto grad_input   = at::matmul(grad_output, weight);
  auto grad_weight = at::matmul(grad_output, X);
  grad_input = crop_gpu_bacward(grad_input, pooled_height, pooled_width, false);

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}
