#include <torch/torch.h>

at::Tensor padh_cpu_forward(
    const at::Tensor &X, // 3d image hwc
    const int pad, const int padw, const bool b
    );

at::Tensor padh_cpu_backward(
    const at::Tensor &X, // 3d image hwc
    const int pad, const int padw, const bool b
    );

at::Tensor crop_cpu_forward(
    const at::Tensor &X, // nchw
    const at::Tensor &R, // k2
    int pooled_height, int pooled_width
    );

at::Tensor crop_cpu_backward(
    const at::Tensor &X, // nkchw
    const at::Tensor &R, // k2
    const int height, const int width
    );

PYBIND11_MODULE(_C, m) {
  m.def("padh_cpu_forward",  &padh_cpu_forward,  "padh_cpu_forward");
  m.def("padh_cpu_backward", &padh_cpu_backward, "padh_cpu_backward");

  m.def("crop_cpu_forward",  &crop_cpu_forward,  "crop_cpu_forward");
  m.def("crop_cpu_backward", &crop_cpu_backward, "crop_cpu_backward");
}
