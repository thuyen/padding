#include <torch/torch.h>

at::Tensor padh_cpu_forward(
    const at::Tensor &X, // 3d image hwc
    const int pad, const int padw, const bool b
    );

at::Tensor padh_cpu_backward(
    const at::Tensor &X, // 3d image hwc
    const int pad, const int padw, const bool b
    );

at::Tensor padh_gpu_forward(
    const at::Tensor &X, // 3d image hwc
    const int pad, const int padw, const bool b
    );

at::Tensor padh_gpu_backward(
    const at::Tensor &X, // 3d image hwc
    const int pad, const int padw, const bool b
    );

at::Tensor crop_cpu_forward(
    const at::Tensor &X, // nchw
    const at::Tensor &R, // k2
    const int pooled_height, const int pooled_width, const bool first
    );

at::Tensor crop_cpu_backward(
    const at::Tensor &X, // nkchw
    const at::Tensor &R, // k2
    const int height, const int width, const bool first
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


PYBIND11_MODULE(_C, m) {
  m.def("padh_cpu_forward",  &padh_cpu_forward,  "padh_cpu_forward");
  m.def("padh_cpu_backward", &padh_cpu_backward, "padh_cpu_backward");
  m.def("padh_gpu_forward",  &padh_gpu_forward,  "padh_gpu_forward");
  m.def("padh_gpu_backward", &padh_gpu_backward, "padh_gpu_backward");

  m.def("crop_cpu_forward",  &crop_cpu_forward,  "crop_cpu_forward");
  m.def("crop_cpu_backward", &crop_cpu_backward, "crop_cpu_backward");
  m.def("crop_gpu_forward",  &crop_gpu_forward,  "crop_gpu_forward");
  m.def("crop_gpu_backward", &crop_gpu_backward, "crop_gpu_backward");
}
