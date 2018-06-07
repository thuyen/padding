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

PYBIND11_MODULE(_C, m) {
  m.def("padh_cpu_forward", &padh_cpu_forward, "padh_cpu_forward");
  m.def("padh_cpu_backward", &padh_cpu_backward, "padh_cpu_backward");
  m.def("padh_gpu_forward", &padh_gpu_forward, "padh_gpu_forward");
  m.def("padh_gpu_backward", &padh_gpu_backward, "padh_gpu_backward");
}
