#include <torch/torch.h>

at::Tensor padh_cpu_forward(
    const at::Tensor &X, // 3d image hwc
    const int pad
    );

at::Tensor padh_cpu_backward(
    const at::Tensor &X, // 3d image hwc
    const int pad
    );

PYBIND11_MODULE(padding, m) {
  m.def("padh_cpu_forward", &padh_cpu_forward, "padh_cpu_forward");
  m.def("padh_cpu_backward", &padh_cpu_backward, "padh_cpu_backward");
}
