#include <ATen/ATen.h>
#include <iostream>

using namespace std;

template <typename T>
void PADH_FW(
    const int numels,
    const T* image,
    const int channels,
    const int height,
    const int width,
    const int pad,
    const int H,
    const int W,
    T* top_data) {
  int i;
#pragma omp parallel for
  for (i = 0; i < numels; ++i) {
    int w = i % width;
    int h = (i / width)% height;
    int c = (i / width / height) % channels;
    int n =  i / width / height / channels;

    if (w >= W) continue;

    h -= pad;
    if (h < 0) h += H;
    else if ( h >= H) h -= H;

    int j = n * channels * H * width + c * H * width + h* width + w;
    top_data[i] = image[j];
  }
}


template <typename T>
void PADH_BW(
    const int numels,
    const T* grad_output,
    const int channels,
    const int height,
    const int width,
    const int pad,
    const int H,
    const int W,
    T* grad_input) {
  int i;
#pragma omp parallel for
  for (i = 0; i < numels; ++i) {
    int w = i % width;
    int h = (i / width)% height;
    int c = (i / width / height) % channels;
    int n =  i / width / height / channels;

    if (w >= W) continue;

    h -= pad;
    if (h < 0) h += H;
    else if ( h >= H) h -= H;

    int j = n * channels * H * width + c * H * width + h* width + w;
#pragma omp atomic
    grad_input[j] += grad_output[i];
  }
}


at::Tensor padh_cpu_forward(
    const at::Tensor &X, // 3d image hwc
    const int pad, const int pad_w, const bool oneside
    ) {

  int channels = X.size(1), height = X.size(2), width = X.size(3);
  int H = height, W = width;
  height += oneside? pad: 2*pad;
  width  += pad_w;

  at::Tensor output = X.type().zeros({X.size(0), channels, height, width});

  AT_DISPATCH_ALL_TYPES(X.type(), "padh_cpu_forward", [&] {
      PADH_FW<scalar_t>(
          output.numel(),
          X.data<scalar_t>(),
          channels,
          height,
          width,
          pad, H, W,
          output.data<scalar_t>());
  });
  return output;
}

at::Tensor padh_cpu_backward(
    const at::Tensor &X, // 3d image hwc
    const int pad, const int pad_w, const bool oneside
    ) {

  int channels = X.size(1), height = X.size(2), width = X.size(3);
  int H = oneside? height - pad: height - 2*pad;
  int W = width - pad_w;

  at::Tensor grad_input = X.type().zeros(
      {X.size(0), channels, H, W});

  PADH_BW<float>(
      X.numel(),
      X.data<float>(),
      channels,
      height,
      width,
      pad, H, W,
      grad_input.data<float>());

  //AT_DISPATCH_ALL_TYPES(X.type(), "padh_cpu_backward", [&] {
  //    PADH_BW<scalar_t>(
  //        X.numel(),
  //        X.data<scalar_t>(),
  //        channels,
  //        height,
  //        width,
  //        pad, H,
  //        grad_input.data<scalar_t>());
  //});
  return grad_input;
}
