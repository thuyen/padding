#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

//#include "ATen/cuda/CUDATypeConversion.cuh"

#ifndef CUDA_1D_KERNEL_LOOP
#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)
#endif

template <typename T>
__global__ void PADH_FW_K(
    const int numels,
    const T* image,
    const int channels,
    const int height,
    const int width,
    const int pad,
    T* top_data) {
  CUDA_1D_KERNEL_LOOP(i, numels) {
    int H = height - 2*pad;
    int w = i % width;
    int h = (i / width)% height;
    int c = (i / width / height) % channels;
    int n =  i / width / height / channels;

    h -= pad;
    if (h < 0) h += H;
    else if ( h >= H) h -= H;

    int j = n * channels * H * width + c * H * width + h* width + w;
    top_data[i] = image[j];
  }
}


template <typename T>
__global__ void PADH_BW_K(
    const int numels,
    const T* grad_output,
    const int channels,
    const int height,
    const int width,
    const int pad,
    T* grad_input) {
  CUDA_1D_KERNEL_LOOP(i, numels) {
    int H = height - 2*pad;
    int w = i % width;
    int h = (i / width)% height;
    int c = (i / width / height) % channels;
    int n =  i / width / height / channels;

    h -= pad;
    if (h < 0) h += H;
    else if ( h >= H) h -= H;

    int j = n * channels * H * width + c * H * width + h* width + w;
    atomicAdd(grad_input + j, grad_output[i]);
    //grad_input[j] += grad_output[i];
  }
}


at::Tensor padh_gpu_forward(
    const at::Tensor &X, // 3d image hwc
    const int pad
    ) {

  int channels = X.size(1), height = X.size(2), width = X.size(3);
  height += 2*pad;

  at::Tensor output = X.type().zeros({X.size(0), channels, height, width});

  const int output_size = output.numel();
  const int threads = 1024;
  const int blocks = (output_size + threads - 1) / threads;


  AT_DISPATCH_ALL_TYPES(X.type(), "padh_cuda_forward", [&] {
      //using cuda_scalar_t = at::cuda::type<scalar_t>;
      PADH_FW_K<scalar_t><<<blocks, threads>>>(
          output_size,
          X.data<scalar_t>(),
          channels,
          height,
          width,
          pad,
          output.data<scalar_t>());
  });
  return output;
}


at::Tensor padh_gpu_backward(
    const at::Tensor &X, // 3d image hwc
    const int pad
    ) {

  int channels = X.size(1), height = X.size(2), width = X.size(3);

  at::Tensor grad_input = X.type().zeros(
      {X.size(0), channels, height-2*pad, width});

  const int output_size = X.numel();
  const int threads = 1024;
  const int blocks = (output_size + threads - 1) / threads;

  //AT_DISPATCH_ALL_TYPES(X.type(), "padh_cuda_backward", [&] {
  //    //using cuda_scalar_t = at::cuda::type<scalar_t>;
  //    PADH_BW_K<scalar_t><<<blocks, threads>>>(
  //        output_size,
  //        X.data<scalar_t>(),
  //        channels,
  //        height,
  //        width,
  //        pad,
  //        grad_input.data<scalar_t>());
  //});


  PADH_BW_K<float><<<blocks, threads>>>(
      output_size,
      X.data<float>(),
      channels,
      height,
      width,
      pad,
      grad_input.data<float>());
  return grad_input;
}
