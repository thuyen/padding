#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

#include "utils.h"


template <typename T>
__global__ void Crop2DF_FW(
    const int numels,
    const T* image,
    const int K,
    const int16_t* fixs,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    T* top_data) {
  CUDA_1D_KERNEL_LOOP(i, numels) {
  //bKchw
    int w = i % pooled_width;
    int h = (i / pooled_width) % pooled_height;
    int c = (i / pooled_width / pooled_height) % channels;
    int k = (i / pooled_width / pooled_height / channels) % K;
    int n = (i / pooled_width / pooled_height / channels / K);

    const int16_t* pos = fixs + 2*k;
    int row = pos[0] - (pooled_height/2 - h);
    int col = pos[1] - (pooled_width/2  - w);

    if (col < 0 || col >= width) continue;

    if (row < 0) row += height;
    if (row >= height) row -=height;


    int j = n * channels * height * width + c * height * width + row * width + col;

    top_data[i] = image[j];
  }
}


template <typename T>
__global__ void Crop2DL_FW(
    const int numels,
    const T* image,
    const int K,
    const int16_t* fixs,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    T* top_data) {
  //bKhwc
  CUDA_1D_KERNEL_LOOP(i, numels) {
    int c = i % channels;
    int w = (i / channels) % pooled_width;
    int h = (i / channels / pooled_width ) % pooled_height;
    int k = (i / channels / pooled_width / pooled_height) % K;
    int n = (i / channels / pooled_width / pooled_height) / K;


    const int16_t* pos = fixs + 2*k;
    int row = pos[0] - (pooled_height/2 - h);
    int col = pos[1] - (pooled_width/2  - w);

    if (col < 0 || col >= width) continue;

    if (row < 0) row += height;
    if (row >= height) row -=height;


    int j = n * height * width * channels + row * width * channels + col * channels + c;

    top_data[i] = image[j];
  }
}


template <typename T>
__global__ void Crop2DF_BW(
    const int numels,
    const T* grad_output,
    const int K,
    const int16_t* fixs,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    T* grad_input) {
  CUDA_1D_KERNEL_LOOP(i, numels) {
    int w = i % pooled_width;
    int h = (i / pooled_width) % pooled_height;
    int c = (i / pooled_width / pooled_height) % channels;
    int k = (i / pooled_width / pooled_height / channels) % K;
    int n = (i / pooled_width / pooled_height / channels / K);

    const int16_t* pos = fixs + 2*k;
    int row = pos[0] - (pooled_height/2 - h);
    int col = pos[1] - (pooled_width/2  - w);

    if (col < 0 || col >= width) continue;

    if (row < 0) row += height;
    if (row >= height) row -=height;


    int j = n * channels * height * width + c * height * width + row * width + col;

    atomicAdd(grad_input + j, grad_output[i]);
  }
}


template <typename T>
__global__ void Crop2DL_BW(
    const int numels,
    const T* grad_output,
    const int K,
    const int16_t* fixs,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    T* grad_input) {
  CUDA_1D_KERNEL_LOOP(i, numels) {
    int c = i % channels;
    int w = (i / channels) % pooled_width;
    int h = (i / channels / pooled_width ) % pooled_height;
    int k = (i / channels / pooled_width / pooled_height) % K;
    int n = (i / channels / pooled_width / pooled_height) / K;

    const int16_t* pos = fixs + 2*k;
    int row = pos[0] - (pooled_height/2 - h);
    int col = pos[1] - (pooled_width/2  - w);

    if (col < 0 || col >= width) continue;

    if (row < 0) row += height;
    if (row >= height) row -=height;


    int j = n * height * width * channels + row * width * channels + col * channels + c;

    atomicAdd(grad_input + j, grad_output[i]);
  }
}


at::Tensor crop_gpu_forward(
    const at::Tensor &X, // nchw
    const at::Tensor &R, // k2
    const int pooled_height, const int pooled_width, const bool first
    ) {

  at::Tensor output;
  if (first) {
    int channels = X.size(1), height = X.size(2), width = X.size(3);
    int K = R.size(0);
    output = X.type().zeros({X.size(0), K, channels, pooled_height, pooled_width});

    const int output_size = output.numel();
    const int threads = 1024;
    const int blocks = (output_size + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES(X.type(), "crop_cuda_forward", [&] {
        Crop2DF_FW<scalar_t><<<blocks, threads>>>(
            output_size,
            X.data<scalar_t>(),
            R.size(0),
            R.data<int16_t>(),
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            output.data<scalar_t>());
    });
  } else {
    int channels = X.size(3), height = X.size(1), width = X.size(2);
    int K = R.size(0);
    output = X.type().zeros({X.size(0), K, pooled_height, pooled_width, channels});

    const int output_size = output.numel();
    const int threads = 1024;
    const int blocks = (output_size + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES(X.type(), "crop_cuda_forward", [&] {
        Crop2DL_FW<scalar_t><<<blocks, threads>>>(
            output_size,
            X.data<scalar_t>(),
            R.size(0),
            R.data<int16_t>(),
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            output.data<scalar_t>());
    });
  }

  return output;
}


at::Tensor crop_gpu_backward(
    const at::Tensor &X, // nkchw
    const at::Tensor &R, // k2
    const int height, const int width, const bool first
    ) {

  at::Tensor grad_input;
  if (first) {
    int channels = X.size(2);
    int pooled_height = X.size(3), pooled_width = X.size(4);
    int K = R.size(0);
    grad_input = X.type().zeros(
        {X.size(0), channels, height, width});

    const int output_size = X.numel();
    const int threads = 1024;
    const int blocks = (output_size + threads - 1) / threads;

    Crop2DF_BW<float><<<blocks, threads>>>(
        output_size,
        X.data<float>(),
        R.size(0),
        R.data<int16_t>(),
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        grad_input.data<float>());

  } else {
    int channels = X.size(4);
    int pooled_height = X.size(2), pooled_width = X.size(3);
    int K = R.size(0);
    grad_input = X.type().zeros(
        {X.size(0), height, width, channels});

    const int output_size = X.numel();
    const int threads = 1024;
    const int blocks = (output_size + threads - 1) / threads;

    Crop2DL_BW<float><<<blocks, threads>>>(
        output_size,
        X.data<float>(),
        R.size(0),
        R.data<int16_t>(),
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        grad_input.data<float>());
  }

  return grad_input;
}
