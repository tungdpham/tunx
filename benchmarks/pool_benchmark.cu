#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "device/device_manager.hpp"
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"
#include "type/cuda/vectorized_types.cuh"
#include "type/type.hpp"

using namespace tunx;

constexpr size_t BATCH_SIZE = 256;
constexpr size_t HEIGHT = 32;
constexpr size_t WIDTH = 32;
constexpr size_t CHANNELS = 128;

constexpr size_t POOL_H = 3;
constexpr size_t POOL_W = 3;
constexpr size_t STRIDE_H = 1;
constexpr size_t STRIDE_W = 1;
constexpr size_t PAD_H = 0;
constexpr size_t PAD_W = 0;

constexpr size_t OUTPUT_H = (HEIGHT + 2 * PAD_H - POOL_H) / STRIDE_H + 1;
constexpr size_t OUTPUT_W = (WIDTH + 2 * PAD_W - POOL_W) / STRIDE_W + 1;

template <typename T>
__global__ void maxpool_fwd_kernel(const T* __restrict__ input, T* __restrict__ output,
                                   int* __restrict__ mask_indices, size_t batch_size, size_t height,
                                   size_t width, size_t channels, size_t pool_h, size_t pool_w,
                                   size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                                   size_t output_h, size_t output_w) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_outputs = batch_size * output_h * output_w * channels;

  if (idx >= total_outputs) return;

  size_t c = idx % channels;
  size_t ow = (idx / channels) % output_w;
  size_t oh = (idx / (channels * output_w)) % output_h;
  size_t b = idx / (channels * output_w * output_h);

  int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
  int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
  int h_end = min(h_start + static_cast<int>(pool_h), static_cast<int>(height));
  int w_end = min(w_start + static_cast<int>(pool_w), static_cast<int>(width));
  h_start = max(h_start, 0);
  w_start = max(w_start, 0);

  float max_val = -1e30f;
  int max_idx = -1;
  for (int h = h_start; h < h_end; ++h) {
    for (int w = w_start; w < w_end; ++w) {
      size_t input_idx = ((b * height + h) * width + w) * channels + c;
      float val = static_cast<float>(input[input_idx]);
      if (val > max_val) {
        max_val = val;
        int h_start_unclamped = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
        int w_start_unclamped = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
        int rel_h = h - h_start_unclamped;
        int rel_w = w - w_start_unclamped;
        max_idx = rel_h * static_cast<int>(pool_w) + rel_w;
      }
    }
  }

  output[idx] = static_cast<T>(max_val);
  if (mask_indices) {
    mask_indices[idx] = max_idx;
  }
}

template <typename T>
__global__ void maxpool_bwd_kernel(const T* __restrict__ grad_output, T* __restrict__ grad_input,
                                   const int* __restrict__ mask_indices, size_t batch_size,
                                   size_t channels, size_t output_h, size_t output_w,
                                   size_t input_h, size_t input_w, size_t pool_w, size_t stride_h,
                                   size_t stride_w, size_t pad_h, size_t pad_w) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_outputs = batch_size * output_h * output_w * channels;

  if (idx >= total_outputs) return;

  int rel_idx = mask_indices[idx];
  if (rel_idx >= 0) {
    size_t c = idx % channels;
    size_t ow = (idx / channels) % output_w;
    size_t oh = (idx / (channels * output_w)) % output_h;
    size_t b = idx / (channels * output_w * output_h);

    int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
    int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);

    int rel_h = rel_idx / static_cast<int>(pool_w);
    int rel_w = rel_idx % static_cast<int>(pool_w);

    int h = h_start + rel_h;
    int w = w_start + rel_w;

    if (h >= 0 && h < static_cast<int>(input_h) && w >= 0 && w < static_cast<int>(input_w)) {
      size_t in_idx = ((b * input_h + h) * input_w + w) * channels + c;
      atomicAdd(&grad_input[in_idx], static_cast<T>(grad_output[idx]));
    }
  }
}

void maxpool_fwd(const Tensor& input, Tensor& output, Tensor& mask) {
  DType_t dtype = input.dtype();
  if (dtype == DType_t::FP32) {
    constexpr int block_size = 256;
    size_t total_outputs = BATCH_SIZE * OUTPUT_H * OUTPUT_W * CHANNELS;
    size_t grid_size = (total_outputs + block_size - 1) / block_size;

    maxpool_fwd_kernel<<<grid_size, block_size>>>(
        input.data_as<float>(), output.data_as<float>(), mask.data_as<int>(), BATCH_SIZE, HEIGHT,
        WIDTH, CHANNELS, POOL_H, POOL_W, STRIDE_H, STRIDE_W, PAD_H, PAD_W, OUTPUT_H, OUTPUT_W);
  }
}

void maxpool_bwd(const Tensor& grad_output, Tensor& grad_input, const Tensor& mask) {
  DType_t dtype = grad_output.dtype();
  if (dtype == DType_t::FP32) {
    constexpr int block_size = 256;
    size_t total_outputs = BATCH_SIZE * OUTPUT_H * OUTPUT_W * CHANNELS;
    size_t grid_size = (total_outputs + block_size - 1) / block_size;

    maxpool_bwd_kernel<<<grid_size, block_size>>>(
        grad_output.data_as<float>(), grad_input.data_as<float>(), mask.data_as<int>(), BATCH_SIZE,
        CHANNELS, OUTPUT_H, OUTPUT_W, HEIGHT, WIDTH, POOL_W, STRIDE_H, STRIDE_W, PAD_H, PAD_W);
  }
}

template <int PH, int PW, int SH, int SW, int pad_h, int pad_w>
__global__ void maxpool_fwd_kernel_opt(const float* __restrict__ input, float* __restrict__ output,
                                   int* __restrict__ mask_indices, size_t height, size_t width,
                                   size_t channels, size_t output_h, size_t output_w) {
  int c_vec = threadIdx.x; // 0..31
  int ow = blockIdx.x * blockDim.y + threadIdx.y;
  int oh = blockIdx.y;
  int b = blockIdx.z;

  if (ow >= output_w) return;

  float4 max_val = make_float4(-1e30f, -1e30f, -1e30f, -1e30f);
  int4 max_idx = make_int4(-1, -1, -1, -1);

  size_t batch_base = b * height * width * channels;

  #pragma unroll
  for (int rel_h = 0; rel_h < PH; ++rel_h) {
    int h = (oh * SH - pad_h) + rel_h;
    if (h >= 0 && h < height) {
      size_t row_base = batch_base + h * width * channels;
      #pragma unroll
      for (int rel_w = 0; rel_w < PW; ++rel_w) {
        int w = (ow * SW - pad_w) + rel_w;
        if (w >= 0 && w < width) {
          size_t input_idx = row_base + w * channels + c_vec * 4;
          float4 val = __ldg(reinterpret_cast<const float4*>(&input[input_idx]));
          int rel = rel_h * PW + rel_w;
          
          if (val.x > max_val.x) { max_val.x = val.x; max_idx.x = rel; }
          if (val.y > max_val.y) { max_val.y = val.y; max_idx.y = rel; }
          if (val.z > max_val.z) { max_val.z = val.z; max_idx.z = rel; }
          if (val.w > max_val.w) { max_val.w = val.w; max_idx.w = rel; }
        }
      }
    }
  }

  size_t out_idx = ((b * output_h + oh) * output_w + ow) * channels + c_vec * 4;
  *reinterpret_cast<float4*>(&output[out_idx]) = max_val;
  if (mask_indices) {
    *reinterpret_cast<int4*>(&mask_indices[out_idx]) = max_idx;
  }
}

template <int PH, int PW, int SH, int SW, int pad_h, int pad_w>
__global__ void maxpool_bwd_kernel_opt(const float* __restrict__ grad_output, float* __restrict__ grad_input,
                                   const int* __restrict__ mask_indices, size_t height, size_t width,
                                   size_t channels, size_t output_h, size_t output_w) {
  int c_vec = threadIdx.x; // 0..31
  int ow = blockIdx.x * blockDim.y + threadIdx.y;
  int oh = blockIdx.y;
  int b = blockIdx.z;

  if (ow >= output_w) return;

  size_t out_idx = ((b * output_h + oh) * output_w + ow) * channels + c_vec * 4;

  float4 grad_val = __ldg(reinterpret_cast<const float4*>(&grad_output[out_idx]));
  int4 mask_val = __ldg(reinterpret_cast<const int4*>(&mask_indices[out_idx]));

  int h_start = oh * SH - pad_h;
  int w_start = ow * SW - pad_w;

  size_t batch_base = b * height * width * channels;

  #define ADD_GRAD(grad, rel_idx, c_offset) \
    if ((rel_idx) >= 0) { \
      int rel_h = (rel_idx) / PW; \
      int rel_w = (rel_idx) % PW; \
      int h = h_start + rel_h; \
      int w = w_start + rel_w; \
      size_t in_idx = batch_base + (h * width + w) * channels + c_vec * 4 + (c_offset); \
      atomicAdd(&grad_input[in_idx], (grad)); \
    }

  ADD_GRAD(grad_val.x, mask_val.x, 0);
  ADD_GRAD(grad_val.y, mask_val.y, 1);
  ADD_GRAD(grad_val.z, mask_val.z, 2);
  ADD_GRAD(grad_val.w, mask_val.w, 3);
  
  #undef ADD_GRAD
}

void maxpool_fwd_opt(const Tensor& input, Tensor& output, Tensor& mask) {
  constexpr int block_x = 32;
  constexpr int block_y = 8;
  dim3 block(block_x, block_y);
  dim3 grid((OUTPUT_W + block_y - 1) / block_y, OUTPUT_H, BATCH_SIZE);

  maxpool_fwd_kernel_opt<POOL_H, POOL_W, STRIDE_H, STRIDE_W, PAD_H, PAD_W><<<grid, block>>>(
      input.data_as<float>(), output.data_as<float>(), mask.data_as<int>(), HEIGHT,
      WIDTH, CHANNELS, OUTPUT_H, OUTPUT_W);
}

void maxpool_bwd_opt(const Tensor& grad_output, Tensor& grad_input, const Tensor& mask) {
  constexpr int block_x = 32;
  constexpr int block_y = 8;
  dim3 block(block_x, block_y);
  dim3 grid((OUTPUT_W + block_y - 1) / block_y, OUTPUT_H, BATCH_SIZE);

  maxpool_bwd_kernel_opt<POOL_H, POOL_W, STRIDE_H, STRIDE_W, PAD_H, PAD_W><<<grid, block>>>(
      grad_output.data_as<float>(), grad_input.data_as<float>(), mask.data_as<int>(), HEIGHT, WIDTH,
      CHANNELS, OUTPUT_H, OUTPUT_W);
}

signed main() {
  Tensor input({BATCH_SIZE, HEIGHT, WIDTH, CHANNELS}, DType_t::FP32, getGPU());
  Tensor output({BATCH_SIZE, OUTPUT_H, OUTPUT_W, CHANNELS}, DType_t::FP32, getGPU());
  Tensor mask({BATCH_SIZE, OUTPUT_H, OUTPUT_W, CHANNELS}, DType_t::INT32, getGPU());
  Tensor grad_output({BATCH_SIZE, OUTPUT_H, OUTPUT_W, CHANNELS}, DType_t::FP32, getGPU());
  Tensor grad_input({BATCH_SIZE, HEIGHT, WIDTH, CHANNELS}, DType_t::FP32, getGPU());

  fill_normal(input, 0.0, 0.1, 123456);
  fill_normal(grad_output, 0.0, 0.1, 654321);
  fill(grad_input, 0.0f);

  constexpr int ITERS = 100;

  // Warmup: let the GPU reach steady state before timing
  for (size_t i = 0; i < 50; i++) {
    maxpool_fwd(input, output, mask);
  }
  cudaDeviceSynchronize();

  cudaEvent_t ev_start, ev_end;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_end);

  cudaEventRecord(ev_start);
  for (int i = 0; i < ITERS; i++) {
    maxpool_fwd(input, output, mask);
  }
  cudaEventRecord(ev_end);
  cudaEventSynchronize(ev_end);

  float fwd_total_ms = 0.0f;
  cudaEventElapsedTime(&fwd_total_ms, ev_start, ev_end);

  // Warmup backward
  for (size_t i = 0; i < 50; i++) {
    maxpool_bwd(grad_output, grad_input, mask);
  }
  cudaDeviceSynchronize();

  cudaEventRecord(ev_start);
  for (int i = 0; i < ITERS; i++) {
    maxpool_bwd(grad_output, grad_input, mask);
  }
  cudaEventRecord(ev_end);
  cudaEventSynchronize(ev_end);

  float bwd_total_ms = 0.0f;
  cudaEventElapsedTime(&bwd_total_ms, ev_start, ev_end);



  printf("maxpool_fwd: %.3f ms\n", fwd_total_ms / ITERS);
  printf("maxpool_bwd: %.3f ms\n", bwd_total_ms / ITERS);

  // Warmup Forward Opt
  for (size_t i = 0; i < 50; i++) {
    maxpool_fwd_opt(input, output, mask);
  }
  cudaDeviceSynchronize();

  cudaEventRecord(ev_start);
  for (int i = 0; i < ITERS; i++) {
    maxpool_fwd_opt(input, output, mask);
  }
  cudaEventRecord(ev_end);
  cudaEventSynchronize(ev_end);

  float fwd_opt_total_ms = 0.0f;
  cudaEventElapsedTime(&fwd_opt_total_ms, ev_start, ev_end);

  // Warmup Backward Opt
  for (size_t i = 0; i < 50; i++) {
    maxpool_bwd_opt(grad_output, grad_input, mask);
  }
  cudaDeviceSynchronize();

  cudaEventRecord(ev_start);
  for (int i = 0; i < ITERS; i++) {
    maxpool_bwd_opt(grad_output, grad_input, mask);
  }
  cudaEventRecord(ev_end);
  cudaEventSynchronize(ev_end);

  float bwd_opt_total_ms = 0.0f;
  cudaEventElapsedTime(&bwd_opt_total_ms, ev_start, ev_end);

  printf("maxpool_fwd_opt: %.3f ms\n", fwd_opt_total_ms / ITERS);
  printf("maxpool_bwd_opt: %.3f ms\n", bwd_opt_total_ms / ITERS);

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_end);
}
