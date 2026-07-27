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

template <typename T>
__global__ void relu_fwd_kernel(const T* __restrict__ input, T* __restrict__ output,
                                size_t total_elements) {
  using VecTraits = VectoredTraits<T>;
  using VecType = typename VecTraits::type;
  constexpr int VEC_SIZE = VecTraits::size;

  size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;
  if (idx >= total_elements) return;

  if (idx + VEC_SIZE <= total_elements) {
    VecType in_vec = *reinterpret_cast<const VecType*>(&input[idx]);
    T* in_vals = reinterpret_cast<T*>(&in_vec);

    VecType out_vec;
    T* out_vals = reinterpret_cast<T*>(&out_vec);

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float val = (float)in_vals[i];
      out_vals[i] = val > 0.0f ? (T)val : (T)0.0f;
    }
    *reinterpret_cast<VecType*>(&output[idx]) = out_vec;
  } else {
    for (size_t i = idx; i < total_elements; ++i) {
      float val = (float)input[i];
      output[i] = val > 0.0f ? (T)val : (T)0.0f;
    }
  }
}

void relu_fwd(const Tensor& input, Tensor& output) {
  DType_t dtype = input.dtype();
  size_t size = input.size();
  if (dtype == DType_t::FP32) {
    constexpr int block_size = 256;
    constexpr int vec_size = VectoredTraits<float>::size;
    size_t num_work_units = (size + vec_size - 1) / vec_size;
    size_t grid_size = (num_work_units + block_size - 1) / block_size;

    relu_fwd_kernel<<<grid_size, block_size>>>(input.data_as<float>(), output.data_as<float>(),
                                               size);
  }
}

signed main() {
  Tensor input({BATCH_SIZE, HEIGHT, WIDTH, CHANNELS}, DType_t::FP32, getGPU());
  Tensor output({BATCH_SIZE, HEIGHT, WIDTH, CHANNELS}, DType_t::FP32, getGPU());
  fill_normal(input, 0.0, 0.1, 123456);

  // Warmup: let the GPU reach steady state before timing
  for (size_t i = 0; i < 50; i++) {
    relu_fwd(input, output);
  }
  cudaDeviceSynchronize();

  cudaEvent_t ev_start, ev_end;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_end);

  constexpr int ITERS = 100;
  cudaEventRecord(ev_start);
  for (int i = 0; i < ITERS; i++) {
    relu_fwd(input, output);
  }
  cudaEventRecord(ev_end);
  cudaEventSynchronize(ev_end);

  float total_ms = 0.0f;
  cudaEventElapsedTime(&total_ms, ev_start, ev_end);

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_end);

  printf("relu: %.3f ms\n", total_ms / ITERS);
}
