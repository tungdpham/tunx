
#include "nn/engines/engine_handle.hpp"
#ifdef TUNX_USE_CUDA
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <ctime>

#include "nn/engines/cuda_engine.hpp"

namespace tunx {

#define BLOCK_SIZE 256
#define WARP_SIZE 32

template <typename T>
__global__ void conv2d_add_bias_kernel(T* output, const T* bias, size_t batch_size, size_t output_h,
                                       size_t output_w, size_t out_channels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = batch_size * out_channels * output_h * output_w;

  if (idx >= total_size) return;

  int remaining = idx % (out_channels * output_h * output_w);
  int c = remaining / (output_h * output_w);

  output[idx] += bias[c];
}

template <typename T>
__global__ void conv2d_nchw_bgrad_kernel(const T* gradient, T* grad_bias, size_t batch_size,
                                         size_t output_h, size_t output_w, size_t out_channels) {
  size_t spatial_size = output_h * output_w;
  size_t channel_stride = spatial_size;
  size_t batch_stride = out_channels * spatial_size;

  int c = blockIdx.x;
  if (c >= out_channels) return;

  extern __shared__ char shared_mem[];
  T* shared = reinterpret_cast<T*>(shared_mem);

  T sum = T(0);

  int tid = threadIdx.x;
  int total_elements = batch_size * spatial_size;

  for (int idx = tid; idx < total_elements; idx += blockDim.x) {
    int n = idx / spatial_size;
    int spatial_idx = idx % spatial_size;
    sum += gradient[n * batch_stride + c * channel_stride + spatial_idx];
  }

  shared[tid] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] += shared[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    grad_bias[c] = shared[0];
  }
}

WorkspaceReq CUDAEngine::query_conv2d_graph(engine_handle backend_handle, const Conv2DStats& stats,
                                            DTypeDesc type_desc) {
  return {0, 0, 0};
}

void CUDAEngine::conv2d_fwd(engine_handle backend_handle, const Conv2DStats& stats,
                            const void* input, const void* weight, const void* bias, void* output,
                            void* workspace, DTypeDesc type_desc) {
  throw std::runtime_error("conv2d_fwd not implemented");
}

void CUDAEngine::conv2d_dgrad(engine_handle backend_handle, const Conv2DStats& stats,
                              const void* grad_output, const void* weight, void* grad_input,
                              void* workspace, DTypeDesc type_desc) {
  throw std::runtime_error("conv2d_dgrad not implemented");
}

void CUDAEngine::conv2d_wgrad(engine_handle backend_handle, const Conv2DStats& stats,
                              const void* grad_output, const void* input, void* grad_weight_prev,
                              void* workspace, DTypeDesc type_desc) {
  throw std::runtime_error("conv2d_wgrad not implemented");
}

void CUDAEngine::conv2d_bgrad(engine_handle backend_handle, const Conv2DStats& stats,
                              const void* grad_output, void* grad_bias_prev, void* workspace,
                              DTypeDesc type_desc) {
  throw std::runtime_error("conv2d_bgrad not implemented");
}

}  // namespace tunx

#endif
