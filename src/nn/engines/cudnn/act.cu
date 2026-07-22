#ifdef TUNX_USE_CUDNN
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn_frontend.h>
#include <cudnn_graph.h>
#include <fmt/core.h>

#include "nn/engines/cudnn_engine.hpp"
#include "nn/engines/engine_handle.hpp"
#include "nn/stats/stats.hpp"
#include "type/type.hpp"

namespace tunx {

template <typename T>
__global__ void relu_fwd_kernel(const T* __restrict__ input, T* __restrict__ output,
                                bool* __restrict__ mask, size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  float val = (float)input[idx];
  bool m = val > 0.0f;
  mask[idx] = m;
  output[idx] = m ? (T)val : (T)0.0f;
}

template <typename T>
__global__ void relu_inf_kernel(const T* __restrict__ input, T* __restrict__ output,
                                size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  float val = (float)input[idx];
  output[idx] = val > 0.0f ? (T)val : (T)0.0f;
}

template <typename T>
__global__ void relu_bwd_kernel(const T* __restrict__ grad_output, T* __restrict__ grad_input,
                                const bool* __restrict__ mask, size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  grad_input[idx] = mask[idx] ? grad_output[idx] : (T)0.0f;
}

WorkspaceReq CuDNNEngine::query_relu_graph(engine_handle backend_handle, const ReLUStats& stats,
                                           DTypeDesc type_desc) {
  return {0, 0, 0};
}

void CuDNNEngine::relu_fwd(engine_handle backend_handle, const ReLUStats& stats, const void* input,
                           void* output, bool* mask, void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  size_t total_elements = stats.batch_size * stats.spatial_size;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  DISPATCH_DTYPE(type_desc.io_dtype, T, {
    relu_fwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<T*>(output), mask, total_elements);
  });
}

void CuDNNEngine::relu_inf(engine_handle backend_handle, const ReLUStats& stats, const void* input,
                           void* output, void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  size_t total_elements = stats.batch_size * stats.spatial_size;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  DISPATCH_DTYPE(type_desc.io_dtype, T, {
    relu_inf_kernel<T><<<blocks, threads, 0, stream>>>(static_cast<const T*>(input),
                                                       static_cast<T*>(output), total_elements);
  });
}

void CuDNNEngine::relu_bwd(engine_handle backend_handle, const ReLUStats& stats,
                           const void* grad_output, void* grad_input, const bool* mask,
                           void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  size_t total_elements = stats.batch_size * stats.spatial_size;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  DISPATCH_DTYPE(type_desc.io_dtype, T, {
    relu_bwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(grad_output), static_cast<T*>(grad_input), mask, total_elements);
  });
}

}  // namespace tunx

#endif
