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
#include "type/cuda/vectorized_types.cuh"

namespace tunx {

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

template <typename T>
__global__ void relu_inf_kernel(const T* __restrict__ input, T* __restrict__ output,
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

template <typename T>
__global__ void relu_bwd_kernel(const T* __restrict__ grad_output, T* __restrict__ grad_input,
                                const T* __restrict__ output, size_t total_elements) {
  using VecTraits = VectoredTraits<T>;
  using VecType = typename VecTraits::type;
  constexpr int VEC_SIZE = VecTraits::size;

  size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;
  if (idx >= total_elements) return;

  if (idx + VEC_SIZE <= total_elements) {
    VecType grad_out_vec = *reinterpret_cast<const VecType*>(&grad_output[idx]);
    VecType out_vec = *reinterpret_cast<const VecType*>(&output[idx]);
    T* grad_out_vals = reinterpret_cast<T*>(&grad_out_vec);
    T* out_vals = reinterpret_cast<T*>(&out_vec);

    VecType grad_in_vec;
    T* grad_in_vals = reinterpret_cast<T*>(&grad_in_vec);

    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      grad_in_vals[i] = (float)out_vals[i] > 0.0f ? grad_out_vals[i] : (T)0.0f;
    }
    *reinterpret_cast<VecType*>(&grad_input[idx]) = grad_in_vec;
  } else {
    for (size_t i = idx; i < total_elements; ++i) {
      grad_input[i] = (float)output[i] > 0.0f ? grad_output[i] : (T)0.0f;
    }
  }
}

WorkspaceReq CuDNNEngine::query_relu_graph(engine_handle backend_handle, const ReLUStats& stats,
                                           DTypeDesc type_desc) {
  return {0, 0, 0};
}

void CuDNNEngine::relu_fwd(engine_handle backend_handle, const ReLUStats& stats, const void* input,
                           void* output, void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  size_t total_elements = stats.batch_size * stats.spatial_size;
  int threads = 256;

  DISPATCH_DTYPE(type_desc.io_dtype, T, {
    constexpr int vec_size = VectoredTraits<T>::size;
    int blocks = (total_elements + threads * vec_size - 1) / (threads * vec_size);
    relu_fwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<T*>(output), total_elements);
  });
}

void CuDNNEngine::relu_inf(engine_handle backend_handle, const ReLUStats& stats, const void* input,
                           void* output, void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  size_t total_elements = stats.batch_size * stats.spatial_size;
  int threads = 256;

  DISPATCH_DTYPE(type_desc.io_dtype, T, {
    constexpr int vec_size = VectoredTraits<T>::size;
    int blocks = (total_elements + threads * vec_size - 1) / (threads * vec_size);
    relu_inf_kernel<T><<<blocks, threads, 0, stream>>>(static_cast<const T*>(input),
                                                       static_cast<T*>(output), total_elements);
  });
}

void CuDNNEngine::relu_bwd(engine_handle backend_handle, const ReLUStats& stats,
                           const void* grad_output, void* grad_input, const void* output,
                           void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  size_t total_elements = stats.batch_size * stats.spatial_size;
  int threads = 256;

  DISPATCH_DTYPE(type_desc.io_dtype, T, {
    constexpr int vec_size = VectoredTraits<T>::size;
    int blocks = (total_elements + threads * vec_size - 1) / (threads * vec_size);
    relu_bwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(grad_output), static_cast<T*>(grad_input),
        static_cast<const T*>(output), total_elements);
  });
}

}  // namespace tunx

#endif
