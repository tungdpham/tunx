#include "device/cuda_device.hpp"
#include "nn/engines/engine_handle.hpp"
#ifdef TUNX_USE_CUDA
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <ctime>

#include "nn/engines/cuda_engine.hpp"
#include "type/cuda/vectorized_types.cuh"
#include "type/type.hpp"

namespace tunx {

#define BLOCK_SIZE 256
#define WARP_SIZE 32

template <typename T>
__global__ void relu_fwd_vec_kernel(const T* input, T* output, size_t n_vectors) {
  using VecT = typename VectoredTraits<T>::type;
  constexpr int VecSize = VectoredTraits<T>::size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_vectors) return;

  const VecT* input_vec = reinterpret_cast<const VecT*>(input);
  VecT* output_vec = reinterpret_cast<VecT*>(output);

  VecT in_val = input_vec[idx];
  VecT out_val;

  const T* in_ptr = reinterpret_cast<const T*>(&in_val);
  T* out_ptr = reinterpret_cast<T*>(&out_val);

  T zero = static_cast<T>(0);

  for (int i = 0; i < VecSize; ++i) {
    out_ptr[i] = in_ptr[i] > zero ? in_ptr[i] : zero;
  }

  output_vec[idx] = out_val;
}

template <typename T>
__global__ void relu_fwd_tail_kernel(const T* input, T* output, size_t offset,
                                     size_t tail_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tail_elements) return;

  size_t pos = offset + idx;
  T val = input[pos];
  T zero = static_cast<T>(0);
  output[pos] = val > zero ? val : zero;
}

template <typename T>
__global__ void relu_dgrad_vec_kernel(const T* grad_output, T* grad_input, const T* output,
                                      size_t n_vectors) {
  using VecT = typename VectoredTraits<T>::type;
  constexpr int VecSize = VectoredTraits<T>::size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_vectors) return;

  const VecT* grad_out_vec = reinterpret_cast<const VecT*>(grad_output);
  const VecT* out_vec = reinterpret_cast<const VecT*>(output);
  VecT* grad_in_vec = reinterpret_cast<VecT*>(grad_input);

  VecT grad_out_val = grad_out_vec[idx];
  VecT out_val = out_vec[idx];
  VecT grad_in_val;

  const T* grad_out_ptr = reinterpret_cast<const T*>(&grad_out_val);
  const T* out_ptr = reinterpret_cast<const T*>(&out_val);
  T* grad_in_ptr = reinterpret_cast<T*>(&grad_in_val);

  for (int i = 0; i < VecSize; ++i) {
    grad_in_ptr[i] = out_ptr[i] > T(0) ? grad_out_ptr[i] : T(0);
  }

  grad_in_vec[idx] = grad_in_val;
}

template <typename T>
__global__ void relu_dgrad_tail_kernel(const T* grad_output, T* grad_input, const T* output,
                                       size_t offset, size_t tail_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tail_elements) return;

  size_t pos = offset + idx;
  grad_input[pos] = output[pos] > T(0) ? grad_output[pos] : T(0);
}

WorkspaceReq CUDAEngine::query_relu_graph(engine_handle backend_handle, const ReLUStats& stats,
                                          DTypeDesc type_desc) {
  return {0, 0, 0};
}

void CUDAEngine::relu_fwd(engine_handle backend_handle, const ReLUStats& stats, const void* input,
                          void* output, void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  size_t num_elements = stats.batch_size * stats.spatial_size;
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    if (num_elements > 0) {
      constexpr int VecSize = VectoredTraits<T>::size;
      constexpr int threads = 256;
      size_t n_vectors = num_elements / VecSize;
      if (n_vectors > 0) {
        int blocks = (n_vectors + threads - 1) / threads;
        relu_fwd_vec_kernel<T>
            <<<blocks, threads, 0, stream>>>(static_cast<const T*>(input), static_cast<T*>(output),
                                             n_vectors);
      }
      size_t tail_offset = n_vectors * VecSize;
      size_t tail_elements = num_elements - tail_offset;
      if (tail_elements > 0) {
        int blocks = (tail_elements + threads - 1) / threads;
        relu_fwd_tail_kernel<T><<<blocks, threads, 0, stream>>>(
            static_cast<const T*>(input), static_cast<T*>(output), tail_elements, tail_offset);
      }
    }
  });
}

void CUDAEngine::relu_inf(engine_handle backend_handle, const ReLUStats& stats, const void* input,
                          void* output, void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  size_t num_elements = stats.batch_size * stats.spatial_size;
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    if (num_elements > 0) {
      constexpr int VecSize = VectoredTraits<T>::size;
      constexpr int threads = 256;
      size_t n_vectors = num_elements / VecSize;
      if (n_vectors > 0) {
        int blocks = (n_vectors + threads - 1) / threads;
        relu_fwd_vec_kernel<T><<<blocks, threads, 0, stream>>>(
            static_cast<const T*>(input), static_cast<T*>(output), n_vectors);
      }
      size_t tail_offset = n_vectors * VecSize;
      size_t tail_elements = num_elements - tail_offset;
      if (tail_elements > 0) {
        int blocks = (tail_elements + threads - 1) / threads;
        relu_fwd_tail_kernel<T><<<blocks, threads, 0, stream>>>(static_cast<const T*>(input),
                                                                static_cast<T*>(output),
                                                                tail_elements, tail_offset);
      }
    }
  });
}

void CUDAEngine::relu_bwd(engine_handle backend_handle, const ReLUStats& stats,
                          const void* grad_output, void* grad_input, const void* output,
                          void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  size_t num_elements = stats.batch_size * stats.spatial_size;
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    if (num_elements > 0) {
      constexpr int VecSize = VectoredTraits<T>::size;
      constexpr int threads = 256;
      size_t n_vectors = num_elements / VecSize;
      if (n_vectors > 0) {
        int blocks = (n_vectors + threads - 1) / threads;
        relu_dgrad_vec_kernel<T><<<blocks, threads, 0, stream>>>(
            static_cast<const T*>(grad_output), static_cast<T*>(grad_input),
            static_cast<const T*>(output), n_vectors);
      }
      size_t tail_offset = n_vectors * VecSize;
      size_t tail_elements = num_elements - tail_offset;
      if (tail_elements > 0) {
        int blocks = (tail_elements + threads - 1) / threads;
        relu_dgrad_tail_kernel<T><<<blocks, threads, 0, stream>>>(
            static_cast<const T*>(grad_output), static_cast<T*>(grad_input),
            static_cast<const T*>(output), tail_offset, tail_elements);
      }
    }
  });
}

}  // namespace tunx

#endif
