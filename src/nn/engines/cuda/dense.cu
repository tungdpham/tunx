#include "device/cuda_device.hpp"
#include "nn/engines/engine_handle.hpp"
#ifdef TUNX_USE_CUDA
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <ctime>

#include "math/cuda/gemm.hpp"
#include "nn/engines/cuda_engine.hpp"
#include "type/type.hpp"

namespace tunx {

#define BLOCK_SIZE 256
#define WARP_SIZE 32

template <typename IO_T, typename Param_T, typename Compute_T>
__global__ void dense_bgrad_kernel_ex(const IO_T* current_grad, Param_T* grad_bias,
                                      size_t batch_size, size_t output_features) {
  int out_f = blockIdx.x;
  if (out_f >= static_cast<int>(output_features)) return;

  extern __shared__ char shared_mem[];
  Compute_T* shared = reinterpret_cast<Compute_T*>(shared_mem);

  Compute_T sum = Compute_T(0);
  int tid = threadIdx.x;
  for (int n = tid; n < static_cast<int>(batch_size); n += blockDim.x) {
    sum += static_cast<Compute_T>(current_grad[n * output_features + out_f]);
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
    Compute_T prev = static_cast<Compute_T>(grad_bias[out_f]);
    grad_bias[out_f] = static_cast<Param_T>(prev + shared[0]);
  }
}

template <typename IO_T, typename Param_T, typename Compute_T>
__global__ void dense_add_bias_kernel(IO_T* output, const Param_T* bias, size_t batch_size,
                                      size_t output_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = batch_size * output_features;

  if (idx >= total_size) return;

  int out_f = idx % output_features;
  output[idx] += static_cast<IO_T>(bias[out_f]);
}

WorkspaceReq CUDAEngine::query_dense_graph(engine_handle backend_handle, const DenseStats& stats,
                                           DTypeDesc type_desc) {
  return {0, 0, 0};
}

void CUDAEngine::dense_fwd(engine_handle backend_handle, const DenseStats& stats, const void* input,
                           const void* weight, const void* bias, void* output, void* workspace,
                           DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    cuda::gemm_ex<T, T, T, float>(static_cast<const T*>(input), static_cast<const T*>(weight),
                                  static_cast<T*>(output), stats.batch_size, stats.out_features,
                                  stats.in_features, false, true, 1.0f, 0.0f, stats.in_features,
                                  stats.in_features, stats.out_features, stream);
    if (stats.use_bias) {
      int total_size = stats.batch_size * stats.out_features;
      int threads_per_block = 256;
      int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;
      dense_add_bias_kernel<T, T, float><<<num_blocks, threads_per_block, 0, stream>>>(
          static_cast<T*>(output), static_cast<const T*>(bias), stats.batch_size,
          stats.out_features);
    }
  });
}

void CUDAEngine::dense_wgrad(engine_handle backend_handle, const DenseStats& stats,
                             const void* grad_output, const void* input, void* grad_weight_prev,
                             void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    cuda::gemm_ex<T, T, T, float>(static_cast<const T*>(grad_output), static_cast<const T*>(input),
                                  static_cast<T*>(grad_weight_prev), stats.out_features,
                                  stats.in_features, stats.batch_size, true, false, 1.0f, 1.0f,
                                  stats.out_features, stats.in_features, stats.in_features, stream);
  });
}

void CUDAEngine::dense_dgrad(engine_handle backend_handle, const DenseStats& stats,
                             const void* grad_output, const void* weight, void* grad_input,
                             void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    cuda::gemm_ex<T, T, T, float>(static_cast<const T*>(grad_output), static_cast<const T*>(weight),
                                  static_cast<T*>(grad_input), stats.batch_size, stats.in_features,
                                  stats.out_features, false, false, 1.0f, 0.0f, stats.out_features,
                                  stats.in_features, stats.in_features, stream);
  });
}

void CUDAEngine::dense_bgrad(engine_handle backend_handle, const DenseStats& stats,
                             const void* grad_output, void* grad_bias_prev, void* workspace,
                             DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    int threads_per_block_b = 256;
    int num_blocks_b = static_cast<int>(stats.out_features);
    size_t shared_mem_size = threads_per_block_b * sizeof(float);
    dense_bgrad_kernel_ex<T, T, float>
        <<<num_blocks_b, threads_per_block_b, shared_mem_size, stream>>>(
            static_cast<const T*>(grad_output), static_cast<T*>(grad_bias_prev), stats.batch_size,
            stats.out_features);
  });
}

}  // namespace tunx

#endif
