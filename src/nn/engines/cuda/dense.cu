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

template <typename IO_T, typename PARAM_T, typename COMPUTE_T>
__global__ void dense_bgrad_kernel_ex(const IO_T* current_grad, PARAM_T* grad_bias,
                                      size_t batch_size, size_t output_features) {
  int out_f = blockIdx.x;
  if (out_f >= static_cast<int>(output_features)) return;

  extern __shared__ char shared_mem[];
  COMPUTE_T* shared = reinterpret_cast<COMPUTE_T*>(shared_mem);

  COMPUTE_T sum = COMPUTE_T(0);
  int tid = threadIdx.x;
  for (int n = tid; n < static_cast<int>(batch_size); n += blockDim.x) {
    sum += static_cast<COMPUTE_T>(current_grad[n * output_features + out_f]);
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
    COMPUTE_T prev = static_cast<COMPUTE_T>(grad_bias[out_f]);
    grad_bias[out_f] = static_cast<PARAM_T>(prev + shared[0]);
  }
}

template <typename IO_T, typename PARAM_T, typename COMPUTE_T>
__global__ void dense_add_bias_kernel(IO_T* output, const PARAM_T* bias, size_t batch_size,
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
  
  cuda::gemm_ex(input, weight, output, stats.batch_size, stats.out_features,
                stats.in_features, false, true, 1.0f, 0.0f, stats.in_features,
                stats.in_features, stats.out_features, 
                type_desc.io_dtype, type_desc.param_dtype, type_desc.io_dtype, type_desc.compute_dtype, stream);

  if (stats.use_bias) {
    DISPATCH_DTYPE3(type_desc.io_dtype, type_desc.param_dtype, type_desc.compute_dtype, IO_T, PARAM_T,
                    COMPUTE_T, {
                      int total_size = stats.batch_size * stats.out_features;
                      int threads_per_block = 256;
                      int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;
                      dense_add_bias_kernel<IO_T, PARAM_T, COMPUTE_T>
                          <<<num_blocks, threads_per_block, 0, stream>>>(
                              static_cast<IO_T*>(output), static_cast<const PARAM_T*>(bias),
                              stats.batch_size, stats.out_features);
                    });
  }
}

void CUDAEngine::dense_wgrad(engine_handle backend_handle, const DenseStats& stats,
                             const void* grad_output, const void* input, void* grad_weight,
                             void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  cuda::gemm_ex(grad_output, input, grad_weight, stats.out_features, stats.in_features,
                stats.batch_size, true, false, 1.0f, 1.0f, stats.out_features,
                stats.in_features, stats.in_features, 
                type_desc.io_dtype, type_desc.io_dtype, type_desc.param_dtype, type_desc.compute_dtype, stream);
}

void CUDAEngine::dense_dgrad(engine_handle backend_handle, const DenseStats& stats,
                             const void* grad_output, const void* weight, void* grad_input,
                             void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  cuda::gemm_ex(grad_output, weight, grad_input, stats.batch_size, stats.in_features,
                stats.out_features, false, false, 1.0f, 0.0f, stats.out_features,
                stats.in_features, stats.in_features, 
                type_desc.io_dtype, type_desc.param_dtype, type_desc.io_dtype, type_desc.compute_dtype, stream);
}

void CUDAEngine::dense_bgrad(engine_handle backend_handle, const DenseStats& stats,
                             const void* grad_output, void* grad_bias, void* workspace,
                             DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE3(type_desc.io_dtype, type_desc.param_dtype, type_desc.compute_dtype, IO_T, PARAM_T,
                  COMPUTE_T, {
                    int threads_per_block_b = 256;
                    int num_blocks_b = static_cast<int>(stats.out_features);
                    size_t shared_mem_size = threads_per_block_b * sizeof(COMPUTE_T);
                    dense_bgrad_kernel_ex<IO_T, PARAM_T, COMPUTE_T>
                        <<<num_blocks_b, threads_per_block_b, shared_mem_size, stream>>>(
                            static_cast<const IO_T*>(grad_output), static_cast<PARAM_T*>(grad_bias),
                            stats.batch_size, stats.out_features);
                  });
}

}  // namespace tunx

#endif
