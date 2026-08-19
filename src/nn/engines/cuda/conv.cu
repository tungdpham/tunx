#include "device/cuda_device.hpp"
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
__device__ __forceinline__ T warp_reduce_sum(T sum) {
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  return sum;
}

template <typename IO_T, typename PARAM_T, typename COMPUTE_T>
__global__ void bgrad_reduce_accumulate_kernel(const IO_T* __restrict__ dy,
                                               PARAM_T* __restrict__ db, int batch_size,
                                               int out_features) {
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int lane_id = threadIdx.x % 32;

  if (warp_id >= out_features) return;

  COMPUTE_T sum = COMPUTE_T(0);
  for (int b = lane_id; b < batch_size; b += 32) {
    int idx = b * out_features + warp_id;
    sum += static_cast<COMPUTE_T>(dy[idx]);
  }

  sum = warp_reduce_sum(sum);

  if (lane_id == 0) {
    db[warp_id] = static_cast<PARAM_T>(sum + static_cast<COMPUTE_T>(db[warp_id]));
  }
}

template <typename IO_T, typename PARAM_T>
__global__ void conv2d_add_bias_kernel(IO_T* output, const PARAM_T* bias, size_t batch_size, size_t output_h,
                                       size_t output_w, size_t out_channels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = batch_size * out_channels * output_h * output_w;

  if (idx >= total_size) return;

  int remaining = idx % (out_channels * output_h * output_w);
  int c = remaining / (output_h * output_w);

  output[idx] += static_cast<IO_T>(bias[c]);
}

template <typename IO_T, typename PARAM_T, typename COMPUTE_T>
__global__ void conv2d_nchw_bgrad_kernel(const IO_T* gradient, PARAM_T* grad_bias, size_t batch_size,
                                         size_t output_h, size_t output_w, size_t out_channels) {
  size_t spatial_size = output_h * output_w;
  size_t channel_stride = spatial_size;
  size_t batch_stride = out_channels * spatial_size;

  int c = blockIdx.x;
  if (c >= out_channels) return;

  extern __shared__ char shared_mem[];
  COMPUTE_T* shared = reinterpret_cast<COMPUTE_T*>(shared_mem);

  COMPUTE_T sum = COMPUTE_T(0);

  int tid = threadIdx.x;
  int total_elements = batch_size * spatial_size;

  for (int idx = tid; idx < total_elements; idx += blockDim.x) {
    int n = idx / spatial_size;
    int spatial_idx = idx % spatial_size;
    sum += static_cast<COMPUTE_T>(gradient[n * batch_stride + c * channel_stride + spatial_idx]);
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
    grad_bias[c] = static_cast<PARAM_T>(shared[0]);
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
                              const void* grad_output, const void* input, void* grad_weight,
                              void* workspace, DTypeDesc type_desc) {
  throw std::runtime_error("conv2d_wgrad not implemented");
}

void CUDAEngine::conv2d_bgrad(engine_handle backend_handle, const Conv2DStats& stats,
                              const void* grad_output, void* grad_bias, void* workspace,
                              DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();

  size_t out_channels = stats.out_channels;
  const int64_t output_h = (stats.input_h + stats.pad_h * 2 - stats.kernel_h) / stats.stride_h + 1;
  const int64_t output_w = (stats.input_w + stats.pad_w * 2 - stats.kernel_w) / stats.stride_w + 1;
  size_t num_elements_to_reduce = stats.batch_size * output_h * output_w;

  int threads_per_block = 128;
  int warps_per_block = threads_per_block / 32;
  int num_blocks = (out_channels + warps_per_block - 1) / warps_per_block;

  DISPATCH_DTYPE3(type_desc.io_dtype, type_desc.param_dtype, type_desc.compute_dtype, IO_T, PARAM_T, COMPUTE_T, {
    bgrad_reduce_accumulate_kernel<IO_T, PARAM_T, COMPUTE_T><<<num_blocks, threads_per_block, 0, stream>>>(
        static_cast<const IO_T*>(grad_output), static_cast<PARAM_T*>(grad_bias),
        static_cast<int>(num_elements_to_reduce), static_cast<int>(out_channels));
  });

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Failed to launch conv_bgrad custom kernel: ") +
                             cudaGetErrorString(err));
  }
}

}  // namespace tunx

#endif
