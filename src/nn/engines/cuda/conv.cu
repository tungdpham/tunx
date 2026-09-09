#include "device/cuda_device.hpp"
#include "nn/engines/engine_handle.hpp"
#ifdef TUNX_USE_CUDA
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <ctime>
#include <stdexcept>

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
__global__ void conv2d_add_bias_kernel(IO_T* output, const PARAM_T* bias, size_t batch_size,
                                       size_t output_h, size_t output_w, size_t out_channels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = batch_size * out_channels * output_h * output_w;

  if (idx >= total_size) return;

  int remaining = idx % (out_channels * output_h * output_w);
  int c = remaining / (output_h * output_w);

  output[idx] += static_cast<IO_T>(bias[c]);
}

template <typename IO_T, typename PARAM_T, typename COMPUTE_T>
__global__ void conv2d_nchw_bgrad_kernel(const IO_T* gradient, PARAM_T* grad_bias,
                                         size_t batch_size, size_t output_h, size_t output_w,
                                         size_t out_channels) {
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

template <typename IO_T, typename PARAM_T, typename COMPUTE_T>
__global__ void conv2d_naive_fwd_kernel(const IO_T* input, const PARAM_T* weight,
                                        const PARAM_T* bias, IO_T* output, Conv2DStats stats,
                                        size_t output_h, size_t output_w) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = stats.batch_size * output_h * output_w * stats.out_channels;
  if (index >= total) return;

  size_t oc = index % stats.out_channels;
  size_t spatial = index / stats.out_channels;
  size_t ow = spatial % output_w;
  size_t oh = (spatial / output_w) % output_h;
  size_t batch = spatial / (output_h * output_w);

  COMPUTE_T sum = bias ? static_cast<COMPUTE_T>(bias[oc]) : COMPUTE_T(0);
  for (size_t kh = 0; kh < stats.kernel_h; ++kh) {
    int ih = static_cast<int>(oh * stats.stride_h + kh) - static_cast<int>(stats.pad_h);
    if (ih < 0 || ih >= static_cast<int>(stats.input_h)) continue;
    for (size_t kw = 0; kw < stats.kernel_w; ++kw) {
      int iw = static_cast<int>(ow * stats.stride_w + kw) - static_cast<int>(stats.pad_w);
      if (iw < 0 || iw >= static_cast<int>(stats.input_w)) continue;
      for (size_t ic = 0; ic < stats.in_channels; ++ic) {
        size_t input_index =
            ((batch * stats.input_h + ih) * stats.input_w + iw) * stats.in_channels + ic;
        size_t weight_index =
            ((oc * stats.kernel_h + kh) * stats.kernel_w + kw) * stats.in_channels + ic;
        sum += static_cast<COMPUTE_T>(input[input_index]) *
               static_cast<COMPUTE_T>(weight[weight_index]);
      }
    }
  }
  output[index] = static_cast<IO_T>(sum);
}

template <typename IO_T, typename PARAM_T, typename COMPUTE_T>
__global__ void conv2d_naive_dgrad_kernel(const IO_T* grad_output, const PARAM_T* weight,
                                          IO_T* grad_input, Conv2DStats stats, size_t output_h,
                                          size_t output_w) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = stats.batch_size * stats.input_h * stats.input_w * stats.in_channels;
  if (index >= total) return;

  size_t ic = index % stats.in_channels;
  size_t spatial = index / stats.in_channels;
  size_t iw = spatial % stats.input_w;
  size_t ih = (spatial / stats.input_w) % stats.input_h;
  size_t batch = spatial / (stats.input_h * stats.input_w);
  COMPUTE_T sum = COMPUTE_T(0);

  for (size_t oc = 0; oc < stats.out_channels; ++oc) {
    for (size_t kh = 0; kh < stats.kernel_h; ++kh) {
      int oh_numerator = static_cast<int>(ih + stats.pad_h) - static_cast<int>(kh);
      if (oh_numerator < 0 || oh_numerator % static_cast<int>(stats.stride_h) != 0) continue;
      size_t oh = oh_numerator / stats.stride_h;
      if (oh >= output_h) continue;
      for (size_t kw = 0; kw < stats.kernel_w; ++kw) {
        int ow_numerator = static_cast<int>(iw + stats.pad_w) - static_cast<int>(kw);
        if (ow_numerator < 0 || ow_numerator % static_cast<int>(stats.stride_w) != 0) continue;
        size_t ow = ow_numerator / stats.stride_w;
        if (ow >= output_w) continue;
        size_t output_index = ((batch * output_h + oh) * output_w + ow) * stats.out_channels + oc;
        size_t weight_index =
            ((oc * stats.kernel_h + kh) * stats.kernel_w + kw) * stats.in_channels + ic;
        sum += static_cast<COMPUTE_T>(grad_output[output_index]) *
               static_cast<COMPUTE_T>(weight[weight_index]);
      }
    }
  }
  grad_input[index] = static_cast<IO_T>(sum);
}

template <typename IO_T, typename PARAM_T, typename COMPUTE_T>
__global__ void conv2d_naive_wgrad_kernel(const IO_T* grad_output, const IO_T* input,
                                          PARAM_T* grad_weight, Conv2DStats stats, size_t output_h,
                                          size_t output_w) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = stats.out_channels * stats.kernel_h * stats.kernel_w * stats.in_channels;
  if (index >= total) return;

  size_t ic = index % stats.in_channels;
  size_t kernel = index / stats.in_channels;
  size_t kw = kernel % stats.kernel_w;
  size_t kh = (kernel / stats.kernel_w) % stats.kernel_h;
  size_t oc = kernel / (stats.kernel_h * stats.kernel_w);
  COMPUTE_T sum = static_cast<COMPUTE_T>(grad_weight[index]);

  for (size_t batch = 0; batch < stats.batch_size; ++batch) {
    for (size_t oh = 0; oh < output_h; ++oh) {
      int ih = static_cast<int>(oh * stats.stride_h + kh) - static_cast<int>(stats.pad_h);
      if (ih < 0 || ih >= static_cast<int>(stats.input_h)) continue;
      for (size_t ow = 0; ow < output_w; ++ow) {
        int iw = static_cast<int>(ow * stats.stride_w + kw) - static_cast<int>(stats.pad_w);
        if (iw < 0 || iw >= static_cast<int>(stats.input_w)) continue;
        size_t output_index = ((batch * output_h + oh) * output_w + ow) * stats.out_channels + oc;
        size_t input_index =
            ((batch * stats.input_h + ih) * stats.input_w + iw) * stats.in_channels + ic;
        sum += static_cast<COMPUTE_T>(grad_output[output_index]) *
               static_cast<COMPUTE_T>(input[input_index]);
      }
    }
  }
  grad_weight[index] = static_cast<PARAM_T>(sum);
}

template <typename Kernel, typename... Args>
void launch_naive_conv(Kernel kernel, size_t total, cudaStream_t stream, Args... args) {
  constexpr int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  kernel<<<blocks, threads, 0, stream>>>(args...);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Failed to launch naive Conv2D kernel: ") +
                             cudaGetErrorString(err));
  }
}

WorkspaceReq CUDAEngine::query_conv2d_graph(engine_handle backend_handle, const Conv2DStats& stats,
                                            DTypeDesc type_desc) {
  return {0, 0, 0};
}

void CUDAEngine::conv2d_fwd(engine_handle backend_handle, const Conv2DStats& stats,
                            const void* input, const void* weight, const void* bias, void* output,
                            void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.get_stream().as<cuda_stream>();
  size_t output_h = (stats.input_h + 2 * stats.pad_h - stats.kernel_h) / stats.stride_h + 1;
  size_t output_w = (stats.input_w + 2 * stats.pad_w - stats.kernel_w) / stats.stride_w + 1;
  DISPATCH_DTYPE3(type_desc.io_dtype, type_desc.param_dtype, type_desc.compute_dtype, IO_T, PARAM_T,
                  COMPUTE_T, {
                    launch_naive_conv(conv2d_naive_fwd_kernel<IO_T, PARAM_T, COMPUTE_T>,
                                      stats.batch_size * output_h * output_w * stats.out_channels,
                                      stream, static_cast<const IO_T*>(input),
                                      static_cast<const PARAM_T*>(weight),
                                      static_cast<const PARAM_T*>(bias), static_cast<IO_T*>(output),
                                      stats, output_h, output_w);
                  });
}

void CUDAEngine::conv2d_dgrad(engine_handle backend_handle, const Conv2DStats& stats,
                              const void* grad_output, const void* weight, void* grad_input,
                              void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.get_stream().as<cuda_stream>();
  size_t output_h = (stats.input_h + 2 * stats.pad_h - stats.kernel_h) / stats.stride_h + 1;
  size_t output_w = (stats.input_w + 2 * stats.pad_w - stats.kernel_w) / stats.stride_w + 1;
  DISPATCH_DTYPE3(type_desc.io_dtype, type_desc.param_dtype, type_desc.compute_dtype, IO_T, PARAM_T,
                  COMPUTE_T, {
                    launch_naive_conv(
                        conv2d_naive_dgrad_kernel<IO_T, PARAM_T, COMPUTE_T>,
                        stats.batch_size * stats.input_h * stats.input_w * stats.in_channels,
                        stream, static_cast<const IO_T*>(grad_output),
                        static_cast<const PARAM_T*>(weight), static_cast<IO_T*>(grad_input), stats,
                        output_h, output_w);
                  });
}

void CUDAEngine::conv2d_wgrad(engine_handle backend_handle, const Conv2DStats& stats,
                              const void* grad_output, const void* input, void* grad_weight,
                              void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.get_stream().as<cuda_stream>();
  size_t output_h = (stats.input_h + 2 * stats.pad_h - stats.kernel_h) / stats.stride_h + 1;
  size_t output_w = (stats.input_w + 2 * stats.pad_w - stats.kernel_w) / stats.stride_w + 1;
  DISPATCH_DTYPE3(type_desc.io_dtype, type_desc.param_dtype, type_desc.compute_dtype, IO_T, PARAM_T,
                  COMPUTE_T, {
                    launch_naive_conv(
                        conv2d_naive_wgrad_kernel<IO_T, PARAM_T, COMPUTE_T>,
                        stats.out_channels * stats.kernel_h * stats.kernel_w * stats.in_channels,
                        stream, static_cast<const IO_T*>(grad_output),
                        static_cast<const IO_T*>(input), static_cast<PARAM_T*>(grad_weight), stats,
                        output_h, output_w);
                  });
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

  DISPATCH_DTYPE3(type_desc.io_dtype, type_desc.param_dtype, type_desc.compute_dtype, IO_T, PARAM_T,
                  COMPUTE_T, {
                    bgrad_reduce_accumulate_kernel<IO_T, PARAM_T, COMPUTE_T>
                        <<<num_blocks, threads_per_block, 0, stream>>>(
                            static_cast<const IO_T*>(grad_output), static_cast<PARAM_T*>(grad_bias),
                            static_cast<int>(num_elements_to_reduce),
                            static_cast<int>(out_channels));
                  });

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Failed to launch conv_bgrad custom kernel: ") +
                             cudaGetErrorString(err));
  }
}

}  // namespace tunx

#endif
