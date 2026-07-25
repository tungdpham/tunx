#include "device/cuda_device.hpp"
#include "nn/engines/engine_handle.hpp"
#ifdef TUNX_USE_CUDA
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <ctime>

#include "cuda/helpers.cuh"
#include "nn/engines/cuda_engine.hpp"
#include "type/cuda/vectorized_types.cuh"
#include "type/type.hpp"

namespace tunx {

#define BLOCK_SIZE 256
#define WARP_SIZE 32

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  val = warpReduceSum(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : T(0);
  if (wid == 0) val = warpReduceSum(val);

  return val;
}

template <typename IO_T, typename Param_T, typename Compute_T>
__global__ void norm_add_bias_kernel(IO_T* output, const Param_T* bias, size_t batch_size,
                                     size_t output_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = batch_size * output_features;

  if (idx >= total_size) return;

  int out_f = idx % output_features;
  output[idx] += static_cast<IO_T>(bias[out_f]);
}

template <typename T>
struct WelfordData {
  T mean;
  T m2;
  T count;

  __device__ WelfordData()
      : mean(0),
        m2(0),
        count(0) {}
  __device__ WelfordData(T m, T v, T c)
      : mean(m),
        m2(v),
        count(c) {}
};

template <typename T>
__device__ WelfordData<T> welford_merge(WelfordData<T> a, WelfordData<T> b) {
  if (b.count == T(0)) return a;
  if (a.count == T(0)) return b;

  T new_count = a.count + b.count;
  T delta = b.mean - a.mean;
  T new_mean = a.mean + (delta * b.count) / new_count;
  T new_m2 = a.m2 + b.m2 + (delta * delta * a.count * b.count) / new_count;

  return WelfordData<T>(new_mean, new_m2, new_count);
}

template <typename T>
__device__ WelfordData<T> warpReduceWelford(WelfordData<T> val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    T other_mean = __shfl_down_sync(0xffffffff, val.mean, offset);
    T other_m2 = __shfl_down_sync(0xffffffff, val.m2, offset);
    T other_count = __shfl_down_sync(0xffffffff, val.count, offset);
    val = welford_merge(val, WelfordData<T>(other_mean, other_m2, other_count));
  }
  return val;
}

template <typename T>
__device__ WelfordData<T> blockReduceWelford(WelfordData<T> val) {
  static __shared__ T shared_mean[32];
  static __shared__ T shared_m2[32];
  static __shared__ T shared_count[32];

  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  val = warpReduceWelford(val);

  if (lane == 0) {
    shared_mean[wid] = val.mean;
    shared_m2[wid] = val.m2;
    shared_count[wid] = val.count;
  }
  __syncthreads();

  WelfordData<T> block_val;

  if (threadIdx.x < (blockDim.x / WARP_SIZE)) {
    block_val.mean = shared_mean[threadIdx.x];
    block_val.m2 = shared_m2[threadIdx.x];
    block_val.count = shared_count[threadIdx.x];
  }

  if (wid == 0) {
    block_val = warpReduceWelford(block_val);
  }

  return block_val;
}

template <typename T>
__global__ void batchnorm_stats_kernel(const T* __restrict__ input, float* __restrict__ mean_out,
                                       float* __restrict__ inv_std_out,
                                       float* __restrict__ running_mean,
                                       float* __restrict__ running_var, size_t N, size_t C,
                                       size_t S, float momentum, float epsilon) {
  int c = blockIdx.x;
  if (c >= C) return;

  size_t channel_stride = C * S;
  size_t channel_offset = c * S;
  size_t count = N * S;

  WelfordData<float> thread;

  for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
    size_t n = i / S;
    size_t s = i % S;
    size_t idx = n * channel_stride + channel_offset + s;

    float val = static_cast<float>(input[idx]);

    thread.count += 1.0f;
    float delta = val - thread.mean;
    thread.mean += delta / thread.count;
    float delta2 = val - thread.mean;
    thread.m2 += delta * delta2;
  }

  WelfordData<float> result = blockReduceWelford(thread);

  if (threadIdx.x == 0) {
    float mu = result.mean;

    float var = result.m2 / result.count;

    mean_out[c] = mu;

    float inv_std = rsqrt(var + epsilon);
    inv_std_out[c] = inv_std;

    float unbiased_var = (result.count > 1.0f) ? (result.m2 / (result.count - 1.0f)) : 0.0f;

    running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mu;
    running_var[c] = (1.0f - momentum) * running_var[c] + momentum * unbiased_var;
  }
}

template <typename T>
__global__ void batchnorm_stats_kernel_vec(const T* __restrict__ input,
                                           float* __restrict__ mean_out,
                                           float* __restrict__ inv_std_out,
                                           float* __restrict__ running_mean,
                                           float* __restrict__ running_var, size_t N, size_t C,
                                           size_t S, float momentum, float epsilon) {
  using VecT = typename VectoredTraits<T>::type;
  constexpr int vec_size = VectoredTraits<T>::size;

  int c = blockIdx.x;
  if (c >= C) return;

  size_t channel_stride = C * S;
  size_t channel_offset = c * S;
  size_t count = N * S;
  size_t num_vectors = count / vec_size;

  WelfordData<float> thread;

  for (size_t i = threadIdx.x; i < num_vectors; i += blockDim.x) {
    size_t scalar_idx_start = i * vec_size;
    size_t n = scalar_idx_start / S;
    size_t s = scalar_idx_start % S;
    size_t idx = n * channel_stride + channel_offset + s;

    VecT val_vec = *reinterpret_cast<const VecT*>(&input[idx]);
    const T* val_arr = reinterpret_cast<const T*>(&val_vec);

#pragma unroll
    for (int k = 0; k < vec_size; ++k) {
      float val = static_cast<float>(val_arr[k]);
      thread.count += 1.0f;
      float delta = val - thread.mean;
      thread.mean += delta / thread.count;
      float delta2 = val - thread.mean;
      thread.m2 += delta * delta2;
    }
  }

  WelfordData<float> result = blockReduceWelford(thread);

  if (threadIdx.x == 0) {
    float mu = result.mean;
    float var = result.m2 / result.count;
    mean_out[c] = mu;
    float inv_std = rsqrt(var + epsilon);
    inv_std_out[c] = inv_std;
    float unbiased_var = (result.count > 1.0f) ? (result.m2 / (result.count - 1.0f)) : 0.0f;
    running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mu;
    running_var[c] = (1.0f - momentum) * running_var[c] + momentum * unbiased_var;
  }
}

template <typename T>
__global__ void batchnorm_fwd_kernel(const T* __restrict__ input, const float* __restrict__ mean,
                                     const float* __restrict__ inv_std,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta, T* __restrict__ output,
                                     float* __restrict__ normalized_cache, size_t N, size_t C,
                                     size_t S, bool affine) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_elements = N * C * S;

  if (idx < total_elements) {
    int c = (idx / S) % C;

    float mu = mean[c];
    float istd = inv_std[c];
    float x = static_cast<float>(input[idx]);

    float norm = (x - mu) * istd;

    if (normalized_cache) normalized_cache[idx] = norm;

    float res = norm;
    if (affine) {
      res = res * gamma[c] + beta[c];
    }
    output[idx] = static_cast<T>(res);
  }
}

template <typename T>
__global__ void batchnorm_fwd_kernel_vec(
    const T* __restrict__ input, const float* __restrict__ mean, const float* __restrict__ inv_std,
    const float* __restrict__ gamma, const float* __restrict__ beta, T* __restrict__ output,
    float* __restrict__ normalized_cache, size_t N, size_t C, size_t S, bool affine) {
  using VecT = typename VectoredTraits<T>::type;
  constexpr int vec_size = VectoredTraits<T>::size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_vectors = (N * C * S) / vec_size;

  if (idx < total_vectors) {
    size_t scalar_idx = idx * vec_size;
    int c = (scalar_idx / S) % C;

    float mu = mean[c];
    float istd = inv_std[c];
    float g = (affine && gamma) ? static_cast<float>(gamma[c]) : 1.0f;
    float b = (affine && beta) ? static_cast<float>(beta[c]) : 0.0f;

    VecT x_vec = reinterpret_cast<const VecT*>(input)[idx];
    const T* x_arr = reinterpret_cast<const T*>(&x_vec);

    VecT out_vec;
    T* out_arr = reinterpret_cast<T*>(&out_vec);

#pragma unroll
    for (int k = 0; k < vec_size; ++k) {
      float x = static_cast<float>(x_arr[k]);
      float norm = (x - mu) * istd;
      if (normalized_cache) normalized_cache[scalar_idx + k] = norm;

      float res = norm;
      if (affine) {
        res = res * g + b;
      }
      out_arr[k] = static_cast<T>(res);
    }

    reinterpret_cast<VecT*>(output)[idx] = out_vec;
  }
}

template <typename T>
__global__ void batchnorm_wgrad_bgrad_reduce_kernel(const T* __restrict__ grad_output,
                                                    const float* __restrict__ normalized_input,
                                                    float* __restrict__ d_gamma,
                                                    float* __restrict__ d_beta, size_t N, size_t C,
                                                    size_t S) {
  int c = blockIdx.x;
  if (c >= C) return;

  size_t count = N * S;
  float sum_dy = 0.0f;
  float sum_dy_x_norm = 0.0f;

  size_t stride = C * S;
  size_t offset = c * S;

  for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
    size_t n = i / S;
    size_t s = i % S;
    size_t idx = n * stride + offset + s;

    float dy = static_cast<float>(grad_output[idx]);
    float x_hat = normalized_input[idx];

    sum_dy += dy;
    sum_dy_x_norm += dy * x_hat;
  }

  sum_dy = blockReduceSum(sum_dy);
  sum_dy_x_norm = blockReduceSum(sum_dy_x_norm);

  if (threadIdx.x == 0) {
    d_gamma[c] = sum_dy_x_norm;
    d_beta[c] = sum_dy;
  }
}

template <typename T>
__global__ void batchnorm_wgrad_bgrad_reduce_kernel_vec(const T* __restrict__ grad_output,
                                                        const float* __restrict__ normalized_input,
                                                        float* __restrict__ d_gamma,
                                                        float* __restrict__ d_beta, size_t N,
                                                        size_t C, size_t S) {
  using VecT = typename VectoredTraits<T>::type;
  constexpr int vec_size = VectoredTraits<T>::size;

  int c = blockIdx.x;
  if (c >= C) return;

  size_t count = N * S;
  size_t num_vectors = count / vec_size;

  float sum_dy = 0.0f;
  float sum_dy_x_norm = 0.0f;

  size_t stride = C * S;
  size_t offset = c * S;

  for (size_t i = threadIdx.x; i < num_vectors; i += blockDim.x) {
    size_t scalar_idx_start = i * vec_size;
    size_t n = scalar_idx_start / S;
    size_t s = scalar_idx_start % S;
    size_t idx = n * stride + offset + s;

    VecT dy_vec = *reinterpret_cast<const VecT*>(&grad_output[idx]);
    const T* dy_arr = reinterpret_cast<const T*>(&dy_vec);

#pragma unroll
    for (int k = 0; k < vec_size; ++k) {
      float dy = static_cast<float>(dy_arr[k]);
      float x_hat = normalized_input[idx + k];
      sum_dy += dy;
      sum_dy_x_norm += dy * x_hat;
    }
  }

  sum_dy = blockReduceSum(sum_dy);
  sum_dy_x_norm = blockReduceSum(sum_dy_x_norm);

  if (threadIdx.x == 0) {
    d_gamma[c] = sum_dy_x_norm;
    d_beta[c] = sum_dy;
  }
}

template <typename T>
__global__ void batchnorm_dgrad_kernel(const T* __restrict__ grad_output,
                                       const float* __restrict__ normalized_input,
                                       const float* __restrict__ inv_std,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ d_gamma,
                                       const float* __restrict__ d_beta, T* __restrict__ grad_input,
                                       size_t N, size_t C, size_t S, bool affine) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_elements = N * C * S;

  if (idx < total_elements) {
    int c = (idx / S) % C;

    float g = (affine && gamma) ? gamma[c] : 1.0f;
    float istd = inv_std[c];

    float sum_dy = d_beta[c];
    float sum_dy_x_norm = d_gamma[c];
    float M = static_cast<float>(N * S);

    float dy = static_cast<float>(grad_output[idx]);
    float x_hat = normalized_input[idx];

    float term1 = (g * istd) / M;
    float term2 = M * dy - sum_dy - (x_hat * sum_dy_x_norm);

    grad_input[idx] = static_cast<T>(term1 * term2);
  }
}

template <typename T>
__global__ void batchnorm_dgrad_kernel_vec(
    const T* __restrict__ grad_output, const float* __restrict__ normalized_input,
    const float* __restrict__ inv_std, const float* __restrict__ gamma,
    const float* __restrict__ d_gamma, const float* __restrict__ d_beta, T* __restrict__ grad_input,
    size_t N, size_t C, size_t S, bool affine) {
  using VecT = typename VectoredTraits<T>::type;
  constexpr int vec_size = VectoredTraits<T>::size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_vectors = (N * C * S) / vec_size;

  if (idx < total_vectors) {
    size_t scalar_idx = idx * vec_size;
    int c = (scalar_idx / S) % C;

    float g = (affine && gamma) ? gamma[c] : 1.0f;
    float istd = inv_std[c];
    float sum_dy = d_beta[c];
    float sum_dy_x_norm = d_gamma[c];
    float M = static_cast<float>(N * S);

    float term1 = (g * istd) / M;

    VecT dy_vec = reinterpret_cast<const VecT*>(grad_output)[idx];
    const T* dy_arr = reinterpret_cast<const T*>(&dy_vec);

    VecT dx_vec;
    T* dx_arr = reinterpret_cast<T*>(&dx_vec);

#pragma unroll
    for (int k = 0; k < vec_size; ++k) {
      float dy = static_cast<float>(dy_arr[k]);
      float x_hat = normalized_input[scalar_idx + k];
      float term2 = M * dy - sum_dy - (x_hat * sum_dy_x_norm);
      dx_arr[k] = static_cast<T>(term1 * term2);
    }

    reinterpret_cast<VecT*>(grad_input)[idx] = dx_vec;
  }
}

template <typename T>
__global__ void batchnorm_nchw_inf_kernel(const T* input, const float* running_mean,
                                          const float* running_var, const float* gamma,
                                          const float* beta, T* output, size_t batch_size,
                                          size_t channels, size_t spatial_size, float epsilon,
                                          bool affine, bool use_relu) {
  extern __shared__ char shared_mem[];
  float* s_mean = reinterpret_cast<float*>(shared_mem);
  float* s_inv_std = s_mean + channels;
  float* s_gamma = s_inv_std + channels;
  float* s_beta = s_gamma + channels;

  for (int c = threadIdx.x; c < channels; c += blockDim.x) {
    s_mean[c] = running_mean[c];
    float var_val = running_var[c];
    s_inv_std[c] = rsqrt(var_val + epsilon);
    if (affine) {
      s_gamma[c] = gamma[c];
      s_beta[c] = beta[c];
    }
  }
  __syncthreads();

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * channels * spatial_size;

  if (idx >= total_elements) return;

  int c = (idx / spatial_size) % channels;

  float input_val = static_cast<float>(input[idx]);
  float normalized_val = (input_val - s_mean[c]) * s_inv_std[c];

  float out_val;
  if (affine) {
    out_val = s_gamma[c] * normalized_val + s_beta[c];
  } else {
    out_val = normalized_val;
  }

  if (use_relu) {
    out_val = out_val > 0.0f ? out_val : 0.0f;
  }

  output[idx] = static_cast<T>(out_val);
}

template <typename T, typename ParamT>
__global__ void layernorm_fwd_kernel(const T* input, T* output, const ParamT* gamma,
                                     const ParamT* beta, size_t channels, T epsilon) {
  size_t n = static_cast<size_t>(blockIdx.x);

  const T* x = input + n * channels;
  T* y = output + n * channels;

  T sum = T(0);
  for (size_t c = 0; c < channels; ++c) {
    sum += x[c];
  }
  const T mean = sum / static_cast<T>(channels);

  T sq_sum = T(0);
  for (size_t c = 0; c < channels; ++c) {
    const T diff = x[c] - mean;
    sq_sum += diff * diff;
  }
  const T var = sq_sum / static_cast<T>(channels);
  const T inv_std = T(1) / static_cast<T>(sqrt(static_cast<double>(var + epsilon)));

  for (size_t c = 0; c < channels; ++c) {
    const T norm = (x[c] - mean) * inv_std;
    const T g = gamma ? static_cast<T>(gamma[c]) : T(1);
    const T b = beta ? static_cast<T>(beta[c]) : T(0);
    y[c] = g * norm + b;
  }
}

template <typename T>
__global__ void layernorm_bwd_kernel(const T* grad_output, const T* input, const T* gamma,
                                     T* grad_input, T* grad_gamma, T* grad_beta, size_t channels,
                                     T epsilon) {
  size_t n = static_cast<size_t>(blockIdx.x);

  const T* x = input + n * channels;
  const T* go = grad_output + n * channels;
  T* gi = grad_input ? (grad_input + n * channels) : nullptr;

  T sum = T(0);
  for (size_t c = 0; c < channels; ++c) {
    sum += x[c];
  }
  const T mean = sum / static_cast<T>(channels);

  T sq_sum = T(0);
  for (size_t c = 0; c < channels; ++c) {
    const T diff = x[c] - mean;
    sq_sum += diff * diff;
  }
  const T var = sq_sum / static_cast<T>(channels);
  const T inv_std = T(1) / static_cast<T>(sqrt(static_cast<double>(var + epsilon)));

  T sum_dl_dnorm = T(0);
  T sum_dl_dnorm_x_hat = T(0);

  for (size_t c = 0; c < channels; ++c) {
    const T g = gamma ? gamma[c] : T(1);
    const T dl_dnorm = go[c] * g;
    const T x_hat = (x[c] - mean) * inv_std;
    sum_dl_dnorm += dl_dnorm;
    sum_dl_dnorm_x_hat += dl_dnorm * x_hat;

    if constexpr (is_floating<T>::value) {
      if (grad_gamma) {
        cuda::gpu_atomic_add(&grad_gamma[c], go[c] * x_hat);
      }
      if (grad_beta) {
        cuda::gpu_atomic_add(&grad_beta[c], go[c]);
      }
    }
  }

  if (gi) {
    const T mean_dl_dnorm = sum_dl_dnorm / static_cast<T>(channels);
    const T mean_dl_dnorm_x_hat = sum_dl_dnorm_x_hat / static_cast<T>(channels);

    for (size_t c = 0; c < channels; ++c) {
      const T g = gamma ? gamma[c] : T(1);
      const T dl_dnorm = go[c] * g;
      const T x_hat = (x[c] - mean) * inv_std;
      gi[c] = inv_std * (dl_dnorm - mean_dl_dnorm - x_hat * mean_dl_dnorm_x_hat);
    }
  }
}

WorkspaceReq CUDAEngine::query_batchnorm_graph(engine_handle backend_handle,
                                               const BatchNormStats& stats, DTypeDesc type_desc) {
  size_t temp = 2 * stats.channels * get_dtype_size(type_desc.param_dtype);
  return {0, temp, 0};
}

WorkspaceReq CUDAEngine::query_layernorm_graph(engine_handle backend_handle,
                                               const LayerNormStats& stats, DTypeDesc type_desc) {
  size_t temp = 2 * stats.channels * get_dtype_size(type_desc.param_dtype);
  return {0, temp, 0};
}

void CUDAEngine::batchnorm_fwd(engine_handle backend_handle, const BatchNormStats& stats,
                               const void* input, const void* gamma, const void* beta, void* output,
                               void* prev_running_mean, void* prev_running_var,
                               void* next_running_mean, void* next_running_var, void* batch_mean,
                               void* batch_invar, void* relu_mask, void* workspace,
                               DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    constexpr int vec_size = VectoredTraits<T>::size;
    if ((stats.height * stats.width) % vec_size == 0) {
      batchnorm_stats_kernel_vec<<<stats.channels, 256, 0, stream>>>(
          static_cast<const T*>(input), static_cast<float*>(batch_mean),
          static_cast<float*>(batch_invar), static_cast<float*>(next_running_mean),
          static_cast<float*>(next_running_var), stats.batch_size, stats.channels,
          (stats.height * stats.width), stats.momentum, stats.epsilon);
      size_t total_elements = stats.batch_size * stats.channels * (stats.height * stats.width);
      size_t total_vectors = total_elements / vec_size;
      int num_blocks = (total_vectors + 256 - 1) / 256;
      batchnorm_fwd_kernel_vec<<<num_blocks, 256, 0, stream>>>(
          static_cast<const T*>(input), static_cast<float*>(batch_mean),
          static_cast<float*>(batch_invar), static_cast<const float*>(gamma),
          static_cast<const float*>(beta), static_cast<T*>(output), static_cast<float*>(workspace),
          stats.batch_size, stats.channels, (stats.height * stats.width), true);
    } else {
      batchnorm_stats_kernel<<<stats.channels, 256, 0, stream>>>(
          static_cast<const T*>(input), static_cast<float*>(batch_mean),
          static_cast<float*>(batch_invar), static_cast<float*>(next_running_mean),
          static_cast<float*>(next_running_var), stats.batch_size, stats.channels,
          (stats.height * stats.width), stats.momentum, stats.epsilon);
      size_t total_elements = stats.batch_size * stats.channels * (stats.height * stats.width);
      int num_blocks = (total_elements + 256 - 1) / 256;
      batchnorm_fwd_kernel<<<num_blocks, 256, 0, stream>>>(
          static_cast<const T*>(input), static_cast<float*>(batch_mean),
          static_cast<float*>(batch_invar), static_cast<const float*>(gamma),
          static_cast<const float*>(beta), static_cast<T*>(output), static_cast<float*>(workspace),
          stats.batch_size, stats.channels, (stats.height * stats.width), true);
    }
  });
}

void CUDAEngine::batchnorm_infer(engine_handle backend_handle, const BatchNormStats& stats,
                                 const void* input, const void* gamma, const void* beta,
                                 const void* saved_mean, const void* saved_var, void* output,
                                 void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    size_t total_elements = stats.batch_size * stats.channels * (stats.height * stats.width);
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    size_t shared_mem_size = 4 * stats.channels * sizeof(float);
    batchnorm_nchw_inf_kernel<<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
        static_cast<const T*>(input), static_cast<const float*>(saved_mean),
        static_cast<const float*>(saved_var), static_cast<const float*>(gamma),
        static_cast<const float*>(beta), static_cast<T*>(output), stats.batch_size, stats.channels,
        (stats.height * stats.width), stats.epsilon, true, stats.use_relu);
  });
}

void CUDAEngine::batchnorm_bwd(engine_handle backend_handle, const BatchNormStats& stats,
                               const void* grad_output, const void* input, const void* relu_mask,
                               const void* gamma, void* grad_input, void* grad_gamma,
                               void* grad_beta, const void* batch_mean, const void* batch_invar,
                               void* workspace, DTypeDesc type_desc) {
  size_t grad_gamma_temp_size = stats.channels * get_dtype_size(type_desc.param_dtype);
  size_t grad_beta_temp_size = stats.channels * get_dtype_size(type_desc.param_dtype);
  void* grad_gamma_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_gamma_temp_size;
  void* grad_beta_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_beta_temp_size;

  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    constexpr int vec_size = VectoredTraits<T>::size;
    if ((stats.height * stats.width) % vec_size == 0) {
      batchnorm_wgrad_bgrad_reduce_kernel_vec<<<stats.channels, 256, 0, stream>>>(
          static_cast<const T*>(grad_output), static_cast<const float*>(workspace),
          static_cast<float*>(grad_gamma_temp), static_cast<float*>(grad_beta_temp),
          stats.batch_size, stats.channels, (stats.height * stats.width));
      size_t total_elements = stats.batch_size * stats.channels * (stats.height * stats.width);
      size_t total_vectors = total_elements / vec_size;
      int num_blocks = (total_vectors + 256 - 1) / 256;
      batchnorm_dgrad_kernel_vec<<<num_blocks, 256, 0, stream>>>(
          static_cast<const T*>(grad_output), static_cast<const float*>(workspace),
          static_cast<const float*>(batch_invar), static_cast<const float*>(gamma),
          static_cast<float*>(grad_gamma_temp), static_cast<float*>(grad_beta_temp),
          static_cast<T*>(grad_input), stats.batch_size, stats.channels,
          (stats.height * stats.width), true);
    } else {
      batchnorm_wgrad_bgrad_reduce_kernel<<<stats.channels, 256, 0, stream>>>(
          static_cast<const T*>(grad_output), static_cast<const float*>(workspace),
          static_cast<float*>(grad_gamma_temp), static_cast<float*>(grad_beta_temp),
          stats.batch_size, stats.channels, (stats.height * stats.width));
      size_t total_elements = stats.batch_size * stats.channels * (stats.height * stats.width);
      int num_blocks = (total_elements + 256 - 1) / 256;
      batchnorm_dgrad_kernel<<<num_blocks, 256, 0, stream>>>(
          static_cast<const T*>(grad_output), static_cast<const float*>(workspace),
          static_cast<const float*>(batch_invar), static_cast<const float*>(gamma),
          static_cast<float*>(grad_gamma_temp), static_cast<float*>(grad_beta_temp),
          static_cast<T*>(grad_input), stats.batch_size, stats.channels,
          (stats.height * stats.width), true);
    }

    if (true) {
      int total_size = 1 * stats.channels;
      int num_blocks = (total_size + 256 - 1) / 256;
      norm_add_bias_kernel<float, float, float><<<num_blocks, 256, 0, stream>>>(
          static_cast<float*>(grad_gamma), static_cast<const float*>(grad_gamma_temp), 1,
          stats.channels);
      norm_add_bias_kernel<float, float, float><<<num_blocks, 256, 0, stream>>>(
          static_cast<float*>(grad_beta), static_cast<const float*>(grad_beta_temp), 1,
          stats.channels);
    }
  });
}

void CUDAEngine::layernorm_fwd(engine_handle backend_handle, const LayerNormStats& stats,
                               const void* input, const void* gamma, const void* beta, void* output,
                               void* mean, void* inv_variance, void* workspace,
                               DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    if (stats.batch_size == 0 || stats.channels == 0) return;
    dim3 blocks(static_cast<unsigned int>(stats.batch_size));
    dim3 threads(1);
    layernorm_fwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<T*>(output), static_cast<const T*>(gamma),
        static_cast<const T*>(beta), stats.channels, static_cast<T>(stats.epsilon));
  });
}

void CUDAEngine::layernorm_infer(engine_handle backend_handle, const LayerNormStats& stats,
                                 const void* input, const void* gamma, const void* beta,
                                 void* output, void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    if (stats.batch_size == 0 || stats.channels == 0) return;
    dim3 blocks(static_cast<unsigned int>(stats.batch_size));
    dim3 threads(1);
    layernorm_fwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<T*>(output), static_cast<const T*>(gamma),
        static_cast<const T*>(beta), stats.channels, static_cast<T>(stats.epsilon));
  });
}

void CUDAEngine::layernorm_bwd(engine_handle backend_handle, const LayerNormStats& stats,
                               const void* grad_output, const void* input, const void* gamma,
                               const void* mean, const void* inv_variance, void* grad_input,
                               void* grad_gamma_prev, void* grad_beta_prev, void* workspace,
                               DTypeDesc type_desc) {
  size_t grad_gamma_temp_size = stats.channels * get_dtype_size(type_desc.param_dtype);
  size_t grad_beta_temp_size = stats.channels * get_dtype_size(type_desc.param_dtype);
  void* grad_gamma_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_gamma_temp_size;
  void* grad_beta_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_beta_temp_size;

  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    if (stats.batch_size == 0 || stats.channels == 0) return;
    dim3 blocks(static_cast<unsigned int>(stats.batch_size));
    dim3 threads(1);
    layernorm_bwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(grad_output), static_cast<const T*>(input),
        static_cast<const T*>(gamma), static_cast<T*>(grad_input), static_cast<T*>(grad_gamma_temp),
        static_cast<T*>(grad_beta_temp), stats.channels, static_cast<T>(stats.epsilon));

    if (true) {
      int total_size = 1 * stats.channels;
      int num_blocks = (total_size + 256 - 1) / 256;
      norm_add_bias_kernel<T, T, float><<<num_blocks, 256, 0, stream>>>(
          static_cast<T*>(grad_gamma_prev), static_cast<const T*>(grad_gamma_temp), 1,
          stats.channels);
      norm_add_bias_kernel<T, T, float><<<num_blocks, 256, 0, stream>>>(
          static_cast<T*>(grad_beta_prev), static_cast<const T*>(grad_beta_temp), 1,
          stats.channels);
    }
  });
}

}  // namespace tunx

#endif
