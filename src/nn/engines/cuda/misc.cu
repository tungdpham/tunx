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
#include "cuda/helpers.cuh"

namespace tunx {
using namespace cuda;

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

template <typename IO_T>
__global__ void dropout_fwd_kernel_vectorized(const IO_T* __restrict__ input,
                                              IO_T* __restrict__ output, bool* __restrict__ mask,
                                              size_t n_elements, IO_T dropout_rate, IO_T scale,
                                              unsigned long long seed) {
  using VecT = typename VectoredTraits<IO_T>::type;
  constexpr int vec_width = VectoredTraits<IO_T>::size;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  const VecT* input_vec = reinterpret_cast<const VecT*>(input);
  VecT* output_vec = reinterpret_cast<VecT*>(output);

  size_t n_vectors = n_elements / vec_width;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, 0, &state);

  for (size_t i = idx; i < n_vectors; i += stride) {
    VecT in_val = input_vec[i];
    VecT out_val;
    const IO_T* in_arr = reinterpret_cast<const IO_T*>(&in_val);
    IO_T* out_arr = reinterpret_cast<IO_T*>(&out_val);

    float4 rand_vals = curand_uniform4(&state);
    float rands[4] = {rand_vals.x, rand_vals.y, rand_vals.z, rand_vals.w};

    size_t base = i * vec_width;

#pragma unroll
    for (int k = 0; k < vec_width; ++k) {
      if (rands[k] < static_cast<float>(dropout_rate)) {
        mask[base + k] = false;
        out_arr[k] = IO_T(0);
      } else {
        mask[base + k] = true;
        out_arr[k] = in_arr[k] * scale;
      }
    }

    output_vec[i] = out_val;
  }

  size_t tail_start = n_vectors * vec_width;
  for (size_t i = tail_start + idx; i < n_elements; i += stride) {
    float r = curand_uniform(&state);
    if (r < static_cast<float>(dropout_rate)) {
      mask[i] = false;
      output[i] = IO_T(0);
    } else {
      mask[i] = true;
      output[i] = input[i] * scale;
    }
  }
}

template <typename IO_T>
__global__ void dropout_fwd_kernel(const IO_T* input, IO_T* output, bool* mask, size_t batch_size,
                                   size_t channels, size_t spatial_size, IO_T dropout_rate,
                                   IO_T scale, unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_elements = batch_size * channels * spatial_size;
  int stride = blockDim.x * gridDim.x;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, 0, &state);

  for (size_t i = idx; i < total_elements; i += stride) {
    float rand_val = curand_uniform(&state);

    if (rand_val < static_cast<float>(dropout_rate)) {
      mask[i] = false;
      output[i] = IO_T(0);
    } else {
      mask[i] = true;
      output[i] = input[i] * scale;
    }
  }
}

template <typename IO_T>
__global__ void dropout_dgrad_kernel(const IO_T* __restrict__ grad_output,
                                     IO_T* __restrict__ grad_input, const bool* __restrict__ mask,
                                     size_t total_elements, IO_T scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (size_t i = idx; i < total_elements; i += stride) {
    grad_input[i] = mask[i] ? grad_output[i] * scale : IO_T(0);
  }
}

template <typename T>
__inline__ __device__ T warpReduceMax(T val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    T other = __shfl_down_sync(0xffffffff, val, offset);
    val = val > other ? val : other;
  }
  return val;
}

template <typename T>
__inline__ __device__ T blockReduceMax(T val) {
  static __shared__ T shared[WARP_SIZE];
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  val = warpReduceMax(val);

  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : lowest_value<T>();
  if (wid == 0) {
    val = warpReduceMax(val);
  }

  return val;
}

template <typename T>
__global__ void sdpa_compute_scores_kernel(
    const T* __restrict__ q, const T* __restrict__ k,
    typename TypeTraits<T>::ComputePrecision* __restrict__ scores, size_t batch_heads,
    size_t seq_len, size_t head_dim, float scale, bool is_causal) {
  using AccT = typename TypeTraits<T>::ComputePrecision;

  size_t row = blockIdx.x;
  size_t total_rows = batch_heads * seq_len;
  if (row >= total_rows) {
    return;
  }

  size_t batch_head = row / seq_len;
  size_t query_idx = row % seq_len;
  size_t data_base = batch_head * seq_len * head_dim;
  const T* q_row = q + data_base + query_idx * head_dim;
  AccT* score_row = scores + row * seq_len;

  for (size_t key_idx = threadIdx.x; key_idx < seq_len; key_idx += blockDim.x) {
    if (is_causal && key_idx > query_idx) {
      score_row[key_idx] = lowest_value<AccT>();
      continue;
    }

    const T* k_row = k + data_base + key_idx * head_dim;
    AccT dot = AccT(0);
    for (size_t dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
      dot += static_cast<AccT>(q_row[dim_idx]) * static_cast<AccT>(k_row[dim_idx]);
    }
    score_row[key_idx] = dot * static_cast<AccT>(scale);
  }
}

template <typename T>
__global__ void sdpa_softmax_kernel(
    const typename TypeTraits<T>::ComputePrecision* __restrict__ scores,
    T* __restrict__ attn_weights, size_t total_rows, size_t seq_len) {
  using AccT = typename TypeTraits<T>::ComputePrecision;

  size_t row = blockIdx.x;
  if (row >= total_rows) {
    return;
  }

  const AccT* score_row = scores + row * seq_len;
  T* attn_row = attn_weights + row * seq_len;

  AccT local_max = lowest_value<AccT>();
  for (size_t idx = threadIdx.x; idx < seq_len; idx += blockDim.x) {
    AccT value = score_row[idx];
    local_max = local_max > value ? local_max : value;
  }
  AccT row_max = blockReduceMax(local_max);

  __shared__ AccT shared_max;
  if (threadIdx.x == 0) {
    shared_max = row_max;
  }
  __syncthreads();
  row_max = shared_max;

  AccT local_sum = AccT(0);
  for (size_t idx = threadIdx.x; idx < seq_len; idx += blockDim.x) {
    local_sum += device_exp(static_cast<AccT>(score_row[idx] - row_max));
  }
  AccT row_sum = blockReduceSum(local_sum);

  __shared__ AccT shared_sum;
  if (threadIdx.x == 0) {
    shared_sum = row_sum;
  }
  __syncthreads();
  row_sum = shared_sum;

  AccT inv_sum = AccT(1) / (row_sum + static_cast<AccT>(1e-9f));
  for (size_t idx = threadIdx.x; idx < seq_len; idx += blockDim.x) {
    attn_row[idx] =
        static_cast<T>(device_exp(static_cast<AccT>(score_row[idx] - row_max)) * inv_sum);
  }
}

template <typename T>
__global__ void sdpa_output_kernel(const T* __restrict__ attn_weights, const T* __restrict__ v,
                                   T* __restrict__ output, size_t batch_heads, size_t seq_len,
                                   size_t head_dim) {
  using AccT = typename TypeTraits<T>::ComputePrecision;

  size_t row = blockIdx.x;
  size_t total_rows = batch_heads * seq_len;
  if (row >= total_rows) {
    return;
  }

  size_t batch_head = row / seq_len;
  size_t query_idx = row % seq_len;
  size_t data_base = batch_head * seq_len * head_dim;
  const T* attn_row = attn_weights + row * seq_len;

  for (size_t dim_idx = threadIdx.x; dim_idx < head_dim; dim_idx += blockDim.x) {
    AccT value = AccT(0);
    for (size_t key_idx = 0; key_idx < seq_len; ++key_idx) {
      value += static_cast<AccT>(attn_row[key_idx]) *
               static_cast<AccT>(v[data_base + key_idx * head_dim + dim_idx]);
    }
    output[data_base + query_idx * head_dim + dim_idx] = static_cast<T>(value);
  }
}

template <typename T>
__global__ void sdpa_dgrad_v_kernel(const T* __restrict__ attn_weights,
                                    const T* __restrict__ grad_output, T* __restrict__ grad_v,
                                    size_t batch_heads, size_t seq_len, size_t head_dim) {
  using AccT = typename TypeTraits<T>::ComputePrecision;

  size_t key_row = blockIdx.x;
  size_t total_key_rows = batch_heads * seq_len;
  if (key_row >= total_key_rows) {
    return;
  }

  size_t batch_head = key_row / seq_len;
  size_t key_idx = key_row % seq_len;
  size_t attn_base = batch_head * seq_len * seq_len;
  size_t data_base = batch_head * seq_len * head_dim;

  for (size_t dim_idx = threadIdx.x; dim_idx < head_dim; dim_idx += blockDim.x) {
    AccT value = AccT(0);
    for (size_t query_idx = 0; query_idx < seq_len; ++query_idx) {
      value += static_cast<AccT>(attn_weights[attn_base + query_idx * seq_len + key_idx]) *
               static_cast<AccT>(grad_output[data_base + query_idx * head_dim + dim_idx]);
    }
    grad_v[data_base + key_idx * head_dim + dim_idx] = static_cast<T>(value);
  }
}

template <typename T>
__global__ void sdpa_dgrad_attn_kernel(
    const T* __restrict__ grad_output, const T* __restrict__ v,
    typename TypeTraits<T>::ComputePrecision* __restrict__ grad_scores, size_t batch_heads,
    size_t seq_len, size_t head_dim) {
  using AccT = typename TypeTraits<T>::ComputePrecision;

  size_t row = blockIdx.x;
  size_t total_rows = batch_heads * seq_len;
  if (row >= total_rows) {
    return;
  }

  size_t batch_head = row / seq_len;
  size_t query_idx = row % seq_len;
  size_t data_base = batch_head * seq_len * head_dim;
  const T* grad_output_row = grad_output + data_base + query_idx * head_dim;
  AccT* grad_score_row = grad_scores + row * seq_len;

  for (size_t key_idx = threadIdx.x; key_idx < seq_len; key_idx += blockDim.x) {
    const T* v_row = v + data_base + key_idx * head_dim;
    AccT value = AccT(0);
    for (size_t dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
      value += static_cast<AccT>(grad_output_row[dim_idx]) * static_cast<AccT>(v_row[dim_idx]);
    }
    grad_score_row[key_idx] = value;
  }
}

template <typename T>
__global__ void sdpa_softmax_dgrad_kernel(
    const T* __restrict__ attn_weights,
    typename TypeTraits<T>::ComputePrecision* __restrict__ grad_scores, size_t total_rows,
    size_t seq_len) {
  using AccT = typename TypeTraits<T>::ComputePrecision;

  size_t row = blockIdx.x;
  if (row >= total_rows) {
    return;
  }

  const T* attn_row = attn_weights + row * seq_len;
  AccT* grad_score_row = grad_scores + row * seq_len;

  AccT local_dot = AccT(0);
  for (size_t idx = threadIdx.x; idx < seq_len; idx += blockDim.x) {
    local_dot += static_cast<AccT>(attn_row[idx]) * grad_score_row[idx];
  }
  AccT dot = blockReduceSum(local_dot);

  __shared__ AccT shared_dot;
  if (threadIdx.x == 0) {
    shared_dot = dot;
  }
  __syncthreads();
  dot = shared_dot;

  for (size_t idx = threadIdx.x; idx < seq_len; idx += blockDim.x) {
    grad_score_row[idx] = static_cast<AccT>(attn_row[idx]) * (grad_score_row[idx] - dot);
  }
}

template <typename T>
__global__ void sdpa_dgrad_q_kernel(
    const typename TypeTraits<T>::ComputePrecision* __restrict__ grad_scores,
    const T* __restrict__ k, T* __restrict__ grad_q, size_t batch_heads, size_t seq_len,
    size_t head_dim, float attn_scale) {
  using AccT = typename TypeTraits<T>::ComputePrecision;

  size_t row = blockIdx.x;
  size_t total_rows = batch_heads * seq_len;
  if (row >= total_rows) {
    return;
  }

  size_t batch_head = row / seq_len;
  size_t query_idx = row % seq_len;
  size_t data_base = batch_head * seq_len * head_dim;
  const AccT* grad_score_row = grad_scores + row * seq_len;

  for (size_t dim_idx = threadIdx.x; dim_idx < head_dim; dim_idx += blockDim.x) {
    AccT value = AccT(0);
    for (size_t key_idx = 0; key_idx < seq_len; ++key_idx) {
      value +=
          grad_score_row[key_idx] * static_cast<AccT>(k[data_base + key_idx * head_dim + dim_idx]);
    }
    grad_q[data_base + query_idx * head_dim + dim_idx] =
        static_cast<T>(value * static_cast<AccT>(attn_scale));
  }
}

template <typename T>
__global__ void sdpa_dgrad_k_kernel(
    const typename TypeTraits<T>::ComputePrecision* __restrict__ grad_scores,
    const T* __restrict__ q, T* __restrict__ grad_k, size_t batch_heads, size_t seq_len,
    size_t head_dim, float attn_scale) {
  using AccT = typename TypeTraits<T>::ComputePrecision;

  size_t key_row = blockIdx.x;
  size_t total_key_rows = batch_heads * seq_len;
  if (key_row >= total_key_rows) {
    return;
  }

  size_t batch_head = key_row / seq_len;
  size_t key_idx = key_row % seq_len;
  size_t data_base = batch_head * seq_len * head_dim;
  const AccT* grad_score_base = grad_scores + batch_head * seq_len * seq_len;

  for (size_t dim_idx = threadIdx.x; dim_idx < head_dim; dim_idx += blockDim.x) {
    AccT value = AccT(0);
    for (size_t query_idx = 0; query_idx < seq_len; ++query_idx) {
      value += grad_score_base[query_idx * seq_len + key_idx] *
               static_cast<AccT>(q[data_base + query_idx * head_dim + dim_idx]);
    }
    grad_k[data_base + key_idx * head_dim + dim_idx] =
        static_cast<T>(value * static_cast<AccT>(attn_scale));
  }
}

template <typename IO_T>
__global__ void slice_fwd_kernel(const IO_T* input, IO_T* output, size_t outer_size,
                                 size_t inner_size, size_t axis_size, size_t start, size_t length,
                                 size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  size_t i = idx % inner_size;
  size_t tmp = idx / inner_size;
  size_t l = tmp % length;
  size_t o = tmp / length;

  size_t input_idx = o * axis_size * inner_size + (start + l) * inner_size + i;
  output[idx] = input[input_idx];
}

template <typename IO_T>
__global__ void slice_dgrad_kernel(const IO_T* grad_output, IO_T* grad_input, size_t outer_size,
                                   size_t inner_size, size_t axis_size, size_t start, size_t length,
                                   size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  size_t i = idx % inner_size;
  size_t tmp = idx / inner_size;
  size_t l = tmp % length;
  size_t o = tmp / length;

  size_t output_idx = o * axis_size * inner_size + (start + l) * inner_size + i;
  grad_input[output_idx] += grad_output[idx];
}

WorkspaceReq CUDAEngine::query_dropout_graph(engine_handle backend_handle,
                                             const DropoutStats& stats, DTypeDesc type_desc) {
  return {0, 0, 0};
}

void CUDAEngine::dropout_fwd(engine_handle backend_handle, const DropoutStats& stats,
                             const void* input, void* output, bool* mask, void* workspace,
                             DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE(type_desc.io_dtype, IO_T, {
    size_t total_elements = stats.batch_size * stats.channels * stats.spatial_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    blocks = std::min(blocks, 4096);
    IO_T scale = IO_T(1.0f) / (IO_T(1.0f) - static_cast<IO_T>(stats.dropout_rate));
    unsigned long long seed = static_cast<unsigned long long>(clock()) +
                              static_cast<unsigned long long>(std::time(nullptr));

    dropout_fwd_kernel_vectorized<IO_T><<<blocks, threads, 0, stream>>>(
        static_cast<const IO_T*>(input), static_cast<IO_T*>(output), static_cast<bool*>(mask),
        total_elements, static_cast<IO_T>(stats.dropout_rate), scale, seed);
  });
}

void CUDAEngine::dropout_bwd(engine_handle backend_handle, const DropoutStats& stats,
                             const void* grad_output, void* grad_input, const bool* mask,
                             double scale_in, void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE(type_desc.io_dtype, IO_T, {
    size_t total_elements = stats.batch_size * stats.channels * stats.spatial_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    blocks = std::min(blocks, 4096);
    IO_T scale = IO_T(1.0f) / (IO_T(1.0f) - static_cast<IO_T>(stats.dropout_rate));
    dropout_dgrad_kernel<IO_T><<<blocks, threads, 0, stream>>>(
        static_cast<const IO_T*>(grad_output), static_cast<IO_T*>(grad_input),
        static_cast<const bool*>(mask), total_elements, scale);
  });
}

struct CudaTransposeParams {
  size_t ndim;
  size_t dim0;
  size_t dim1;
  size_t shape[8];
  size_t strides[8];
  size_t out_strides[8];
};

template <typename IO_T>
__global__ void transpose_kernel(const IO_T* input, IO_T* output, CudaTransposeParams p,
                                 size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  size_t in_idx = idx;
  size_t out_idx = 0;
  size_t coords[8];

  for (size_t i = 0; i < p.ndim; ++i) {
    coords[i] = in_idx / p.strides[i];
    in_idx %= p.strides[i];
  }

  size_t temp = coords[p.dim0];
  coords[p.dim0] = coords[p.dim1];
  coords[p.dim1] = temp;

  for (size_t i = 0; i < p.ndim; ++i) {
    out_idx += coords[i] * p.out_strides[i];
  }

  output[out_idx] = input[idx];
}

WorkspaceReq CUDAEngine::query_sdpa_graph(engine_handle backend_handle, const AttentionStats& stats,
                                          DTypeDesc type_desc) {
  if (type_desc.io_dtype != DType_t::FP32) {
    throw std::runtime_error("Generic CUDAEngine SDPA only implemented for FP32. Use CuDNNEngine for FP16/BF16.");
  }
  size_t num_elements = stats.batch_size * stats.num_heads * stats.seq_len * stats.seq_len;
  size_t fwd_workspace = num_elements * sizeof(float) + num_elements * sizeof(float);
  size_t bwd_workspace = num_elements * sizeof(float) + num_elements * sizeof(float) + num_elements * sizeof(float);
  return WorkspaceReq{fwd_workspace, bwd_workspace, fwd_workspace};
}

void CUDAEngine::sdpa_fwd(engine_handle backend_handle, const AttentionStats& stats,
                          const void* q_data, const void* k_data, const void* v_data, void* o_data,
                          void* stats_data, void* workspace, DTypeDesc type_desc) {
  if (type_desc.io_dtype != DType_t::FP32) {
    throw std::runtime_error("Generic CUDAEngine SDPA only implemented for FP32.");
  }
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  size_t batch_heads = stats.batch_size * stats.num_heads;
  size_t seq_len = stats.seq_len;
  size_t head_dim = stats.head_dim;
  size_t total_rows = batch_heads * seq_len;
  
  if (total_rows == 0) return;

  float* scores = reinterpret_cast<float*>(workspace);
  float* attn_weights = scores + batch_heads * seq_len * seq_len;
  
  size_t threads = 256;
  size_t blocks = total_rows;
  
  sdpa_compute_scores_kernel<float><<<blocks, threads, 0, stream>>>(
      static_cast<const float*>(q_data), static_cast<const float*>(k_data), scores,
      batch_heads, seq_len, head_dim, stats.attn_scale, stats.is_causal);

  sdpa_softmax_kernel<float><<<blocks, threads, 0, stream>>>(
      scores, attn_weights, total_rows, seq_len);

  sdpa_output_kernel<float><<<blocks, threads, 0, stream>>>(
      attn_weights, static_cast<const float*>(v_data), static_cast<float*>(o_data),
      batch_heads, seq_len, head_dim);
}

void CUDAEngine::sdpa_bwd(engine_handle backend_handle, const AttentionStats& stats,
                          const void* q_data, const void* k_data, const void* v_data,
                          const void* o_data, const void* dO_data, const void* stats_data,
                          void* dQ_data, void* dK_data, void* dV_data, void* workspace,
                          DTypeDesc type_desc) {
  if (type_desc.io_dtype != DType_t::FP32) {
    throw std::runtime_error("Generic CUDAEngine SDPA only implemented for FP32.");
  }
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  size_t batch_heads = stats.batch_size * stats.num_heads;
  size_t seq_len = stats.seq_len;
  size_t head_dim = stats.head_dim;
  size_t total_rows = batch_heads * seq_len;
  
  if (total_rows == 0) return;

  float* scores = reinterpret_cast<float*>(workspace);
  float* attn_weights = scores + batch_heads * seq_len * seq_len;
  float* grad_scores = attn_weights + batch_heads * seq_len * seq_len;
  
  size_t threads = 256;
  size_t blocks = total_rows;
  
  // Recompute scores and attn_weights
  sdpa_compute_scores_kernel<float><<<blocks, threads, 0, stream>>>(
      static_cast<const float*>(q_data), static_cast<const float*>(k_data), scores,
      batch_heads, seq_len, head_dim, stats.attn_scale, stats.is_causal);

  sdpa_softmax_kernel<float><<<blocks, threads, 0, stream>>>(
      scores, attn_weights, total_rows, seq_len);
      
  // Backprop through SDPA
  sdpa_dgrad_v_kernel<float><<<blocks, threads, 0, stream>>>(
      attn_weights, static_cast<const float*>(dO_data), static_cast<float*>(dV_data),
      batch_heads, seq_len, head_dim);
      
  sdpa_dgrad_attn_kernel<float><<<blocks, threads, 0, stream>>>(
      static_cast<const float*>(dO_data), static_cast<const float*>(v_data), grad_scores,
      batch_heads, seq_len, head_dim);
      
  sdpa_softmax_dgrad_kernel<float><<<blocks, threads, 0, stream>>>(
      attn_weights, grad_scores, total_rows, seq_len);
      
  sdpa_dgrad_q_kernel<float><<<blocks, threads, 0, stream>>>(
      grad_scores, static_cast<const float*>(k_data), static_cast<float*>(dQ_data),
      batch_heads, seq_len, head_dim, stats.attn_scale);
      
  sdpa_dgrad_k_kernel<float><<<blocks, threads, 0, stream>>>(
      grad_scores, static_cast<const float*>(q_data), static_cast<float*>(dK_data),
      batch_heads, seq_len, head_dim, stats.attn_scale);
}

WorkspaceReq CUDAEngine::query_transpose_graph(engine_handle backend_handle,
                                               const TransposeStats& stats, DTypeDesc type_desc) {
  return WorkspaceReq{0, 0, 0};
}

void CUDAEngine::transpose(engine_handle backend_handle, const TransposeStats& stats,
                           const void* input, void* output, void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();

  CudaTransposeParams p;
  p.ndim = stats.ndim;
  p.dim0 = stats.dim0;
  p.dim1 = stats.dim1;

  size_t total_elements = 1;
  for (int i = static_cast<int>(p.ndim) - 1; i >= 0; --i) {
    p.shape[i] = stats.shape[i];
    p.strides[i] = total_elements;
    total_elements *= p.shape[i];
  }

  size_t out_shape[8] = {0};
  for (size_t i = 0; i < p.ndim; ++i) out_shape[i] = p.shape[i];
  std::swap(out_shape[p.dim0], out_shape[p.dim1]);

  size_t out_total = 1;
  for (int i = static_cast<int>(p.ndim) - 1; i >= 0; --i) {
    p.out_strides[i] = out_total;
    out_total *= out_shape[i];
  }

  if (total_elements == 0) return;

  size_t threads = 256;
  size_t blocks = (total_elements + threads - 1) / threads;

  DISPATCH_DTYPE(type_desc.io_dtype, IO_T, {
    transpose_kernel<IO_T><<<blocks, threads, 0, stream>>>(
        static_cast<const IO_T*>(input), static_cast<IO_T*>(output), p, total_elements);
  });
}

WorkspaceReq CUDAEngine::query_slice_graph(engine_handle backend_handle, const SliceStats& stats,
                                           DTypeDesc type_desc) {
  return WorkspaceReq{0, 0, 0};
}

void CUDAEngine::slice_fwd(engine_handle backend_handle, const SliceStats& stats, const void* input,
                           void* output, void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  size_t total_elements = stats.outer_size * stats.length * stats.inner_size;
  if (total_elements == 0) return;
  size_t threads = 256;
  size_t blocks = (total_elements + threads - 1) / threads;
  DISPATCH_DTYPE(type_desc.io_dtype, IO_T, {
    slice_fwd_kernel<IO_T><<<blocks, threads, 0, stream>>>(
        static_cast<const IO_T*>(input), static_cast<IO_T*>(output), stats.outer_size,
        stats.inner_size, stats.axis_size, stats.start, stats.length, total_elements);
  });
}

void CUDAEngine::slice_bwd(engine_handle backend_handle, const SliceStats& stats,
                           const void* grad_output, void* grad_input, void* workspace,
                           DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  size_t total_elements = stats.outer_size * stats.length * stats.inner_size;
  if (total_elements == 0) return;
  size_t threads = 256;
  size_t blocks = (total_elements + threads - 1) / threads;
  DISPATCH_DTYPE(type_desc.io_dtype, IO_T, {
    slice_dgrad_kernel<IO_T><<<blocks, threads, 0, stream>>>(
        static_cast<const IO_T*>(grad_output), static_cast<IO_T*>(grad_input), stats.outer_size,
        stats.inner_size, stats.axis_size, stats.start, stats.length, total_elements);
  });
}

}  // namespace tunx

#endif
