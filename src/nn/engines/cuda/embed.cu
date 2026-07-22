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
#include "type/type.hpp"

namespace tunx {

#define BLOCK_SIZE 256
#define WARP_SIZE 32

template <typename T>
__global__ void class_token_fwd_kernel(const T* input, const T* token, T* output, size_t seq_len,
                                       size_t embed_dim, size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  size_t output_seq_len = seq_len + 1;

  size_t e = idx % embed_dim;
  size_t tmp = idx / embed_dim;
  size_t s_out = tmp % output_seq_len;
  size_t n = tmp / output_seq_len;

  if (s_out == 0) {
    output[idx] = token[e];
  } else {
    size_t s_in = s_out - 1;
    size_t in_idx = n * seq_len * embed_dim + s_in * embed_dim + e;
    output[idx] = input[in_idx];
  }
}

template <typename T>
__global__ void class_token_dgrad_kernel(const T* grad_output, T* grad_input, size_t seq_len,
                                         size_t embed_dim, size_t total_input_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_input_elements) return;

  size_t e = idx % embed_dim;
  size_t tmp = idx / embed_dim;
  size_t s_in = tmp % seq_len;
  size_t n = tmp / seq_len;

  size_t output_seq_len = seq_len + 1;

  size_t out_idx = n * output_seq_len * embed_dim + (s_in + 1) * embed_dim + e;

  grad_input[idx] = grad_output[out_idx];
}

template <typename T>
__global__ void class_token_wgrad_kernel(const T* grad_output, T* grad_token, size_t batch_size,
                                         size_t seq_len, size_t embed_dim) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_items = batch_size * embed_dim;
  if (idx >= total_items) return;

  size_t e = idx % embed_dim;
  size_t n = idx / embed_dim;

  size_t output_seq_len = seq_len + 1;

  size_t out_idx = n * output_seq_len * embed_dim + 0 * embed_dim + e;

  if constexpr (tunx::is_floating<T>::value) {
    cuda::gpu_atomic_add(&grad_token[e], grad_output[out_idx]);
  }
}

template <typename T>
__global__ void embedding_fwd_kernel(const T* input, const T* weight, T* output, size_t num_indices,
                                     size_t vocab_size, size_t embed_dim, size_t padding_idx) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_indices * embed_dim) return;

  size_t token_idx = tid / embed_dim;
  size_t dim_idx = tid % embed_dim;

  size_t vocab_idx = static_cast<size_t>(input[token_idx]);
  if (vocab_idx >= vocab_size) vocab_idx = 0;

  if (padding_idx < vocab_size && vocab_idx == padding_idx) {
    output[tid] = T(0);
    return;
  }

  output[tid] = weight[vocab_idx * embed_dim + dim_idx];
}

template <typename T>
__global__ void embedding_wgrad_kernel(const T* input, const T* grad, T* grad_weight,
                                       size_t num_indices, size_t vocab_size, size_t embed_dim,
                                       size_t padding_idx) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_indices * embed_dim) return;

  size_t token_idx = tid / embed_dim;
  size_t dim_idx = tid % embed_dim;

  size_t vocab_idx = static_cast<size_t>(input[token_idx]);
  if (vocab_idx >= vocab_size) vocab_idx = 0;

  if (padding_idx < vocab_size && vocab_idx == padding_idx) return;

  if constexpr (tunx::is_floating<T>::value) {
    T g_val = grad[tid];
    cuda::gpu_atomic_add(&grad_weight[vocab_idx * embed_dim + dim_idx], g_val);
  }
}

WorkspaceReq CUDAEngine::query_class_token_graph(engine_handle backend_handle,
                                                 const ClassTokenStats& stats,
                                                 DTypeDesc type_desc) {
  return {0, 0, 0};
}

WorkspaceReq CUDAEngine::query_embedding_graph(engine_handle backend_handle,
                                               const EmbeddingStats& stats, DTypeDesc type_desc) {
  return {0, 0, 0};
}

WorkspaceReq CUDAEngine::query_positional_embedding_graph(engine_handle backend_handle,
                                                          const PositionalEmbeddingStats& stats,
                                                          DTypeDesc type_desc) {
  return {0, 0, 0};
}

void CUDAEngine::class_token_fwd(engine_handle backend_handle, const ClassTokenStats& stats,
                                 const void* input, const void* token, void* output,
                                 void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    size_t output_seq_len = stats.seq_len + 1;
    size_t total_elements = stats.batch_size * output_seq_len * stats.embed_dim;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    class_token_fwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<const T*>(token), static_cast<T*>(output),
        stats.seq_len, stats.embed_dim, total_elements);
  });
}

void CUDAEngine::class_token_bwd(engine_handle backend_handle, const ClassTokenStats& stats,
                                 const void* grad_output, void* grad_input, void* grad_token_prev,
                                 void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    size_t total_input = stats.batch_size * stats.seq_len * stats.embed_dim;
    int threads = 256;
    int blocks = (total_input + threads - 1) / threads;
    class_token_dgrad_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(grad_output), static_cast<T*>(grad_input), stats.seq_len,
        stats.embed_dim, total_input);

    size_t total_token_contribs = stats.batch_size * stats.embed_dim;
    blocks = (total_token_contribs + threads - 1) / threads;
    class_token_wgrad_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(grad_output), static_cast<T*>(grad_token_prev), stats.batch_size,
        stats.seq_len, stats.embed_dim);
  });
}

void CUDAEngine::embedding_fwd(engine_handle backend_handle, const EmbeddingStats& stats,
                               const void* input, const void* weight, void* output, void* workspace,
                               DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    size_t total_elements = stats.num_indices * stats.embed_dim;
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    embedding_fwd_kernel<<<numBlocks, blockSize, 0, stream>>>(
        static_cast<const T*>(input), static_cast<const T*>(weight), static_cast<T*>(output),
        stats.num_indices, stats.vocab_size, stats.embed_dim, stats.padding_idx);
  });
}

void CUDAEngine::embedding_bwd(engine_handle backend_handle, const EmbeddingStats& stats,
                               const void* grad_output, const void* input, void* grad_weight_prev,
                               void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    size_t total_elements = stats.num_indices * stats.embed_dim;
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    embedding_wgrad_kernel<<<numBlocks, blockSize, 0, stream>>>(
        static_cast<const T*>(input), static_cast<const T*>(grad_output),
        static_cast<T*>(grad_weight_prev), stats.num_indices, stats.vocab_size, stats.embed_dim,
        stats.padding_idx);
  });
}

template <typename T_IO, typename T_PARAM, typename T_COMPUTE>
__global__ void pos_embedding_fwd_kernel(const T_IO* input, const T_PARAM* pos_embed, T_IO* output,
                                         size_t batch_size, size_t sample_size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = batch_size * sample_size;
  if (idx < total) {
    size_t embed_idx = idx % sample_size;
    output[idx] = static_cast<T_IO>(static_cast<T_COMPUTE>(input[idx]) +
                                    static_cast<T_COMPUTE>(pos_embed[embed_idx]));
  }
}

template <typename T_IO, typename T_PARAM, typename T_COMPUTE>
__global__ void pos_embedding_bwd_kernel(const T_IO* grad_output, T_PARAM* grad_pos_embed,
                                         size_t batch_size, size_t sample_size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < sample_size) {
    float sum = 0.0f;
    for (size_t b = 0; b < batch_size; ++b) {
      sum += static_cast<float>(grad_output[b * sample_size + idx]);
    }
    cuda::gpu_atomic_add(&grad_pos_embed[idx], static_cast<T_PARAM>(sum));
  }
}

void CUDAEngine::positional_embedding_fwd(engine_handle backend_handle,
                                          const PositionalEmbeddingStats& stats, const void* input,
                                          const void* pos_embedding, void* output, void* workspace,
                                          DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE3(
      type_desc.io_dtype, type_desc.param_dtype, type_desc.compute_dtype, T_IO, T_PARAM, T_COMPUTE,
      {
        size_t sample_size = stats.seq_len * stats.embed_dim;
        size_t total_elements = stats.batch_size * sample_size;
        int blockSize = 256;
        int numBlocks = (total_elements + blockSize - 1) / blockSize;
        pos_embedding_fwd_kernel<T_IO, T_PARAM, T_COMPUTE><<<numBlocks, blockSize, 0, stream>>>(
            static_cast<const T_IO*>(input), static_cast<const T_PARAM*>(pos_embedding),
            static_cast<T_IO*>(output), stats.batch_size, sample_size);
      });
}

void CUDAEngine::positional_embedding_bwd(engine_handle backend_handle,
                                          const PositionalEmbeddingStats& stats,
                                          const void* grad_output, void* grad_pos_embedding,
                                          void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE3(
      type_desc.io_dtype, type_desc.param_dtype, type_desc.compute_dtype, T_IO, T_PARAM, T_COMPUTE,
      {
        size_t sample_size = stats.seq_len * stats.embed_dim;
        int blockSize = 256;
        int numBlocks = (sample_size + blockSize - 1) / blockSize;
        pos_embedding_bwd_kernel<T_IO, T_PARAM, T_COMPUTE><<<numBlocks, blockSize, 0, stream>>>(
            static_cast<const T_IO*>(grad_output), static_cast<T_PARAM*>(grad_pos_embedding),
            stats.batch_size, sample_size);
      });
}

}  // namespace tunx

#endif
