#ifdef TUNX_USE_CUDNN
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn_frontend.h>
#include <cudnn_frontend/graph_interface.h>
#include <cudnn_frontend/graph_properties.h>
#include <cudnn_frontend_utils.h>
#include <cudnn_graph.h>
#include <fmt/core.h>

#include "cuda/helpers.cuh"
#include "nn/engines/cudnn_engine.hpp"
#include "nn/engines/engine_handle.hpp"
#include "nn/stats/stats.hpp"
#include "type/type.hpp"

namespace tunx {

template <typename T>
__global__ void class_token_fwd_kernel(const T* __restrict__ input, const T* __restrict__ token,
                                       T* __restrict__ output, size_t seq_len, size_t embed_dim,
                                       size_t total_elements) {
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
__global__ void class_token_bwd_kernel(const T* __restrict__ grad_output,
                                       T* __restrict__ grad_input, T* __restrict__ grad_token,
                                       size_t batch_size, size_t seq_len, size_t embed_dim) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t output_seq_len = seq_len + 1;
  size_t total_elements = batch_size * output_seq_len * embed_dim;

  if (idx >= total_elements) return;

  size_t e = idx % embed_dim;
  size_t tmp = idx / embed_dim;
  size_t s_out = tmp % output_seq_len;
  size_t n = tmp / output_seq_len;

  if (s_out == 0) {
    tunx::cuda::gpu_atomic_add(&grad_token[e], grad_output[idx]);
  } else {
    size_t s_in = s_out - 1;
    size_t in_idx = n * seq_len * embed_dim + s_in * embed_dim + e;
    grad_input[in_idx] = grad_output[idx];
  }
}

template <typename T>
__global__ void embedding_fwd_kernel(const T* __restrict__ input, const T* __restrict__ weight,
                                     T* __restrict__ output, size_t num_indices, size_t vocab_size,
                                     size_t embed_dim, size_t padding_idx) {
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

template <typename T>
__global__ void embedding_bwd_kernel(const T* __restrict__ input, const T* __restrict__ grad_output,
                                     T* __restrict__ grad_weight, size_t num_indices,
                                     size_t vocab_size, size_t embed_dim, size_t padding_idx) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_indices * embed_dim) return;

  size_t token_idx = tid / embed_dim;
  size_t dim_idx = tid % embed_dim;

  size_t vocab_idx = static_cast<size_t>(input[token_idx]);
  if (vocab_idx >= vocab_size) vocab_idx = 0;

  if (padding_idx < vocab_size && vocab_idx == padding_idx) return;

  T g_val = grad_output[tid];
  tunx::cuda::gpu_atomic_add(&grad_weight[vocab_idx * embed_dim + dim_idx], g_val);
}

WorkspaceReq CuDNNEngine::query_class_token_graph(engine_handle backend_handle,
                                                  const ClassTokenStats& stats,
                                                  DTypeDesc type_desc) {
  size_t temp = stats.embed_dim * get_dtype_size(type_desc.param_dtype);
  return {0, temp, 0};
}

WorkspaceReq CuDNNEngine::query_embedding_graph(engine_handle backend_handle,
                                                const EmbeddingStats& stats, DTypeDesc type_desc) {
  size_t temp = stats.vocab_size * stats.embed_dim * get_dtype_size(type_desc.param_dtype);
  return {0, temp, 0};
}

WorkspaceReq CuDNNEngine::query_positional_embedding_graph(engine_handle backend_handle,
                                                           const PositionalEmbeddingStats& stats,
                                                           DTypeDesc type_desc) {
  return {0, 0, 0};
}

void CuDNNEngine::class_token_fwd(engine_handle backend_handle, const ClassTokenStats& stats,
                                  const void* input, const void* token, void* output,
                                  void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  size_t output_seq_len = stats.seq_len + 1;
  size_t total_elements = stats.batch_size * output_seq_len * stats.embed_dim;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  DISPATCH_DTYPE(type_desc.io_dtype, T, {
    class_token_fwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<const T*>(token), static_cast<T*>(output),
        stats.seq_len, stats.embed_dim, total_elements);
  });
}

void CuDNNEngine::class_token_bwd(engine_handle backend_handle, const ClassTokenStats& stats,
                                  const void* grad_output, void* grad_input, void* grad_token,
                                  void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  size_t output_seq_len = stats.seq_len + 1;
  size_t total_elements = stats.batch_size * output_seq_len * stats.embed_dim;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  DISPATCH_DTYPE(type_desc.io_dtype, T, {
    class_token_bwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(grad_output), static_cast<T*>(grad_input),
        static_cast<T*>(grad_token), stats.batch_size, stats.seq_len, stats.embed_dim);
  });
}

void CuDNNEngine::embedding_fwd(engine_handle backend_handle, const EmbeddingStats& stats,
                                const void* input, const void* weight, void* output,
                                void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  size_t total_elements = stats.num_indices * stats.embed_dim;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  DISPATCH_DTYPE(type_desc.io_dtype, T, {
    embedding_fwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<const T*>(weight), static_cast<T*>(output),
        stats.num_indices, stats.vocab_size, stats.embed_dim, stats.padding_idx);
  });
}

void CuDNNEngine::embedding_bwd(engine_handle backend_handle, const EmbeddingStats& stats,
                                const void* grad_output, const void* input, void* grad_weight,
                                void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  size_t total_elements = stats.num_indices * stats.embed_dim;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  DISPATCH_DTYPE(type_desc.io_dtype, T, {
    embedding_wgrad_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<const T*>(grad_output),
        static_cast<T*>(grad_weight), stats.num_indices, stats.vocab_size, stats.embed_dim,
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

void CuDNNEngine::positional_embedding_fwd(engine_handle backend_handle,
                                           const PositionalEmbeddingStats& stats, const void* input,
                                           const void* pos_embedding, void* output, void* workspace,
                                           DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);
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

void CuDNNEngine::positional_embedding_bwd(engine_handle backend_handle,
                                           const PositionalEmbeddingStats& stats,
                                           const void* grad_output, void* grad_pos_embedding,
                                           void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);
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
