#include "internal.hpp"
#include "nn/engines/cpu_engine.hpp"
#include "threading/thread_handler.hpp"

namespace tunx {
namespace {

// Helpers for class token
template <typename T>
void class_token_fwd_impl(const T* input, const T* token, T* output, size_t batch_size,
                          size_t seq_len, size_t embed_dim) {
  size_t S = seq_len;
  size_t E = embed_dim;
  size_t output_S = S + 1;
  parallel_for<size_t>(0, batch_size, [&](size_t n) {
    T* out_seq = output + n * output_S * E;
    const T* in_seq = input + n * S * E;
    std::memcpy(out_seq, token, E * sizeof(T));
    std::memcpy(out_seq + E, in_seq, S * E * sizeof(T));
  });
}

template <typename T>
void class_token_bwd_impl(const T* grad_output, T* grad_input, T* grad_token, size_t batch_size,
                          size_t seq_len, size_t embed_dim) {
  size_t S = seq_len;
  size_t E = embed_dim;
  size_t output_S = S + 1;
  parallel_for<size_t>(0, batch_size, [&](size_t n) {
    const T* grad_out_seq = grad_output + n * output_S * E;
    T* grad_in_seq = grad_input + n * S * E;
    std::memcpy(grad_in_seq, grad_out_seq + E, S * E * sizeof(T));
  });
  parallel_for<size_t>(0, E, [&](size_t e) {
    T sum = 0;
    for (size_t n = 0; n < batch_size; ++n) {
      const T* grad_out_seq = grad_output + n * output_S * E;
      sum += grad_out_seq[e];
    }
    grad_token[e] += sum;
  });
}

template <typename INDEX_T, typename IO_T, typename PARAM_T>
void embedding_fwd_impl(const INDEX_T* input_data, const PARAM_T* weight_data, IO_T* output_data,
                        size_t num_indices, size_t vocab_size, size_t embed_dim,
                        size_t padding_idx) {
  parallel_for<size_t>(0, num_indices, [&](size_t i) {
    size_t idx = static_cast<size_t>(input_data[i]);
    if (idx >= vocab_size) idx = 0;
    IO_T* out_row = output_data + i * embed_dim;
    if (padding_idx < vocab_size && idx == padding_idx) {
      std::fill(out_row, out_row + embed_dim, IO_T(0));
    } else {
      const PARAM_T* w_row = weight_data + idx * embed_dim;
      for (size_t j = 0; j < embed_dim; ++j) {
        out_row[j] = static_cast<IO_T>(w_row[j]);
      }
    }
  });
}

template <typename INDEX_T, typename IO_T, typename PARAM_T>
void embedding_bwd_impl(const INDEX_T* input_data, const IO_T* gradient_data,
                        PARAM_T* grad_weight_data, size_t num_indices, size_t vocab_size,
                        size_t embed_dim, size_t padding_idx) {
  parallel_for<size_t>(0, embed_dim, [&](size_t j) {
    for (size_t i = 0; i < num_indices; ++i) {
      size_t idx = static_cast<size_t>(input_data[i]);
      if (idx >= vocab_size) idx = 0;
      if (padding_idx < vocab_size && idx == padding_idx) continue;
      grad_weight_data[idx * embed_dim + j] += static_cast<PARAM_T>(gradient_data[i * embed_dim + j]);
    }
  });
}

template <typename T_IO, typename T_PARAM, typename T_COMPUTE>
void pos_embedding_fwd_impl(const T_IO* input, const T_PARAM* pos_embed, T_IO* output,
                            size_t batch_size, size_t seq_len, size_t embed_dim) {
  size_t sample_size = seq_len * embed_dim;
  parallel_for<size_t>(0, batch_size, [&](size_t b) {
    const T_IO* in_b = input + b * sample_size;
    T_IO* out_b = output + b * sample_size;
    for (size_t s = 0; s < seq_len; ++s) {
      for (size_t e = 0; e < embed_dim; ++e) {
        size_t idx = s * embed_dim + e;
        out_b[idx] = static_cast<T_IO>(static_cast<T_COMPUTE>(in_b[idx]) +
                                       static_cast<T_COMPUTE>(pos_embed[idx]));
      }
    }
  });
}

template <typename T_IO, typename T_PARAM, typename T_COMPUTE>
void pos_embedding_bwd_impl(const T_IO* grad_output, T_PARAM* grad_pos_embed, size_t batch_size,
                            size_t seq_len, size_t embed_dim) {
  size_t sample_size = seq_len * embed_dim;
  parallel_for<size_t>(0, sample_size, [&](size_t i) {
    T_COMPUTE sum = 0;
    for (size_t b = 0; b < batch_size; ++b) {
      sum += static_cast<T_COMPUTE>(grad_output[b * sample_size + i]);
    }
    grad_pos_embed[i] = static_cast<T_PARAM>(static_cast<T_COMPUTE>(grad_pos_embed[i]) + sum);
  });
}

}  // namespace

void CPUEngine::class_token_fwd(engine_handle backend_handle, const ClassTokenStats& stats,
                                const void* input, const void* token, void* output, void* workspace,
                                DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    class_token_fwd_impl<T>(static_cast<const T*>(input), static_cast<const T*>(token),
                            static_cast<T*>(output), stats.batch_size, stats.seq_len,
                            stats.embed_dim);
  });
}

void CPUEngine::class_token_bwd(engine_handle backend_handle, const ClassTokenStats& stats,
                                const void* grad_output, void* grad_input, void* grad_token,
                                void* workspace, DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    class_token_bwd_impl<T>(static_cast<const T*>(grad_output), static_cast<T*>(grad_input),
                            static_cast<T*>(grad_token), stats.batch_size, stats.seq_len,
                            stats.embed_dim);
  });
}

void CPUEngine::embedding_fwd(engine_handle backend_handle, const EmbeddingStats& stats,
                              const void* input, const void* weight, void* output, void* workspace,
                              DTypeDesc type_desc) {
  DISPATCH_ANY_DTYPE2(type_desc.io_dtype, type_desc.param_dtype, T_IO, T_PARAM, {
    embedding_fwd_impl<int32_t, T_IO, T_PARAM>(static_cast<const int32_t*>(input),
                                               static_cast<const T_PARAM*>(weight),
                                               static_cast<T_IO*>(output), stats.num_indices,
                                               stats.vocab_size, stats.embed_dim, stats.padding_idx);
  });
}

void CPUEngine::embedding_bwd(engine_handle backend_handle, const EmbeddingStats& stats,
                              const void* grad_output, const void* input, void* grad_weight,
                              void* workspace, DTypeDesc type_desc) {
  DISPATCH_ANY_DTYPE2(type_desc.io_dtype, type_desc.param_dtype, T_IO, T_PARAM, {
    embedding_bwd_impl<int32_t, T_IO, T_PARAM>(static_cast<const int32_t*>(input),
                                               static_cast<const T_IO*>(grad_output),
                                               static_cast<T_PARAM*>(grad_weight), stats.num_indices,
                                               stats.vocab_size, stats.embed_dim, stats.padding_idx);
  });
}

void CPUEngine::positional_embedding_fwd(engine_handle backend_handle,
                                         const PositionalEmbeddingStats& stats, const void* input,
                                         const void* pos_embedding, void* output, void* workspace,
                                         DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_ANY_DTYPE(type_desc.io_dtype, T, {
    pos_embedding_fwd_impl<T, T, T>(static_cast<const T*>(input),
                                    static_cast<const T*>(pos_embedding), static_cast<T*>(output),
                                    stats.batch_size, stats.seq_len, stats.embed_dim);
  });
}

void CPUEngine::positional_embedding_bwd(engine_handle backend_handle,
                                         const PositionalEmbeddingStats& stats,
                                         const void* grad_output, void* grad_pos_embedding,
                                         void* workspace, DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_ANY_DTYPE(type_desc.io_dtype, T, {
    pos_embedding_bwd_impl<T, T, T>(static_cast<const T*>(grad_output),
                                    static_cast<T*>(grad_pos_embedding), stats.batch_size,
                                    stats.seq_len, stats.embed_dim);
  });
}

WorkspaceReq CPUEngine::query_positional_embedding_graph(engine_handle backend_handle,
                                                         const PositionalEmbeddingStats&,
                                                         DTypeDesc) {
  return {0, 0, 0};
}

WorkspaceReq CPUEngine::query_class_token_graph(engine_handle backend_handle,
                                                const ClassTokenStats& stats, DTypeDesc type_desc) {
  return {0, 0, 0};
}

WorkspaceReq CPUEngine::query_embedding_graph(engine_handle backend_handle,
                                              const EmbeddingStats& stats, DTypeDesc type_desc) {
  return {0, 0, 0};
}

}  // namespace tunx
