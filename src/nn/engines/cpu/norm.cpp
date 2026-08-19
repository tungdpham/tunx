#include "internal.hpp"
#include "nn/engines/cpu_engine.hpp"
#include "threading/thread_handler.hpp"

namespace tunx {
namespace {

template <typename T>
void batchnorm_infer_impl(const T* input, const float* running_mean, const float* running_var,
                          const float* gamma, const float* beta, T* output, size_t N, size_t C,
                          size_t S, float epsilon, bool affine, bool use_relu) {
  std::vector<float> scale(C);
  std::vector<float> bias(C);
  for (size_t c = 0; c < C; ++c) {
    float inv_std = 1.0f / std::sqrt(running_var[c] + epsilon);
    float g = affine ? gamma[c] : 1.0f;
    scale[c] = g * inv_std;
    bias[c] = beta[c] - (running_mean[c] * scale[c]);
  }
  size_t M = N * S;
  parallel_for<size_t>(0, M, [&](size_t i) {
    for (size_t c = 0; c < C; ++c) {
      size_t idx = i * C + c;
      float val = static_cast<float>(input[idx]);
      float out_val = val * scale[c] + bias[c];
      if (use_relu) {
        out_val = std::max(out_val, 0.0f);
      }
      output[idx] = static_cast<T>(out_val);
    }
  });
}

template <typename T, typename ParamT>
void batchnorm_fwd_impl(const T* input, float* mean, float* inv_std, float* running_mean,
                        float* running_var, const ParamT* gamma, const ParamT* beta, T* output,
                        bool* relu_mask, size_t N, size_t C, size_t S, float momentum,
                        float epsilon, bool affine, bool use_relu) {
  size_t M = N * S;
  float inv_M = 1.0f / static_cast<float>(M);
  std::vector<float> partial_sums(N * C, 0.0f);
  std::vector<float> partial_sq_sums(N * C, 0.0f);
  parallel_for<size_t>(0, N, [&](size_t n) {
    for (size_t s = 0; s < S; ++s) {
      for (size_t c = 0; c < C; ++c) {
        size_t idx = n * S * C + s * C + c;
        float val = static_cast<float>(input[idx]);
        partial_sums[n * C + c] += val;
        partial_sq_sums[n * C + c] += val * val;
      }
    }
  });
  std::vector<float> scale(C);
  std::vector<float> bias_term(C);
  for (size_t c = 0; c < C; ++c) {
    float sum = 0.0f;
    float sq_sum = 0.0f;
    for (size_t n = 0; n < N; ++n) {
      sum += partial_sums[n * C + c];
      sq_sum += partial_sq_sums[n * C + c];
    }
    float mu = sum * inv_M;
    mean[c] = mu;
    float var = (sq_sum * inv_M) - (mu * mu);
    var = std::max(var, 0.0f);
    float istd = 1.0f / std::sqrt(var + epsilon);
    inv_std[c] = istd;
    float unbiased_var = (M > 1) ? (var * M) / static_cast<float>(M - 1) : 0.0f;
    running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mu;
    running_var[c] = (1.0f - momentum) * running_var[c] + momentum * unbiased_var;
    float g = affine ? static_cast<float>(gamma[c]) : 1.0f;
    float b = static_cast<float>(beta[c]);
    scale[c] = g * istd;
    bias_term[c] = b - (mu * scale[c]);
  }
  parallel_for<size_t>(0, M, [&](size_t i) {
    for (size_t c = 0; c < C; ++c) {
      size_t idx = i * C + c;
      float val = static_cast<float>(input[idx]) * scale[c] + bias_term[c];
      if (use_relu) {
        bool active = val > 0.0f;
        if (relu_mask) relu_mask[idx] = active;
        output[idx] = val * static_cast<float>(active);
      } else {
        output[idx] = static_cast<T>(val);
      }
    }
  });
}

template <typename T>
void batchnorm_bwd_impl(const T* grad_output, const T* input, const float* mean,
                        const float* inv_std, const float* gamma, float* d_gamma, float* d_beta,
                        T* grad_input, const bool* relu_mask, size_t N, size_t C, size_t S,
                        bool affine, bool use_relu) {
  size_t M = N * S;
  float inv_M = 1.0f / static_cast<float>(M);
  std::vector<float> partial_dy(N * C, 0.0f);
  std::vector<float> partial_dy_xn(N * C, 0.0f);
  parallel_for<size_t>(0, N, [&](size_t n) {
    for (size_t s = 0; s < S; ++s) {
      for (size_t c = 0; c < C; ++c) {
        size_t idx = n * S * C + s * C + c;
        float dy = (!use_relu || (relu_mask && relu_mask[idx]))
                       ? static_cast<float>(grad_output[idx])
                       : 0.0f;
        float x_hat = (static_cast<float>(input[idx]) - mean[c]) * inv_std[c];
        partial_dy[n * C + c] += dy;
        partial_dy_xn[n * C + c] += dy * x_hat;
      }
    }
  });
  std::vector<float> sum_dy(C, 0.0f);
  std::vector<float> sum_dy_xnorm(C, 0.0f);
  for (size_t c = 0; c < C; ++c) {
    float s_dy = 0.0f;
    float s_dy_xn = 0.0f;
    for (size_t n = 0; n < N; ++n) {
      s_dy += partial_dy[n * C + c];
      s_dy_xn += partial_dy_xn[n * C + c];
    }
    sum_dy[c] = s_dy;
    sum_dy_xnorm[c] = s_dy_xn;
    if (affine && d_gamma && d_beta) {
      d_gamma[c] += s_dy_xn;
      d_beta[c] += s_dy;
    }
  }
  parallel_for<size_t>(0, M, [&](size_t i) {
    for (size_t c = 0; c < C; ++c) {
      size_t idx = i * C + c;
      float g = (affine && gamma) ? gamma[c] : 1.0;
      float term1 = (g * inv_std[c]) * inv_M;
      float dy = (!use_relu || (relu_mask && relu_mask[idx])) ? static_cast<float>(grad_output[idx])
                                                              : 0.0f;
      float x_hat = (static_cast<float>(input[idx]) - mean[c]) * inv_std[c];
      float term2 = static_cast<float>(M) * dy - sum_dy[c] - x_hat * sum_dy_xnorm[c];
      grad_input[idx] = static_cast<T>(term1 * term2);
    }
  });
}

template <typename T, typename ParamT>
void layernorm_fwd_impl(const T* input, T* output, const ParamT* gamma, const ParamT* beta,
                        T* mean_out, T* inv_variance_out, size_t batch_size, size_t channels, T epsilon) {
  size_t batch_stride = channels;
  parallel_for<size_t>(0, batch_size, [&](size_t n) {
    T sum = 0;
    T sq_sum = 0;
    size_t base_idx = n * batch_stride;
    for (size_t c = 0; c < channels; ++c) {
      T val = input[base_idx + c];
      sum += val;
    }
    T mean = sum / static_cast<T>(channels);
    if (mean_out) mean_out[n] = mean;
    for (size_t c = 0; c < channels; ++c) {
      T val = input[base_idx + c];
      sq_sum += (val - mean) * (val - mean);
    }
    T var = sq_sum / static_cast<T>(channels);
    T inv_std = T(1) / static_cast<T>(std::sqrt(static_cast<double>(var + epsilon)));
    if (inv_variance_out) inv_variance_out[n] = inv_std;
    for (size_t c = 0; c < channels; ++c) {
      size_t idx = base_idx + c;
      T val = input[idx];
      T normalized = (val - mean) * inv_std;
      T g = gamma ? static_cast<T>(gamma[c]) : T(1);
      T b = beta ? static_cast<T>(beta[c]) : T(0);
      output[idx] = normalized * g + b;
    }
  });
}

template <typename T, typename ParamT>
void layernorm_bwd_impl(const T* grad_output, const T* input, const ParamT* gamma, T* grad_input,
                        ParamT* grad_gamma, ParamT* grad_beta, size_t batch_size, size_t channels,
                        T epsilon) {
  size_t batch_stride = channels;
  std::vector<T> means(batch_size);
  std::vector<T> inv_stds(batch_size);

  parallel_for<size_t>(0, batch_size, [&](size_t n) {
    size_t base_idx = n * batch_stride;
    T sum = 0;
    for (size_t c = 0; c < channels; ++c) sum += input[base_idx + c];
    T mean = sum / static_cast<T>(channels);
    means[n] = mean;
    T sq_sum = 0;
    for (size_t c = 0; c < channels; ++c) {
      T diff = input[base_idx + c] - mean;
      sq_sum += diff * diff;
    }
    inv_stds[n] =
        T(1) /
        static_cast<T>(std::sqrt(static_cast<double>(sq_sum / static_cast<T>(channels) + epsilon)));
  });

  parallel_for<size_t>(0, batch_size, [&](size_t n) {
    size_t base_idx = n * batch_stride;
    T mean = means[n];
    T inv_std = inv_stds[n];
    T sum_grad_normalized = 0;
    T sum_grad_gamma_normalized = 0;
    for (size_t c = 0; c < channels; ++c) {
      size_t idx = base_idx + c;
      T go = grad_output[idx];
      T val = input[idx];
      T normalized = (val - mean) * inv_std;
      T g = gamma ? static_cast<T>(gamma[c]) : T(1);
      T dx_hat = go * g;
      sum_grad_normalized += dx_hat * normalized;
      sum_grad_gamma_normalized += dx_hat;
    }
    T factor = inv_std / static_cast<T>(channels);
    for (size_t c = 0; c < channels; ++c) {
      size_t idx = base_idx + c;
      T val = input[idx];
      T normalized = (val - mean) * inv_std;
      T g = gamma ? static_cast<T>(gamma[c]) : T(1);
      T go = grad_output[idx];
      T dx_hat = go * g;
      grad_input[idx] = factor * (static_cast<T>(channels) * dx_hat - sum_grad_gamma_normalized -
                                  normalized * sum_grad_normalized);
    }
  });

  if (grad_gamma || grad_beta) {
    parallel_for<size_t>(0, channels, [&](size_t c) {
      T dgamma = 0;
      T dbeta = 0;
      for (size_t n = 0; n < batch_size; ++n) {
        size_t idx = n * batch_stride + c;
        T go = grad_output[idx];
        if (grad_gamma) {
          T val = input[idx];
          T normalized = (val - means[n]) * inv_stds[n];
          dgamma += go * normalized;
        }
        if (grad_beta) dbeta += go;
      }
      if (grad_gamma) grad_gamma[c] = static_cast<ParamT>(static_cast<T>(grad_gamma[c]) + dgamma);
      if (grad_beta) grad_beta[c] = static_cast<ParamT>(static_cast<T>(grad_beta[c]) + dbeta);
    });
  }
}

template <typename T, typename ParamT>
void legacy_batchnorm_inf_impl(const T* input_data, const float* running_mean_data,
                               const float* running_var_data, const ParamT* gamma_data,
                               const ParamT* beta_data, T* output_data, size_t batch_size,
                               size_t channels, size_t spatial_size, float epsilon, bool affine) {
  size_t channel_stride = channels * spatial_size;

  parallel_for_2d<size_t>(batch_size, channels, [&](size_t n, size_t c) {
    float mean_val = running_mean_data[c];
    float var_val = running_var_data[c];
    float std_val = std::sqrt(var_val + epsilon);
    const float inv_std = 1.0f / std_val;

    size_t base_idx = n * channel_stride + c * spatial_size;

    const T* input_ptr = input_data + base_idx;
    T* output_ptr = output_data + base_idx;

    if (affine) {
      const float gamma_val = static_cast<float>(gamma_data[c]);
      const float beta_val = static_cast<float>(beta_data[c]);

      for (size_t i = 0; i < spatial_size; ++i) {
        float normalized_val = (static_cast<float>(input_ptr[i]) - mean_val) * inv_std;
        output_ptr[i] = static_cast<T>(gamma_val * normalized_val + beta_val);
      }
    } else {
      for (size_t i = 0; i < spatial_size; ++i) {
        output_ptr[i] = static_cast<T>((static_cast<float>(input_ptr[i]) - mean_val) * inv_std);
      }
    }
  });
}

template <typename T, typename ParamT>
void legacy_batchnorm_fwd_impl(const T* input, float* mean, float* inv_std, float* running_mean,
                               float* running_var, const ParamT* gamma, const ParamT* beta,
                               T* output, float* norm_cache, size_t N, size_t C, size_t S,
                               float momentum, float epsilon, bool affine) {
  size_t total_elements = N * S;
  size_t channel_stride = C * S;
  const float inv_total = 1.0f / static_cast<float>(total_elements);

  parallel_for<size_t>(0, C, [&](size_t c) {
    float sum = 0.0f;
    size_t c_offset = c * S;

    for (size_t n = 0; n < N; ++n) {
      size_t n_offset = n * channel_stride;
      size_t base_idx = n_offset + c_offset;
      const T* input_ptr = input + base_idx;

      for (size_t s = 0; s < S; ++s) {
        sum += static_cast<float>(input_ptr[s]);
      }
    }

    float mu = sum * inv_total;
    mean[c] = mu;

    float var_sum = 0.0f;
    for (size_t n = 0; n < N; ++n) {
      size_t n_offset = n * channel_stride;
      size_t base_idx = n_offset + c_offset;
      const T* input_ptr = input + base_idx;

      for (size_t s = 0; s < S; ++s) {
        float diff = static_cast<float>(input_ptr[s]) - mu;
        var_sum += diff * diff;
      }
    }

    float var = var_sum * inv_total;

    inv_std[c] = 1.0f / std::sqrt(var + epsilon);

    float unbiased_var = var_sum / static_cast<float>(total_elements - 1);

    running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mu;
    running_var[c] = (1.0f - momentum) * running_var[c] + momentum * unbiased_var;
  });

  parallel_for_2d(N, C, [&](size_t n, size_t c) {
    const float mu = mean[c];
    const float istd = inv_std[c];

    size_t n_offset = n * channel_stride;
    size_t c_offset = c * S;
    size_t base_idx = n_offset + c_offset;

    const T* input_ptr = input + base_idx;
    T* output_ptr = output + base_idx;
    float* norm_ptr = norm_cache ? (norm_cache + base_idx) : nullptr;

    for (size_t s = 0; s < S; ++s) {
      float x = static_cast<float>(input_ptr[s]);
      float norm = (x - mu) * istd;

      if (norm_ptr) norm_ptr[s] = norm;

      if (affine) {
        output_ptr[s] =
            static_cast<T>(norm * static_cast<float>(gamma[c]) + static_cast<float>(beta[c]));
      } else {
        output_ptr[s] = static_cast<T>(norm);
      }
    }
  });
}

template <typename T, typename ParamT>
void legacy_batchnorm_bwd_impl(const T* grad_output, const float* norm_input, const float* inv_std,
                               const ParamT* gamma, ParamT* d_gamma, ParamT* d_beta, T* grad_input,
                               size_t N, size_t C, size_t S, bool affine) {
  size_t channel_stride = C * S;
  size_t M = N * S;
  const float inv_M = 1.0f / static_cast<float>(M);

  parallel_for<size_t>(0, C, [&](size_t c) {
    float sum_dy = 0.0f;
    float sum_dy_x_norm = 0.0f;
    size_t c_offset = c * S;

    for (size_t n = 0; n < N; ++n) {
      size_t n_offset = n * channel_stride;
      size_t base_idx = n_offset + c_offset;

      for (size_t s = 0; s < S; ++s) {
        size_t idx = base_idx + s;
        float dy = static_cast<float>(grad_output[idx]);
        float x_hat = norm_input[idx];

        sum_dy += dy;
        sum_dy_x_norm += dy * x_hat;
      }
    }

    if (affine) {
      d_gamma[c] = static_cast<ParamT>(static_cast<float>(d_gamma[c]) + sum_dy_x_norm);
      d_beta[c] = static_cast<ParamT>(static_cast<float>(d_beta[c]) + sum_dy);
    } else {
      d_gamma[c] = static_cast<ParamT>(sum_dy_x_norm);
      d_beta[c] = static_cast<ParamT>(sum_dy);
    }
  });

  parallel_for_2d(N, C, [&](size_t n, size_t c) {
    const float g = (affine && gamma) ? static_cast<float>(gamma[c]) : 1.0f;
    const float istd = inv_std[c];

    const float sum_dy = static_cast<float>(d_beta[c]);
    const float sum_dy_x_norm = static_cast<float>(d_gamma[c]);

    size_t n_offset = n * channel_stride;
    size_t c_offset = c * S;
    size_t base_idx = n_offset + c_offset;

    const float term1 = (g * istd) * inv_M;

    for (size_t s = 0; s < S; ++s) {
      size_t idx = base_idx + s;
      float dy = static_cast<float>(grad_output[idx]);
      float x_hat = norm_input[idx];

      float term2 = static_cast<float>(M) * dy - sum_dy - (x_hat * sum_dy_x_norm);
      grad_input[idx] = static_cast<T>(term1 * term2);
    }
  });
}

}  // namespace

void CPUEngine::batchnorm_fwd(engine_handle backend_handle, const BatchNormStats& stats,
                              const void* input, const void* gamma, const void* beta, void* output,
                              void* prev_running_mean, void* prev_running_var,
                              void* next_running_mean, void* next_running_var, void* batch_mean,
                              void* batch_invar, void* relu_mask, void* workspace,
                              DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    batchnorm_fwd_impl<T>(static_cast<const T*>(input), static_cast<float*>(batch_mean),
                          static_cast<float*>(batch_invar), static_cast<float*>(next_running_mean),
                          static_cast<float*>(next_running_var), static_cast<const float*>(gamma),
                          static_cast<const float*>(beta), static_cast<T*>(output),
                          static_cast<bool*>(relu_mask), stats.batch_size, stats.channels,
                          stats.height * stats.width, stats.momentum, stats.epsilon,
                          gamma != nullptr, stats.use_relu);
  });
}

void CPUEngine::batchnorm_infer(engine_handle backend_handle, const BatchNormStats& stats,
                                const void* input, const void* gamma, const void* beta,
                                const void* saved_mean, const void* saved_var, void* output,
                                void* workspace, DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    batchnorm_infer_impl<T>(static_cast<const T*>(input), static_cast<const float*>(saved_mean),
                            static_cast<const float*>(saved_var), static_cast<const float*>(gamma),
                            static_cast<const float*>(beta), static_cast<T*>(output),
                            stats.batch_size, stats.channels, stats.height * stats.width,
                            stats.epsilon, gamma != nullptr, stats.use_relu);
  });
}

void CPUEngine::batchnorm_bwd(engine_handle backend_handle, const BatchNormStats& stats,
                              const void* grad_output, const void* input, const void* relu_mask,
                              const void* gamma, void* grad_input, void* grad_gamma,
                              void* grad_beta, const void* batch_mean, const void* batch_invar,
                              void* workspace, DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    batchnorm_bwd_impl<T>(static_cast<const T*>(grad_output), static_cast<const T*>(input),
                          static_cast<const float*>(batch_mean),
                          static_cast<const float*>(batch_invar), static_cast<const float*>(gamma),
                          static_cast<float*>(grad_gamma), static_cast<float*>(grad_beta),
                          static_cast<T*>(grad_input), static_cast<const bool*>(relu_mask),
                          stats.batch_size, stats.channels, stats.height * stats.width,
                          gamma != nullptr, stats.use_relu);
  });
}

void CPUEngine::layernorm_fwd(engine_handle backend_handle, const LayerNormStats& stats,
                              const void* input, const void* gamma, const void* beta, void* output,
                              void* mean, void* inv_variance, void* workspace,
                              DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    layernorm_fwd_impl<T>(static_cast<const T*>(input), static_cast<T*>(output),
                          static_cast<const T*>(gamma), static_cast<const T*>(beta),
                          static_cast<T*>(mean), static_cast<T*>(inv_variance),
                          stats.batch_size * stats.seq_len, stats.channels, static_cast<T>(stats.epsilon));
  });
}

void CPUEngine::layernorm_infer(engine_handle backend_handle, const LayerNormStats& stats,
                                const void* input, const void* gamma, const void* beta,
                                void* output, void* workspace, DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    layernorm_fwd_impl<T>(static_cast<const T*>(input), static_cast<T*>(output),
                          static_cast<const T*>(gamma), static_cast<const T*>(beta),
                          nullptr, nullptr,
                          stats.batch_size * stats.seq_len, stats.channels, static_cast<T>(stats.epsilon));
  });
}

void CPUEngine::layernorm_bwd(engine_handle backend_handle, const LayerNormStats& stats,
                              const void* grad_output, const void* input, const void* gamma,
                              const void* mean, const void* inv_variance, void* grad_input,
                              void* grad_gamma, void* grad_beta, void* workspace,
                              DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    layernorm_bwd_impl<T>(static_cast<const T*>(grad_output), static_cast<const T*>(input),
                          static_cast<const T*>(gamma), static_cast<T*>(grad_input),
                          static_cast<T*>(grad_gamma), static_cast<T*>(grad_beta),
                          stats.batch_size * stats.seq_len, stats.channels, static_cast<T>(stats.epsilon));
  });
}

void CPUEngine::legacy_batchnorm_fwd(engine_handle backend_handle, const void* input,
                                     void* batch_mean, void* batch_inv_std, void* running_mean,
                                     void* running_var, const void* gamma, const void* beta,
                                     void* output, void* norm, size_t batch_size, size_t channels,
                                     size_t spatial_size, float momentum, float epsilon,
                                     bool affine, DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    legacy_batchnorm_fwd_impl<T>(
        static_cast<const T*>(input), static_cast<float*>(batch_mean),
        static_cast<float*>(batch_inv_std), static_cast<float*>(running_mean),
        static_cast<float*>(running_var), static_cast<const float*>(gamma),
        static_cast<const float*>(beta), static_cast<T*>(output), static_cast<float*>(norm),
        batch_size, channels, spatial_size, momentum, epsilon, affine);
  });
}

void CPUEngine::legacy_batchnorm_infer(engine_handle backend_handle, const void* input,
                                       const void* running_mean, const void* running_var,
                                       const void* gamma, const void* beta, void* output,
                                       size_t batch_size, size_t channels, size_t spatial_size,
                                       float epsilon, bool affine, DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    legacy_batchnorm_inf_impl<T>(
        static_cast<const T*>(input), static_cast<const float*>(running_mean),
        static_cast<const float*>(running_var), static_cast<const float*>(gamma),
        static_cast<const float*>(beta), static_cast<T*>(output), batch_size, channels,
        spatial_size, epsilon, affine);
  });
}

void CPUEngine::legacy_batchnorm_bwd(engine_handle backend_handle, const void* grad_output,
                                     const void* norm_input, const void* inv_std, const void* gamma,
                                     void* d_gamma, void* d_beta, void* grad_input,
                                     size_t batch_size, size_t channels, size_t spatial_size,
                                     bool affine, DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    legacy_batchnorm_bwd_impl<T>(
        static_cast<const T*>(grad_output), static_cast<const float*>(norm_input),
        static_cast<const float*>(inv_std), static_cast<const float*>(gamma),
        static_cast<float*>(d_gamma), static_cast<float*>(d_beta), static_cast<T*>(grad_input),
        batch_size, channels, spatial_size, affine);
  });
}

WorkspaceReq CPUEngine::query_batchnorm_graph(engine_handle backend_handle,
                                              const BatchNormStats& stats, DTypeDesc type_desc) {
  return {0, 0, 0};
}

WorkspaceReq CPUEngine::query_layernorm_graph(engine_handle backend_handle,
                                              const LayerNormStats& stats, DTypeDesc type_desc) {
  return WorkspaceReq{0, 0, 0};
}

}  // namespace tunx
