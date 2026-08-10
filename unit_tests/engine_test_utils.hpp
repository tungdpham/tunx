#pragma once

#include <fmt/core.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace tunx {

// Math baseline for Dense Forward
template <typename T>
void math_dense_fwd(const T* input, const T* weight, const T* bias, T* output, size_t batch_size,
                    size_t in_features, size_t out_features) {
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t o = 0; o < out_features; ++o) {
      float sum = bias ? static_cast<float>(bias[o]) : 0.0f;
      for (size_t i = 0; i < in_features; ++i) {
        sum += static_cast<float>(input[b * in_features + i]) *
               static_cast<float>(weight[o * in_features + i]);
      }
      output[b * out_features + o] = static_cast<T>(sum);
    }
  }
}

// Math baseline for Dense WGrad
template <typename T>
void math_dense_wgrad(const T* input, const T* grad_output, T* grad_weight, size_t batch_size,
                      size_t in_features, size_t out_features) {
  for (size_t o = 0; o < out_features; ++o) {
    for (size_t i = 0; i < in_features; ++i) {
      float sum = 0.0f;
      for (size_t b = 0; b < batch_size; ++b) {
        sum += static_cast<float>(grad_output[b * out_features + o]) *
               static_cast<float>(input[b * in_features + i]);
      }
      grad_weight[o * in_features + i] = static_cast<T>(sum);
    }
  }
}

// Math baseline for Dense DGrad
template <typename T>
void math_dense_dgrad(const T* grad_output, const T* weight, T* grad_input, size_t batch_size,
                      size_t in_features, size_t out_features) {
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t i = 0; i < in_features; ++i) {
      float sum = 0.0f;
      for (size_t o = 0; o < out_features; ++o) {
        sum += static_cast<float>(grad_output[b * out_features + o]) *
               static_cast<float>(weight[o * in_features + i]);
      }
      grad_input[b * in_features + i] = static_cast<T>(sum);
    }
  }
}

// Math baseline for Dense BGrad
template <typename T>
void math_dense_bgrad(const T* grad_output, T* grad_bias, size_t batch_size, size_t out_features) {
  for (size_t o = 0; o < out_features; ++o) {
    float sum = 0.0f;
    for (size_t b = 0; b < batch_size; ++b) {
      sum += static_cast<float>(grad_output[b * out_features + o]);
    }
    grad_bias[o] = static_cast<T>(sum);
  }
}

}  // namespace tunx

// Math baseline for ReLU Fwd
template <typename T>
void math_relu_fwd(const T* input_data, T* output_data, bool* mask_data, size_t num_elements) {
  T zero = static_cast<T>(0);
  for (size_t i = 0; i < num_elements; ++i) {
    bool is_positive = input_data[i] > zero;
    output_data[i] = is_positive ? input_data[i] : zero;
    if (mask_data) mask_data[i] = is_positive;
  }
}

// Math baseline for ReLU Bwd
template <typename T>
void math_relu_bwd(const T* grad_output_data, T* grad_input_data, const bool* mask_data,
                   size_t num_elements) {
  for (size_t i = 0; i < num_elements; ++i) {
    grad_input_data[i] = grad_output_data[i] * static_cast<T>(mask_data[i]);
  }
}

// Math baseline for Dropout Fwd
template <typename T>
void math_dropout_fwd(const T* input_data, T* output_data, const bool* mask_data, size_t batch_size,
                      size_t channels, size_t spatial_size, T dropout_rate) {
  T scale = static_cast<T>(1.0 / (1.0 - static_cast<double>(dropout_rate)));
  size_t total_elements = batch_size * channels * spatial_size;
  for (size_t i = 0; i < total_elements; ++i) {
    output_data[i] = mask_data[i] ? input_data[i] * scale : T(0);
  }
}

// Math baseline for Dropout Bwd
template <typename T>
void math_dropout_bwd(const T* grad_output_data, T* grad_input_data, const bool* mask_data,
                      size_t batch_size, size_t channels, size_t spatial_size, T scale) {
  for (size_t n = 0; n < batch_size; ++n) {
    for (size_t c = 0; c < channels; ++c) {
      size_t offset = (n * channels + c) * spatial_size;
      for (size_t i = 0; i < spatial_size; ++i) {
        grad_input_data[offset + i] =
            mask_data[offset + i] ? grad_output_data[offset + i] * scale : T(0);
      }
    }
  }
}

// Math baseline for Embedding Fwd
template <typename T_IO, typename T_PARAM>
void math_embedding_fwd(const T_IO* input_data, const T_PARAM* weight_data, T_PARAM* output_data,
                        size_t num_indices, size_t vocab_size, size_t embed_dim,
                        size_t padding_idx) {
  for (size_t i = 0; i < num_indices; ++i) {
    size_t idx = static_cast<size_t>(input_data[i]);
    if (idx >= vocab_size) idx = 0;
    T_PARAM* out_row = output_data + i * embed_dim;
    if (padding_idx < vocab_size && idx == padding_idx) {
      for (size_t j = 0; j < embed_dim; ++j) out_row[j] = T_PARAM(0);
    } else {
      const T_PARAM* w_row = weight_data + idx * embed_dim;
      for (size_t j = 0; j < embed_dim; ++j) out_row[j] = w_row[j];
    }
  }
}

// Math baseline for Embedding Bwd
template <typename T_IO, typename T_PARAM>
void math_embedding_bwd(const T_IO* input_data, const T_PARAM* gradient_data,
                        T_PARAM* grad_weight_data, size_t num_indices, size_t vocab_size,
                        size_t embed_dim, size_t padding_idx) {
  for (size_t i = 0; i < num_indices; ++i) {
    size_t idx = static_cast<size_t>(input_data[i]);
    if (idx >= vocab_size) idx = 0;
    if (padding_idx < vocab_size && idx == padding_idx) continue;
    for (size_t j = 0; j < embed_dim; ++j) {
      grad_weight_data[idx * embed_dim + j] += gradient_data[i * embed_dim + j];
    }
  }
}

// Math baseline for ClassToken Fwd
template <typename T>
void math_class_token_fwd(const T* input, const T* token, T* output, size_t batch_size,
                          size_t seq_len, size_t embed_dim) {
  size_t S = seq_len;
  size_t E = embed_dim;
  size_t output_S = S + 1;
  for (size_t n = 0; n < batch_size; ++n) {
    T* out_seq = output + n * output_S * E;
    const T* in_seq = input + n * S * E;
    for (size_t j = 0; j < E; ++j) out_seq[j] = token[j];
    for (size_t i = 0; i < S * E; ++i) out_seq[E + i] = in_seq[i];
  }
}

// Math baseline for ClassToken Bwd
template <typename T>
void math_class_token_bwd(const T* grad_output, T* grad_input, T* grad_token, size_t batch_size,
                          size_t seq_len, size_t embed_dim) {
  size_t S = seq_len;
  size_t E = embed_dim;
  size_t output_S = S + 1;
  for (size_t n = 0; n < batch_size; ++n) {
    const T* grad_out_seq = grad_output + n * output_S * E;
    T* grad_in_seq = grad_input + n * S * E;
    for (size_t i = 0; i < S * E; ++i) grad_in_seq[i] = grad_out_seq[E + i];
    for (size_t j = 0; j < E; ++j) grad_token[j] += grad_out_seq[j];
  }
}

template <typename T>
void math_avgpool_fwd(const T* input, T* output, size_t batch_size, size_t height, size_t width,
                      size_t channels, size_t pool_h, size_t pool_w, size_t stride_h,
                      size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h,
                      size_t output_w) {
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t oh = 0; oh < output_h; ++oh) {
      for (size_t ow = 0; ow < output_w; ++ow) {
        for (size_t c = 0; c < channels; ++c) {
          float sum = 0.0f;
          int count = 0;
          int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
          int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
          int h_end = std::min(h_start + static_cast<int>(pool_h), static_cast<int>(height));
          int w_end = std::min(w_start + static_cast<int>(pool_w), static_cast<int>(width));
          h_start = std::max(h_start, 0);
          w_start = std::max(w_start, 0);
          for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
              size_t input_idx = ((b * height + h) * width + w) * channels + c;
              sum += static_cast<float>(input[input_idx]);
              ++count;
            }
          }
          size_t output_idx = ((b * output_h + oh) * output_w + ow) * channels + c;
          output[output_idx] = static_cast<T>(count > 0 ? sum / count : 0.0f);
        }
      }
    }
  }
}

template <typename T>
void math_avgpool_bwd(const T* grad_output, T* grad_input, size_t batch_size, size_t input_h,
                      size_t input_w, size_t channels, size_t pool_h, size_t pool_w,
                      size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h,
                      size_t output_w) {
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t oh = 0; oh < output_h; ++oh) {
      for (size_t ow = 0; ow < output_w; ++ow) {
        for (size_t c = 0; c < channels; ++c) {
          size_t output_idx = ((b * output_h + oh) * output_w + ow) * channels + c;
          float grad = static_cast<float>(grad_output[output_idx]);
          int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
          int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
          int h_end = std::min(h_start + static_cast<int>(pool_h), static_cast<int>(input_h));
          int w_end = std::min(w_start + static_cast<int>(pool_w), static_cast<int>(input_w));
          h_start = std::max(h_start, 0);
          w_start = std::max(w_start, 0);
          int count = (h_end - h_start) * (w_end - w_start);
          if (count == 0) continue;
          float grad_per_element = grad / count;
          for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
              size_t input_idx = ((b * input_h + h) * input_w + w) * channels + c;
              grad_input[input_idx] =
                  static_cast<T>(static_cast<float>(grad_input[input_idx]) + grad_per_element);
            }
          }
        }
      }
    }
  }
}

template <typename T>
void math_maxpool2d_fwd(const T* input, T* output, int32_t* mask, size_t batch_size, size_t input_h,
                        size_t input_w, size_t channels, size_t pool_h, size_t pool_w,
                        size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                        size_t output_h, size_t output_w) {
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t oh = 0; oh < output_h; ++oh) {
      for (size_t ow = 0; ow < output_w; ++ow) {
        for (size_t c = 0; c < channels; ++c) {
          int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
          int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
          int h_end = std::min(h_start + static_cast<int>(pool_h), static_cast<int>(input_h));
          int w_end = std::min(w_start + static_cast<int>(pool_w), static_cast<int>(input_w));
          h_start = std::max(h_start, 0);
          w_start = std::max(w_start, 0);

          float max_val = -INFINITY;
          int max_idx = -1;
          for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
              size_t input_idx = ((b * input_h + h) * input_w + w) * channels + c;
              float val = static_cast<float>(input[input_idx]);
              if (val > max_val) {
                max_val = val;

                int h_start_unclamped = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
                int w_start_unclamped = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
                int rel_h = h - h_start_unclamped;
                int rel_w = w - w_start_unclamped;
                max_idx = rel_h * static_cast<int>(pool_w) + rel_w;
              }
            }
          }
          size_t output_idx = ((b * output_h + oh) * output_w + ow) * channels + c;
          output[output_idx] = static_cast<T>(max_val);
          if (mask) mask[output_idx] = max_idx;
        }
      }
    }
  }
}

template <typename T>
void math_maxpool2d_bwd(const T* grad_output, T* grad_input, const int32_t* mask, size_t batch_size,
                        size_t channels, size_t output_h, size_t output_w, size_t input_h,
                        size_t input_w, size_t pool_w, size_t stride_h, size_t stride_w,
                        size_t pad_h, size_t pad_w) {
  for (size_t b = 0; b < batch_size; ++b) {
    size_t total_outputs_per_batch = output_h * output_w * channels;
    for (size_t i = 0; i < total_outputs_per_batch; ++i) {
      size_t global_i = b * total_outputs_per_batch + i;
      int rel_idx = mask[global_i];
      if (rel_idx >= 0) {
        size_t c = i % channels;
        size_t ow = (i / channels) % output_w;
        size_t oh = (i / (channels * output_w)) % output_h;

        int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
        int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);

        int rel_h = rel_idx / static_cast<int>(pool_w);
        int rel_w = rel_idx % static_cast<int>(pool_w);

        int h = h_start + rel_h;
        int w = w_start + rel_w;

        if (h >= 0 && h < static_cast<int>(input_h) && w >= 0 && w < static_cast<int>(input_w)) {
          size_t in_idx = ((b * input_h + h) * input_w + w) * channels + c;
          grad_input[in_idx] = static_cast<T>(static_cast<float>(grad_input[in_idx]) +
                                              static_cast<float>(grad_output[global_i]));
        }
      }
    }
  }
}

template <typename T>
void math_layernorm_bwd(const T* grad_output, const T* input, const T* gamma, T* grad_input,
                        T* grad_gamma, T* grad_beta, size_t batch_size, size_t channels,
                        T epsilon) {
  size_t batch_stride = channels;
  std::vector<T> means(batch_size);
  std::vector<T> inv_stds(batch_size);

  for (size_t n = 0; n < batch_size; ++n) {
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
  }

  for (size_t n = 0; n < batch_size; ++n) {
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
      T g = gamma ? gamma[c] : T(1);
      T dx_hat = go * g;
      sum_grad_normalized += dx_hat * normalized;
      sum_grad_gamma_normalized += dx_hat;
    }
    T factor = inv_std / static_cast<T>(channels);
    for (size_t c = 0; c < channels; ++c) {
      size_t idx = base_idx + c;
      T val = input[idx];
      T normalized = (val - mean) * inv_std;
      T g = gamma ? gamma[c] : T(1);
      T go = grad_output[idx];
      T dx_hat = go * g;
      grad_input[idx] = factor * (static_cast<T>(channels) * dx_hat - sum_grad_gamma_normalized -
                                  normalized * sum_grad_normalized);
    }
  }

  if (grad_gamma || grad_beta) {
    for (size_t c = 0; c < channels; ++c) {
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
      if (grad_gamma) grad_gamma[c] += dgamma;
      if (grad_beta) grad_beta[c] += dbeta;
    }
  }
}

template <typename T>
void math_batchnorm_bwd(const T* grad_output, const T* input, const float* mean,
                        const float* inv_std, const float* gamma, float* d_gamma, float* d_beta,
                        T* grad_input, const bool* relu_mask, size_t N, size_t C, size_t S,
                        bool affine, bool use_relu) {
  size_t M = N * S;
  float inv_M = 1.0f / static_cast<float>(M);
  std::vector<float> partial_dy(N * C, 0.0f);
  std::vector<float> partial_dy_xn(N * C, 0.0f);
  for (size_t n = 0; n < N; ++n) {
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
  }
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
  for (size_t i = 0; i < M; ++i) {
    for (size_t c = 0; c < C; ++c) {
      size_t idx = i * C + c;
      float g = gamma[c];
      float term1 = (g * inv_std[c]) * inv_M;
      float dy = (!use_relu || (relu_mask && relu_mask[idx])) ? static_cast<float>(grad_output[idx])
                                                              : 0.0f;
      float x_hat = (static_cast<float>(input[idx]) - mean[c]) * inv_std[c];
      float term2 = static_cast<float>(M) * dy - sum_dy[c] - x_hat * sum_dy_xnorm[c];
      grad_input[idx] = static_cast<T>(term1 * term2);
    }
  }
}

template <typename T>
void math_conv2d_wgrad_naive(const T* grad_output, const T* input, T* grad_weight,
                             size_t batch_size, size_t input_h, size_t input_w, size_t in_channels,
                             size_t out_channels, size_t kernel_h, size_t kernel_w, size_t stride_h,
                             size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h,
                             size_t output_w) {
  for (size_t oc = 0; oc < out_channels; ++oc) {
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t oh = 0; oh < output_h; ++oh) {
        for (size_t ow = 0; ow < output_w; ++ow) {
          size_t o_idx = ((b * output_h + oh) * output_w + ow) * out_channels + oc;
          float go = static_cast<float>(grad_output[o_idx]);
          for (size_t kh = 0; kh < kernel_h; ++kh) {
            int ih = static_cast<int>(oh * stride_h + kh) - static_cast<int>(pad_h);
            if (ih >= 0 && ih < static_cast<int>(input_h)) {
              for (size_t kw = 0; kw < kernel_w; ++kw) {
                int iw = static_cast<int>(ow * stride_w + kw) - static_cast<int>(pad_w);
                if (iw >= 0 && iw < static_cast<int>(input_w)) {
                  for (size_t ic = 0; ic < in_channels; ++ic) {
                    size_t w_idx = ((oc * kernel_h + kh) * kernel_w + kw) * in_channels + ic;
                    size_t i_idx = ((b * input_h + ih) * input_w + iw) * in_channels + ic;
                    grad_weight[w_idx] = static_cast<T>(static_cast<float>(grad_weight[w_idx]) +
                                                        go * static_cast<float>(input[i_idx]));
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

template <typename T>
void math_conv2d_bgrad_naive(const T* grad_output, T* grad_bias, size_t batch_size,
                             size_t out_channels, size_t output_h, size_t output_w) {
  for (size_t oc = 0; oc < out_channels; ++oc) {
    float sum = 0.0f;
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t oh = 0; oh < output_h; ++oh) {
        for (size_t ow = 0; ow < output_w; ++ow) {
          size_t o_idx = ((b * output_h + oh) * output_w + ow) * out_channels + oc;
          sum += static_cast<float>(grad_output[o_idx]);
        }
      }
    }
    grad_bias[oc] += static_cast<T>(sum);
  }
}

template <typename T>
void math_transpose(const T* input, T* output, const size_t* shape, size_t ndim, size_t dim0, size_t dim1) {
    size_t total_elements = 1;
    std::vector<size_t> strides(ndim);
    for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
        strides[i] = total_elements;
        total_elements *= shape[i];
    }
    std::vector<size_t> out_shape(shape, shape + ndim);
    std::swap(out_shape[dim0], out_shape[dim1]);
    
    std::vector<size_t> out_strides(ndim);
    size_t out_total = 1;
    for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
        out_strides[i] = out_total;
        out_total *= out_shape[i];
    }
    
    for (size_t i = 0; i < total_elements; ++i) {
        size_t linear_idx = i;
        size_t out_idx = 0;
        for (size_t d = 0; d < ndim; ++d) {
            size_t coord = linear_idx / strides[d];
            linear_idx %= strides[d];
            
            size_t out_d = d;
            if (d == dim0) out_d = dim1;
            else if (d == dim1) out_d = dim0;
            
            out_idx += coord * out_strides[out_d];
        }
        output[out_idx] = input[i];
    }
}

template <typename T>
void math_sdpa_fwd(const T* q, const T* k, const T* v, T* o, T* stats, size_t batch_size,
                   size_t num_heads, size_t seq_len, size_t head_dim, float attn_scale,
                   bool is_causal) {
    size_t B = batch_size, H = num_heads, S = seq_len, D = head_dim;
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t s_q = 0; s_q < S; ++s_q) {
                std::vector<float> scores(S, -INFINITY);
                float max_score = -INFINITY;
                for (size_t s_k = 0; s_k < S; ++s_k) {
                    if (is_causal && s_k > s_q) continue;
                    float dot = 0.0f;
                    for (size_t d = 0; d < D; ++d) {
                        float q_val = static_cast<float>(q[b * H * S * D + h * S * D + s_q * D + d]);
                        float k_val = static_cast<float>(k[b * H * S * D + h * S * D + s_k * D + d]);
                        dot += q_val * k_val;
                    }
                    dot *= attn_scale;
                    scores[s_k] = dot;
                    if (dot > max_score) max_score = dot;
                }
                
                float sum_exp = 0.0f;
                for (size_t s_k = 0; s_k < S; ++s_k) {
                    if (is_causal && s_k > s_q) continue;
                    scores[s_k] = std::exp(scores[s_k] - max_score);
                    sum_exp += scores[s_k];
                }
                
                for (size_t d = 0; d < D; ++d) {
                    float out_val = 0.0f;
                    for (size_t s_k = 0; s_k < S; ++s_k) {
                        if (is_causal && s_k > s_q) continue;
                        float v_val = static_cast<float>(v[b * H * S * D + h * S * D + s_k * D + d]);
                        out_val += (scores[s_k] / sum_exp) * v_val;
                    }
                    o[b * H * S * D + h * S * D + s_q * D + d] = static_cast<T>(out_val);
                }
                if (stats) {
                    stats[b * H * S + h * S + s_q] = static_cast<T>(std::log(sum_exp) + max_score);
                }
            }
        }
    }
}

template <typename T>
void math_sdpa_bwd(const T* q, const T* k, const T* v, const T* o, const T* do_, T* dq, T* dk, T* dv,
                   size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim,
                   float attn_scale, bool is_causal) {
    size_t B = batch_size, H = num_heads, S = seq_len, D = head_dim;
    size_t total_elements = B * H * S * D;
    for (size_t i = 0; i < total_elements; ++i) {
        dq[i] = 0; dk[i] = 0; dv[i] = 0;
    }

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t s_q = 0; s_q < S; ++s_q) {
                std::vector<float> scores(S, -INFINITY);
                float max_score = -INFINITY;
                for (size_t s_k = 0; s_k < S; ++s_k) {
                    if (is_causal && s_k > s_q) continue;
                    float dot = 0.0f;
                    for (size_t d = 0; d < D; ++d) {
                        float q_val = static_cast<float>(q[b * H * S * D + h * S * D + s_q * D + d]);
                        float k_val = static_cast<float>(k[b * H * S * D + h * S * D + s_k * D + d]);
                        dot += q_val * k_val;
                    }
                    dot *= attn_scale;
                    scores[s_k] = dot;
                    if (dot > max_score) max_score = dot;
                }
                
                float sum_exp = 0.0f;
                std::vector<float> probs(S, 0.0f);
                for (size_t s_k = 0; s_k < S; ++s_k) {
                    if (is_causal && s_k > s_q) continue;
                    probs[s_k] = std::exp(scores[s_k] - max_score);
                    sum_exp += probs[s_k];
                }
                for (size_t s_k = 0; s_k < S; ++s_k) {
                    if (is_causal && s_k > s_q) continue;
                    probs[s_k] /= sum_exp;
                }

                std::vector<float> dS(S, 0.0f);
                for (size_t s_k = 0; s_k < S; ++s_k) {
                    if (is_causal && s_k > s_q) continue;
                    float ds_val = 0.0f;
                    for (size_t d = 0; d < D; ++d) {
                        float do_val = static_cast<float>(do_[b * H * S * D + h * S * D + s_q * D + d]);
                        float v_val = static_cast<float>(v[b * H * S * D + h * S * D + s_k * D + d]);
                        ds_val += do_val * v_val;
                        
                        float dv_val = static_cast<float>(dv[b * H * S * D + h * S * D + s_k * D + d]);
                        dv_val += probs[s_k] * do_val;
                        dv[b * H * S * D + h * S * D + s_k * D + d] = static_cast<T>(dv_val);
                    }
                    dS[s_k] = ds_val;
                }

                float dS_dot_probs = 0.0f;
                for (size_t s_k = 0; s_k < S; ++s_k) {
                    if (is_causal && s_k > s_q) continue;
                    dS_dot_probs += dS[s_k] * probs[s_k];
                }

                std::vector<float> dP(S, 0.0f);
                for (size_t s_k = 0; s_k < S; ++s_k) {
                    if (is_causal && s_k > s_q) continue;
                    dP[s_k] = probs[s_k] * (dS[s_k] - dS_dot_probs);
                }

                for (size_t d = 0; d < D; ++d) {
                    float dq_val = 0.0f;
                    float q_val = static_cast<float>(q[b * H * S * D + h * S * D + s_q * D + d]);
                    for (size_t s_k = 0; s_k < S; ++s_k) {
                        if (is_causal && s_k > s_q) continue;
                        float k_val = static_cast<float>(k[b * H * S * D + h * S * D + s_k * D + d]);
                        float dp_scaled = dP[s_k] * attn_scale;
                        dq_val += dp_scaled * k_val;
                        
                        float dk_val = static_cast<float>(dk[b * H * S * D + h * S * D + s_k * D + d]);
                        dk_val += dp_scaled * q_val;
                        dk[b * H * S * D + h * S * D + s_k * D + d] = static_cast<T>(dk_val);
                    }
                    float old_dq = static_cast<float>(dq[b * H * S * D + h * S * D + s_q * D + d]);
                    dq[b * H * S * D + h * S * D + s_q * D + d] = static_cast<T>(old_dq + dq_val);
                }
            }
        }
    }
}
