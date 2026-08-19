#include "internal.hpp"
#include "math/cpu/gemm.hpp"
#include "nn/engines/cpu_engine.hpp"
#include "threading/thread_handler.hpp"

namespace tunx {
namespace {

template <typename T>
void add_bias_impl(T* output_data, const T* bias_data, size_t batch_size, size_t output_features) {
  parallel_for_2d(batch_size, output_features, [&](size_t n, size_t out_f) {
    output_data[n * output_features + out_f] += bias_data[out_f];
  });
}

// Naive NHWC Conv2D
template <typename T>
void conv2d_fwd_naive_impl(const T* input, const T* weight, const T* bias, T* output,
                           size_t batch_size, size_t input_h, size_t input_w, size_t in_channels,
                           size_t out_channels, size_t kernel_h, size_t kernel_w, size_t stride_h,
                           size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h,
                           size_t output_w) {
  parallel_for_2d(batch_size, output_h, [&](size_t b, size_t oh) {
    for (size_t ow = 0; ow < output_w; ++ow) {
      for (size_t oc = 0; oc < out_channels; ++oc) {
        float sum = bias ? static_cast<float>(bias[oc]) : 0.0f;
        for (size_t kh = 0; kh < kernel_h; ++kh) {
          int ih = static_cast<int>(oh * stride_h + kh) - static_cast<int>(pad_h);
          if (ih >= 0 && ih < static_cast<int>(input_h)) {
            for (size_t kw = 0; kw < kernel_w; ++kw) {
              int iw = static_cast<int>(ow * stride_w + kw) - static_cast<int>(pad_w);
              if (iw >= 0 && iw < static_cast<int>(input_w)) {
                for (size_t ic = 0; ic < in_channels; ++ic) {
                  size_t i_idx = ((b * input_h + ih) * input_w + iw) * in_channels + ic;
                  size_t w_idx = ((oc * kernel_h + kh) * kernel_w + kw) * in_channels + ic;
                  sum += static_cast<float>(input[i_idx]) * static_cast<float>(weight[w_idx]);
                }
              }
            }
          }
        }
        size_t o_idx = ((b * output_h + oh) * output_w + ow) * out_channels + oc;
        output[o_idx] = static_cast<T>(sum);
      }
    }
  });
}

template <typename T>
void conv2d_dgrad_naive_impl(const T* grad_output, const T* weight, T* grad_input,
                             size_t batch_size, size_t input_h, size_t input_w, size_t in_channels,
                             size_t out_channels, size_t kernel_h, size_t kernel_w, size_t stride_h,
                             size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h,
                             size_t output_w) {
  size_t num_elements = batch_size * input_h * input_w * in_channels;
  parallel_for<size_t>(0, num_elements, [&](size_t i) { grad_input[i] = T(0); });
  parallel_for<size_t>(0, batch_size, [&](size_t b) {
    for (size_t oh = 0; oh < output_h; ++oh) {
      for (size_t ow = 0; ow < output_w; ++ow) {
        for (size_t oc = 0; oc < out_channels; ++oc) {
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
                    grad_input[i_idx] = static_cast<T>(static_cast<float>(grad_input[i_idx]) +
                                                       go * static_cast<float>(weight[w_idx]));
                  }
                }
              }
            }
          }
        }
      }
    }
  });
}

template <typename T>
void conv2d_wgrad_naive_impl(const T* grad_output, const T* input, T* grad_weight,
                             size_t batch_size, size_t input_h, size_t input_w, size_t in_channels,
                             size_t out_channels, size_t kernel_h, size_t kernel_w, size_t stride_h,
                             size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h,
                             size_t output_w) {
  parallel_for<size_t>(0, out_channels, [&](size_t oc) {
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
  });
}

template <typename T>
void conv2d_bgrad_naive_impl(const T* grad_output, T* grad_bias, size_t batch_size,
                             size_t out_channels, size_t output_h, size_t output_w) {
  parallel_for<size_t>(0, out_channels, [&](size_t oc) {
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
  });
}

}  // namespace

void CPUEngine::conv2d_fwd(engine_handle backend_handle, const Conv2DStats& stats,
                           const void* input, const void* weight, const void* bias, void* output,
                           void* workspace, DTypeDesc type_desc) {
  size_t output_h = (stats.input_h + 2 * stats.pad_h - stats.kernel_h) / stats.stride_h + 1;
  size_t output_w = (stats.input_w + 2 * stats.pad_w - stats.kernel_w) / stats.stride_w + 1;
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    conv2d_fwd_naive_impl<T>(static_cast<const T*>(input), static_cast<const T*>(weight),
                             static_cast<const T*>(bias), static_cast<T*>(output), stats.batch_size,
                             stats.input_h, stats.input_w, stats.in_channels, stats.out_channels,
                             stats.kernel_h, stats.kernel_w, stats.stride_h, stats.stride_w,
                             stats.pad_h, stats.pad_w, output_h, output_w);
  });
}

void CPUEngine::conv2d_dgrad(engine_handle backend_handle, const Conv2DStats& stats,
                             const void* grad_output, const void* weight, void* grad_input,
                             void* workspace, DTypeDesc type_desc) {
  size_t output_h = (stats.input_h + 2 * stats.pad_h - stats.kernel_h) / stats.stride_h + 1;
  size_t output_w = (stats.input_w + 2 * stats.pad_w - stats.kernel_w) / stats.stride_w + 1;
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    conv2d_dgrad_naive_impl<T>(static_cast<const T*>(grad_output), static_cast<const T*>(weight),
                               static_cast<T*>(grad_input), stats.batch_size, stats.input_h,
                               stats.input_w, stats.in_channels, stats.out_channels, stats.kernel_h,
                               stats.kernel_w, stats.stride_h, stats.stride_w, stats.pad_h,
                               stats.pad_w, output_h, output_w);
  });
}

void CPUEngine::conv2d_wgrad(engine_handle backend_handle, const Conv2DStats& stats,
                             const void* grad_output, const void* input, void* grad_weight,
                             void* workspace, DTypeDesc type_desc) {
  size_t output_h = (stats.input_h + 2 * stats.pad_h - stats.kernel_h) / stats.stride_h + 1;
  size_t output_w = (stats.input_w + 2 * stats.pad_w - stats.kernel_w) / stats.stride_w + 1;
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    conv2d_wgrad_naive_impl<T>(static_cast<const T*>(grad_output), static_cast<const T*>(input),
                               static_cast<T*>(grad_weight), stats.batch_size, stats.input_h,
                               stats.input_w, stats.in_channels, stats.out_channels, stats.kernel_h,
                               stats.kernel_w, stats.stride_h, stats.stride_w, stats.pad_h,
                               stats.pad_w, output_h, output_w);
  });
}

void CPUEngine::conv2d_bgrad(engine_handle backend_handle, const Conv2DStats& stats,
                             const void* grad_output, void* grad_bias, void* workspace,
                             DTypeDesc type_desc) {
  size_t output_h = (stats.input_h + 2 * stats.pad_h - stats.kernel_h) / stats.stride_h + 1;
  size_t output_w = (stats.input_w + 2 * stats.pad_w - stats.kernel_w) / stats.stride_w + 1;
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    conv2d_bgrad_naive_impl<T>(static_cast<const T*>(grad_output), static_cast<T*>(grad_bias),
                               stats.batch_size, stats.out_channels, output_h, output_w);
  });
}

void CPUEngine::legacy_conv2d_fwd(engine_handle backend_handle, const void* col_data,
                                  const void* weight_data, void* output_data, size_t output_size,
                                  size_t kernel_size, size_t out_channels, DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    cpu::legacy_gemm(static_cast<const T*>(weight_data), static_cast<const T*>(col_data),
                     static_cast<T*>(output_data), out_channels, output_size, kernel_size, false, false,
                 T(1.0), T(0.0));
  });
}

void CPUEngine::legacy_conv2d_wgrad(engine_handle backend_handle, const void* col_data,
                                    const void* gradient_data, void* grad_weight_data,
                                    size_t output_size, size_t kernel_size, size_t out_channels,
                                    DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    cpu::legacy_gemm(static_cast<const T*>(gradient_data), static_cast<const T*>(col_data),
                     static_cast<T*>(grad_weight_data), out_channels, kernel_size, output_size, false,
                 true, T(1.0), T(1.0));
  });
}

void CPUEngine::legacy_conv2d_dgrad(engine_handle backend_handle, const void* gradient_data,
                                    const void* weight_data, void* col_grad_data,
                                    size_t output_size, size_t kernel_size, size_t out_channels,
                                    DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    cpu::legacy_gemm(static_cast<const T*>(weight_data), static_cast<const T*>(gradient_data),
                     static_cast<T*>(col_grad_data), kernel_size, output_size, out_channels, true,
                 false, T(1.0), T(0.0));
  });
}

void CPUEngine::legacy_conv2d_bgrad(engine_handle backend_handle, const void* gradient_data,
                                    void* grad_bias_data, size_t batch_size, size_t output_h,
                                    size_t output_w, size_t out_channels, DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    conv2d_bgrad_naive_impl<T>(static_cast<const T*>(gradient_data),
                               static_cast<T*>(grad_bias_data), batch_size, out_channels, output_h,
                               output_w);
  });
}

void CPUEngine::legacy_conv2d_add_bias(engine_handle backend_handle, void* output_data,
                                       const void* bias_data, size_t batch_size, size_t output_h,
                                       size_t output_w, size_t out_channels, DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    add_bias_impl<T>(static_cast<T*>(output_data), static_cast<const T*>(bias_data),
                     batch_size * output_h * output_w, out_channels);
  });
}

WorkspaceReq CPUEngine::query_conv2d_graph(engine_handle backend_handle, const Conv2DStats& stats,
                                           DTypeDesc type_desc) {
  return {0, 0, 0};
}

}  // namespace tunx
