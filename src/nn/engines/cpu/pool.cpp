#include "internal.hpp"
#include "nn/engines/cpu_engine.hpp"
#include "threading/thread_handler.hpp"
#include "type/type.hpp"

namespace tunx {
namespace {

template <typename T>
void avgpool_fwd_impl(const T* input, T* output, size_t batch_size, size_t height, size_t width,
                      size_t channels, size_t pool_h, size_t pool_w, size_t stride_h,
                      size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h,
                      size_t output_w) {
  parallel_for_2d(batch_size, output_h, [&](size_t b, size_t oh) {
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
  });
}

template <typename T>
void avgpool_bwd_impl(const T* grad_output, T* grad_input, size_t batch_size, size_t input_h,
                      size_t input_w, size_t channels, size_t pool_h, size_t pool_w,
                      size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h,
                      size_t output_w) {
  parallel_for<size_t>(0, batch_size, [&](size_t b) {
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
  });
}

template <typename T>
void maxpool2d_fwd_impl(const T* input, T* output, int32* mask, size_t batch_size, size_t input_h,
                        size_t input_w, size_t channels, size_t pool_h, size_t pool_w,
                        size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                        size_t output_h, size_t output_w) {
  parallel_for_2d(batch_size, output_h, [&](size_t b, size_t oh) {
    for (size_t ow = 0; ow < output_w; ++ow) {
      int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
      int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
      int h_end = std::min(h_start + static_cast<int>(pool_h), static_cast<int>(input_h));
      int w_end = std::min(w_start + static_cast<int>(pool_w), static_cast<int>(input_w));
      h_start = std::max(h_start, 0);
      w_start = std::max(w_start, 0);

      size_t out_base_idx = ((b * output_h + oh) * output_w + ow) * channels;

      for (size_t c = 0; c < channels; ++c) {
        output[out_base_idx + c] = std::numeric_limits<T>::lowest();
        if (mask) mask[out_base_idx + c] = -1;
      }

      for (int h = h_start; h < h_end; ++h) {
        for (int w = w_start; w < w_end; ++w) {
          size_t in_base_idx = ((b * input_h + h) * input_w + w) * channels;

          for (size_t c = 0; c < channels; ++c) {
            T val = input[in_base_idx + c];
            if (val > output[out_base_idx + c]) {
              output[out_base_idx + c] = val;
              if (mask) {
                int h_start_unclamped = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
                int w_start_unclamped = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
                int rel_h = h - h_start_unclamped;
                int rel_w = w - w_start_unclamped;
                mask[out_base_idx + c] = rel_h * static_cast<int>(pool_w) + rel_w;
              }
            }
          }
        }
      }
    }
  });
}

template <typename T>
void maxpool2d_bwd_impl(const T* grad_output, T* grad_input, const int32* mask, size_t batch_size,
                        size_t input_h, size_t input_w, size_t channels, size_t output_h,
                        size_t output_w, size_t pool_w, size_t stride_h, size_t stride_w,
                        size_t pad_h, size_t pad_w) {
  size_t total_elements = batch_size * input_h * input_w * channels;
  parallel_for<size_t>(0, total_elements, [=](size_t i) { grad_input[i] = 0; });
  parallel_for<size_t>(0, batch_size, [&](size_t b) {
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
          grad_input[in_idx] += grad_output[global_i];
        }
      }
    }
  });
}

template <typename T>
void legacy_avgpool2d_fwd_impl(const T* input_data, T* output_data, size_t batch_size,
                               size_t channels, size_t input_h, size_t input_w, size_t output_h,
                               size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                               size_t stride_w, size_t pad_h, size_t pad_w) {
  const T pool_size_inv = T(1.0) / T(pool_h * pool_w);

  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    size_t input_offset = (n * channels + c) * input_h * input_w;
    size_t output_offset = (n * channels + c) * output_h * output_w;

    for (size_t out_h = 0; out_h < output_h; ++out_h) {
      for (size_t out_w = 0; out_w < output_w; ++out_w) {
        long h_start = static_cast<long>(out_h * stride_h) - static_cast<long>(pad_h);
        long w_start = static_cast<long>(out_w * stride_w) - static_cast<long>(pad_w);

        long h_start_valid = std::max(0L, h_start);
        long w_start_valid = std::max(0L, w_start);
        long h_end_valid =
            std::min(static_cast<long>(input_h), h_start + static_cast<long>(pool_h));
        long w_end_valid =
            std::min(static_cast<long>(input_w), w_start + static_cast<long>(pool_w));

        T sum = T(0);

        for (long ih = h_start_valid; ih < h_end_valid; ++ih) {
          for (long iw = w_start_valid; iw < w_end_valid; ++iw) {
            sum += input_data[input_offset + ih * input_w + iw];
          }
        }

        size_t output_idx = output_offset + out_h * output_w + out_w;
        output_data[output_idx] = sum * pool_size_inv;
      }
    }
  });
}

template <typename T>
void legacy_avgpool2d_bwd_impl(const T* gradient_data, T* grad_input_data, size_t batch_size,
                               size_t channels, size_t input_h, size_t input_w, size_t output_h,
                               size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                               size_t stride_w, size_t pad_h, size_t pad_w) {
  const T pool_size_inv = T(1.0) / T(pool_h * pool_w);

  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    size_t input_offset = (n * channels + c) * input_h * input_w;
    size_t output_offset = (n * channels + c) * output_h * output_w;

    for (size_t out_h = 0; out_h < output_h; ++out_h) {
      for (size_t out_w = 0; out_w < output_w; ++out_w) {
        size_t output_idx = output_offset + out_h * output_w + out_w;

        const T grad_val = gradient_data[output_idx] * pool_size_inv;

        long h_start = static_cast<long>(out_h * stride_h) - static_cast<long>(pad_h);
        long w_start = static_cast<long>(out_w * stride_w) - static_cast<long>(pad_w);

        long h_start_valid = std::max(0L, h_start);
        long w_start_valid = std::max(0L, w_start);
        long h_end_valid =
            std::min(static_cast<long>(input_h), h_start + static_cast<long>(pool_h));
        long w_end_valid =
            std::min(static_cast<long>(input_w), w_start + static_cast<long>(pool_w));

        for (long ih = h_start_valid; ih < h_end_valid; ++ih) {
          for (long iw = w_start_valid; iw < w_end_valid; ++iw) {
            grad_input_data[input_offset + ih * input_w + iw] += grad_val;
          }
        }
      }
    }
  });
}

template <typename T>
void legacy_maxpool2d_fwd_impl(const T* input_data, T* output_data, size_t batch_size,
                               size_t channels, size_t input_h, size_t input_w, size_t output_h,
                               size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                               size_t stride_w, size_t pad_h, size_t pad_w, size_t* mask_indices) {
  const T MIN_VALUE = std::numeric_limits<T>::lowest();

  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    size_t input_offset = (n * channels + c) * input_h * input_w;
    size_t output_offset = (n * channels + c) * output_h * output_w;

    for (size_t out_h = 0; out_h < output_h; ++out_h) {
      for (size_t out_w = 0; out_w < output_w; ++out_w) {
        long h_start = static_cast<long>(out_h * stride_h) - static_cast<long>(pad_h);
        long w_start = static_cast<long>(out_w * stride_w) - static_cast<long>(pad_w);
        long h_end = h_start + pool_h;
        long w_end = w_start + pool_w;

        long h_start_valid = std::max(0L, h_start);
        long w_start_valid = std::max(0L, w_start);
        long h_end_valid = std::min(static_cast<long>(input_h), h_end);
        long w_end_valid = std::min(static_cast<long>(input_w), w_end);

        T max_val = MIN_VALUE;
        size_t max_idx = 0;

        for (long ih = h_start_valid; ih < h_end_valid; ++ih) {
          for (long iw = w_start_valid; iw < w_end_valid; ++iw) {
            size_t cur_input_idx = input_offset + ih * input_w + iw;
            T val = input_data[cur_input_idx];

            if (val > max_val) {
              max_val = val;
              max_idx = cur_input_idx;
            }
          }
        }

        size_t out_idx = output_offset + out_h * output_w + out_w;
        output_data[out_idx] = max_val;
        mask_indices[out_idx] = max_idx;
      }
    }
  });
}

template <typename T>
void legacy_maxpool2d_bwd_impl(const T* gradient_data, T* grad_input_data, size_t batch_size,
                               size_t channels, size_t output_h, size_t output_w,
                               const size_t* mask_indices) {
  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    size_t output_offset = (n * channels + c) * output_h * output_w;

    for (size_t i = 0; i < output_h * output_w; ++i) {
      size_t out_idx = output_offset + i;
      size_t input_idx = mask_indices[out_idx];
      grad_input_data[input_idx] += gradient_data[out_idx];
    }
  });
}

}  // namespace

void CPUEngine::avgpool_fwd(engine_handle backend_handle, const AvgPool2DStats& stats,
                            const void* input, void* output, void* workspace, DTypeDesc type_desc) {
  size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
  size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    avgpool_fwd_impl<T>(static_cast<const T*>(input), static_cast<T*>(output), stats.batch_size,
                        stats.height, stats.width, stats.channels, stats.pool_h, stats.pool_w,
                        stats.stride_h, stats.stride_w, stats.pad_h, stats.pad_w, output_h,
                        output_w);
  });
}

void CPUEngine::avgpool_bwd(engine_handle backend_handle, const AvgPool2DStats& stats,
                            const void* grad_output, void* grad_input, void* workspace,
                            DTypeDesc type_desc) {
  size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
  size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    avgpool_bwd_impl<T>(static_cast<const T*>(grad_output), static_cast<T*>(grad_input),
                        stats.batch_size, stats.height, stats.width, stats.channels, stats.pool_h,
                        stats.pool_w, stats.stride_h, stats.stride_w, stats.pad_h, stats.pad_w,
                        output_h, output_w);
  });
}

void CPUEngine::maxpool2d_fwd(engine_handle backend_handle, const MaxPool2DStats& stats,
                              const void* input, void* output, void* mask, void* workspace,
                              DTypeDesc type_desc) {
  size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
  size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    maxpool2d_fwd_impl<T>(static_cast<const T*>(input), static_cast<T*>(output),
                          static_cast<int*>(mask), stats.batch_size, stats.height, stats.width,
                          stats.channels, stats.pool_h, stats.pool_w, stats.stride_h,
                          stats.stride_w, stats.pad_h, stats.pad_w, output_h, output_w);
  });
}

void CPUEngine::maxpool2d_infer(engine_handle backend_handle, const MaxPool2DStats& stats,
                                const void* input, void* output, void* workspace,
                                DTypeDesc type_desc) {
  size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
  size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    maxpool2d_fwd_impl<T>(static_cast<const T*>(input), static_cast<T*>(output),
                          static_cast<int32*>(nullptr), stats.batch_size, stats.height, stats.width,
                          stats.channels, stats.pool_h, stats.pool_w, stats.stride_h,
                          stats.stride_w, stats.pad_h, stats.pad_w, output_h, output_w);
  });
}

void CPUEngine::maxpool2d_bwd(engine_handle backend_handle, const MaxPool2DStats& stats,
                              const void* grad_output, void* grad_input, const void* mask,
                              void* workspace, DTypeDesc type_desc) {
  size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
  size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    maxpool2d_bwd_impl<T>(static_cast<const T*>(grad_output), static_cast<T*>(grad_input),
                          static_cast<const int*>(mask), stats.batch_size, stats.height,
                          stats.width, stats.channels, output_h, output_w, stats.pool_w,
                          stats.stride_h, stats.stride_w, stats.pad_h, stats.pad_w);
  });
}

void CPUEngine::legacy_avgpool2d_fwd(engine_handle backend_handle, const void* input, void* output,
                                     size_t batch_size, size_t channels, size_t input_h,
                                     size_t input_w, size_t output_h, size_t output_w,
                                     size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w,
                                     size_t pad_h, size_t pad_w, DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    legacy_avgpool2d_fwd_impl<T>(static_cast<const T*>(input), static_cast<T*>(output), batch_size,
                                 channels, input_h, input_w, output_h, output_w, pool_h, pool_w,
                                 stride_h, stride_w, pad_h, pad_w);
  });
}

void CPUEngine::legacy_avgpool2d_bwd(engine_handle backend_handle, const void* grad_output,
                                     void* grad_input, size_t batch_size, size_t channels,
                                     size_t input_h, size_t input_w, size_t output_h,
                                     size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                                     size_t stride_w, size_t pad_h, size_t pad_w,
                                     DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    legacy_avgpool2d_bwd_impl<T>(static_cast<const T*>(grad_output), static_cast<T*>(grad_input),
                                 batch_size, channels, input_h, input_w, output_h, output_w, pool_h,
                                 pool_w, stride_h, stride_w, pad_h, pad_w);
  });
}

void CPUEngine::legacy_maxpool2d_fwd(engine_handle backend_handle, const void* input, void* output,
                                     size_t batch_size, size_t channels, size_t input_h,
                                     size_t input_w, size_t output_h, size_t output_w,
                                     size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w,
                                     size_t pad_h, size_t pad_w, void* mask_indices,
                                     DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    legacy_maxpool2d_fwd_impl<T>(static_cast<const T*>(input), static_cast<T*>(output), batch_size,
                                 channels, input_h, input_w, output_h, output_w, pool_h, pool_w,
                                 stride_h, stride_w, pad_h, pad_w,
                                 static_cast<size_t*>(mask_indices));
  });
}

void CPUEngine::legacy_maxpool2d_bwd(engine_handle backend_handle, const void* grad_output,
                                     void* grad_input, size_t batch_size, size_t channels,
                                     size_t output_h, size_t output_w, const void* mask_indices,
                                     DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    legacy_maxpool2d_bwd_impl<T>(static_cast<const T*>(grad_output), static_cast<T*>(grad_input),
                                 batch_size, channels, output_h, output_w,
                                 static_cast<const size_t*>(mask_indices));
  });
}

WorkspaceReq CPUEngine::query_avgpool_graph(engine_handle backend_handle, const AvgPool2DStats&,
                                            DTypeDesc) {
  return {0, 0, 0};
}

WorkspaceReq CPUEngine::query_maxpool2d_graph(engine_handle backend_handle, const MaxPool2DStats&,
                                              DTypeDesc) {
  return {0, 0, 0};
}

}  // namespace tunx
