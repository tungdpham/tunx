#include "nn/activations_impl/cpu/softmax_kernels.hpp"

#include <cmath>

#include "threading/thread_handler.hpp"
#include "type/type.hpp"

namespace tunx {
namespace func {
namespace cpu {
template <typename T>
void softmax_impl(const T *input, T *output, size_t batch_size, size_t channels, size_t height,
                  size_t width) {
  size_t spatial_size = height * width;
  size_t channel_stride = spatial_size;
  size_t batch_stride = channels * channel_stride;

  parallel_for<size_t>(0, batch_size, [&](size_t n) {
    for (size_t h = 0; h < height; ++h) {
      for (size_t w = 0; w < width; ++w) {
        size_t spatial_idx = h * width + w;

        T max_val = input[n * batch_stride + spatial_idx];
        for (size_t c = 1; c < channels; ++c) {
          size_t idx = n * batch_stride + c * channel_stride + spatial_idx;
          T val = input[idx];
          if (val > max_val) {
            max_val = val;
          }
        }

        T sum_exp = T(0);
        for (size_t c = 0; c < channels; ++c) {
          size_t idx = n * batch_stride + c * channel_stride + spatial_idx;
          T exp_val = exp(input[idx] - max_val);
          output[idx] = exp_val;
          sum_exp += exp_val;
        }

        for (size_t c = 0; c < channels; ++c) {
          size_t idx = n * batch_stride + c * channel_stride + spatial_idx;
          output[idx] /= sum_exp;
        }
      }
    }
  });
}

template <typename T>
void softmax_gradient_impl(const T *input, const T *grad_output, T *grad_input, size_t batch_size,
                           size_t channels, size_t height, size_t width) {
  size_t spatial_size = height * width;
  size_t channel_stride = spatial_size;
  size_t batch_stride = channels * channel_stride;

  T *softmax_values = new T[batch_size * channels * spatial_size];
  softmax_impl(input, softmax_values, batch_size, channels, height, width);

  parallel_for<size_t>(0, batch_size, [&](size_t n) {
    for (size_t h = 0; h < height; ++h) {
      for (size_t w = 0; w < width; ++w) {
        size_t spatial_idx = h * width + w;

        T dot_product = T(0);
        for (size_t j = 0; j < channels; ++j) {
          size_t idx = n * batch_stride + j * channel_stride + spatial_idx;
          dot_product += softmax_values[idx] * grad_output[idx];
        }

        for (size_t i = 0; i < channels; ++i) {
          size_t idx = n * batch_stride + i * channel_stride + spatial_idx;
          T s_i = softmax_values[idx];
          T upstream_i = grad_output[idx];
          grad_input[idx] = s_i * (upstream_i - dot_product);
        }
      }
    }
  });

  delete[] softmax_values;
}

void softmax(DType_t dtype, const void *input, void *output, size_t batch_size, size_t channels,
             size_t height, size_t width) {
  DISPATCH_DTYPE(dtype, T,
                 softmax_impl<T>(static_cast<const T *>(input), static_cast<T *>(output),
                                 batch_size, channels, height, width));
}

void softmax_gradient(DType_t dtype, const void *input, const void *grad_output, void *grad_input,
                      size_t batch_size, size_t channels, size_t height, size_t width) {
  DISPATCH_DTYPE(
      dtype, T,
      softmax_gradient_impl<T>(static_cast<const T *>(input), static_cast<const T *>(grad_output),
                               static_cast<T *>(grad_input), batch_size, channels, height, width));
}

}  // namespace cpu
}  // namespace func
}  // namespace tunx
