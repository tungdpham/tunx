#include "nn/activations_impl/cpu/elu_kernels.hpp"

#include <cmath>

#include "threading/thread_handler.hpp"
#include "type/type.hpp"

namespace tunx {
namespace func {
namespace cpu {
template <typename T>
void elu_impl(const T *input, T *output, size_t size, T alpha) {
  parallel_for<size_t>(0, size, [&](size_t i) {
    output[i] = input[i] > T(0) ? input[i] : alpha * (static_cast<T>(exp(input[i])) - T(1));
  });
}

template <typename T>
void elu_gradient_impl(const T *input, const T *grad_output, T *grad_input, size_t size, T alpha) {
  parallel_for<size_t>(0, size, [&](size_t i) {
    grad_input[i] =
        input[i] > T(0) ? grad_output[i] : grad_output[i] * alpha * static_cast<T>(exp(input[i]));
  });
}

void elu(DType_t dtype, const void *input, void *output, size_t size, double alpha) {
  DISPATCH_DTYPE(dtype, T,
                 elu_impl<T>(static_cast<const T *>(input), static_cast<T *>(output), size,
                             static_cast<T>(alpha)));
}

void elu_gradient(DType_t dtype, const void *input, const void *grad_output, void *grad_input,
                  size_t size, double alpha) {
  DISPATCH_DTYPE(
      dtype, T,
      elu_gradient_impl<T>(static_cast<const T *>(input), static_cast<const T *>(grad_output),
                           static_cast<T *>(grad_input), size, static_cast<T>(alpha)));
}

}  // namespace cpu
}  // namespace func
}  // namespace tunx
