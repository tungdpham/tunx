#include "nn/activations_impl/cpu/tanh_kernels.hpp"

#include <cmath>

#include "threading/thread_handler.hpp"
#include "type/type.hpp"

namespace tunx {
namespace func {
namespace cpu {
template <typename T>
void tanh_impl(const T *input, T *output, size_t size) {
  parallel_for<size_t>(0, size, [&](size_t i) {
    output[i] = static_cast<T>(std::tanh(static_cast<double>(input[i])));
  });
}

template <typename T>
void tanh_gradient_impl(const T *input, const T *grad_output, T *grad_input, size_t size) {
  parallel_for<size_t>(0, size, [&](size_t i) {
    double tanh_val = std::tanh(static_cast<double>(input[i]));
    grad_input[i] =
        static_cast<T>(static_cast<double>(grad_output[i]) * (1.0 - tanh_val * tanh_val));
  });
}

void tanh(DType_t dtype, const void *input, void *output, size_t size) {
  DISPATCH_DTYPE(dtype, T,
                 tanh_impl<T>(static_cast<const T *>(input), static_cast<T *>(output), size));
}

void tanh_gradient(DType_t dtype, const void *input, const void *grad_output, void *grad_input,
                   size_t size) {
  DISPATCH_DTYPE(
      dtype, T,
      tanh_gradient_impl<T>(static_cast<const T *>(input), static_cast<const T *>(grad_output),
                            static_cast<T *>(grad_input), size));
}

}  // namespace cpu
}  // namespace func
}  // namespace tunx
