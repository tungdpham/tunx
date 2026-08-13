#include "nn/activations_impl/cpu/sigmoid_kernels.hpp"

#include <cmath>

#include "threading/thread_handler.hpp"
#include "type/type.hpp"

namespace tunx {
namespace func {
namespace cpu {
template <typename T>
void sigmoid_impl(const T *input, T *output, size_t size) {
  parallel_for<size_t>(
      0, size, [&](size_t i) { output[i] = T(1) / (T(1) + static_cast<T>(exp(-input[i]))); });
}

template <typename T>
void sigmoid_gradient_impl(const T *input, const T *grad_output, T *grad_input, size_t size) {
  parallel_for<size_t>(0, size, [&](size_t i) {
    T sigmoid_val = T(1) / (T(1) + static_cast<T>(exp(-input[i])));
    grad_input[i] = grad_output[i] * sigmoid_val * (T(1) - sigmoid_val);
  });
}

void sigmoid(DType_t dtype, const void *input, void *output, size_t size) {
  DISPATCH_DTYPE(dtype, T,
                 sigmoid_impl<T>(static_cast<const T *>(input), static_cast<T *>(output), size));
}

void sigmoid_gradient(DType_t dtype, const void *input, const void *grad_output, void *grad_input,
                      size_t size) {
  DISPATCH_DTYPE(
      dtype, T,
      sigmoid_gradient_impl<T>(static_cast<const T *>(input), static_cast<const T *>(grad_output),
                               static_cast<T *>(grad_input), size));
}

}  // namespace cpu
}  // namespace func
}  // namespace tunx
