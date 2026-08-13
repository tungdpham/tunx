#include "nn/activations_impl/cpu/relu_kernels.hpp"

#include "threading/thread_handler.hpp"
#include "type/type.hpp"

namespace tunx {
namespace func {
namespace cpu {
template <typename T>
void relu_impl(const T *input, T *output, size_t size) {
  parallel_for<size_t>(0, size, [&](size_t i) { output[i] = input[i] > T(0) ? input[i] : T(0); });
}

template <typename T>
void relu_gradient_impl(const T *input, const T *grad_output, T *grad_input, size_t size) {
  parallel_for<size_t>(0, size,
                       [&](size_t i) { grad_input[i] = input[i] > T(0) ? grad_output[i] : T(0); });
}

void relu(DType_t dtype, const void *input, void *output, size_t size) {
  DISPATCH_DTYPE(dtype, T,
                 relu_impl<T>(static_cast<const T *>(input), static_cast<T *>(output), size));
}

void relu_gradient(DType_t dtype, const void *input, const void *grad_output, void *grad_input,
                   size_t size) {
  DISPATCH_DTYPE(
      dtype, T,
      relu_gradient_impl<T>(static_cast<const T *>(input), static_cast<const T *>(grad_output),
                            static_cast<T *>(grad_input), size));
}

}  // namespace cpu
}  // namespace func
}  // namespace tunx