#include "nn/activations_impl/cpu/gelu_kernels.hpp"

#include <cmath>

#include "type/type.hpp"

namespace tunx {
namespace func {
namespace cpu {

template <typename T>
void gelu_impl(const T *input, T *output, size_t size) {
  const double k0 = 0.7978845608028654;  // sqrt(2/pi)
  const double k1 = 0.044715;

  for (size_t i = 0; i < size; ++i) {
    double x = static_cast<double>(input[i]);
    double x3 = x * x * x;
    double inner = k0 * (x + k1 * x3);
    output[i] = static_cast<T>(0.5 * x * (1.0 + std::tanh(inner)));
  }
}

template <typename T>
void gelu_gradient_impl(const T *input, const T *grad_output, T *grad_input, size_t size) {
  const double k0 = 0.7978845608028654;  // sqrt(2/pi)
  const double k1 = 0.044715;

  for (size_t i = 0; i < size; ++i) {
    double x = static_cast<double>(input[i]);
    double x3 = x * x * x;
    double inner = k0 * (x + k1 * x3);
    double tanh_inner = std::tanh(inner);

    double sechip = 1.0 / std::cosh(inner);
    double sechip2 = sechip * sechip;

    double d_inner_dx = k0 * (1.0 + 3.0 * k1 * x * x);

    // d(GELU)/dx = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * d_inner_dx
    double grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sechip2 * d_inner_dx;

    grad_input[i] = static_cast<T>(static_cast<double>(grad_output[i]) * grad);
  }
}

void gelu(DType_t dtype, const void *input, void *output, size_t size) {
  DISPATCH_DTYPE(dtype, T,
                 gelu_impl<T>(static_cast<const T *>(input), static_cast<T *>(output), size));
}

void gelu_gradient(DType_t dtype, const void *input, const void *grad_output, void *grad_input,
                   size_t size) {
  DISPATCH_DTYPE(
      dtype, T,
      gelu_gradient_impl<T>(static_cast<const T *>(input), static_cast<const T *>(grad_output),
                            static_cast<T *>(grad_input), size));
}

}  // namespace cpu
}  // namespace func
}  // namespace tunx
