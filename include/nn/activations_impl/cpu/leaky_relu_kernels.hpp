#pragma once

#include <cstddef>
#include "type/type.hpp"

namespace tunx {
namespace func {
namespace cpu {

void leaky_relu(DType_t dtype, const void *input, void *output, size_t size, double negative_slope);

void leaky_relu_gradient(DType_t dtype, const void *input, const void *grad_output, void *grad_input, size_t size,
                         double negative_slope);

}  // namespace cpu
}  // namespace func
}  // namespace tunx
