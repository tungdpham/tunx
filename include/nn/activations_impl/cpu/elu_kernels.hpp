#pragma once

#include <cstddef>
#include "type/type.hpp"

namespace tunx {
namespace func {
namespace cpu {

void elu(DType_t dtype, const void *input, void *output, size_t size, double alpha);

void elu_gradient(DType_t dtype, const void *input, const void *grad_output, void *grad_input, size_t size, double alpha);

}  // namespace cpu
}  // namespace func
}  // namespace tunx
