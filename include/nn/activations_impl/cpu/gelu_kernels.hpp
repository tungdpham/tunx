#pragma once

#include <cstddef>
#include "type/type.hpp"

namespace tunx {
namespace func {
namespace cpu {

void gelu(DType_t dtype, const void *input, void *output, size_t size);

void gelu_gradient(DType_t dtype, const void *input, const void *grad_output, void *grad_input, size_t size);

}  // namespace cpu
}  // namespace func
}  // namespace tunx
