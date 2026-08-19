#pragma once

#include <cstddef>
#include "type/type.hpp"

namespace tunx {
namespace func {
namespace cpu {

void softmax(DType_t dtype, const void *input, void *output, size_t batch_size, size_t channels, size_t height,
             size_t width);

void softmax_gradient(DType_t dtype, const void *input, const void *grad_output, void *grad_input, size_t batch_size,
                      size_t channels, size_t height, size_t width);

}  // namespace cpu
}  // namespace func
}  // namespace tunx
