#pragma once

#ifdef TUNX_USE_CUDA
#include <cuda_runtime.h>

#include <cstddef>
#include "type/type.hpp"

namespace tunx {
namespace cuda {
void softmax(DType_t dtype, const void *input, void *output, size_t batch_size, size_t channels, size_t height,
             size_t width, cudaStream_t stream);

void softmax_gradient(DType_t dtype, const void *input, const void *grad_output, void *grad_input, size_t batch_size,
                      size_t channels, size_t height, size_t width, cudaStream_t stream);
}  // namespace cuda
}  // namespace tunx

#endif  // TUNX_USE_CUDA
