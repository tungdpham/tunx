#pragma once

#ifdef TUNX_USE_CUDA
#include <cuda_runtime.h>

#include <cstddef>
#include "type/type.hpp"

namespace tunx {
namespace cuda {
void elu(DType_t dtype, const void *input, void *output, size_t size, double alpha, cudaStream_t stream);

void elu_gradient(DType_t dtype, const void *input, const void *grad_output, void *grad_input, size_t size, double alpha,
                  cudaStream_t stream);
}  // namespace cuda
}  // namespace tunx

#endif  // TUNX_USE_CUDA
