/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include <cuda_runtime.h>

#include <cstddef>
#include "type/type.hpp"

namespace tunx {
namespace cuda {

void gelu(DType_t dtype, const void *input, void *output, size_t size, cudaStream_t stream);

void gelu_gradient(DType_t dtype, const void *input, const void *grad_output, void *grad_input, size_t size,
                   cudaStream_t stream);

}  // namespace cuda
}  // namespace tunx
