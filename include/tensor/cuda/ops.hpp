#pragma once

#ifdef TUNX_USE_CUDA
#include <cuda_runtime.h>

#include <cstddef>

#include "type/type.hpp"

namespace tunx {
namespace cuda {

void im2col(DType_t dtype, const void *input, void *col_data, size_t batch_size, size_t channels,
            size_t height, size_t width, size_t kernel_h, size_t kernel_w, size_t stride_h,
            size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h, size_t output_w,
            cudaStream_t stream);

void col2im(DType_t dtype, const void *col_data, void *output, size_t batch_size, size_t channels,
            size_t height, size_t width, size_t kernel_h, size_t kernel_w, size_t stride_h,
            size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h, size_t output_w,
            cudaStream_t stream);

void pad(DType_t dtype, const void *input, void *output, size_t batch_size, size_t channels,
         size_t height, size_t width, size_t pad_h, size_t pad_w, double value,
         cudaStream_t stream);

void unpad(DType_t dtype, const void *input, void *output, size_t batch_size, size_t channels,
           size_t height, size_t width, size_t pad_h, size_t pad_w, cudaStream_t stream);

void crop(DType_t dtype, const void *input, void *output, size_t batch_size, size_t channels,
          size_t height, size_t width, size_t start_h, size_t start_w, size_t new_height,
          size_t new_width, cudaStream_t stream);

void transpose_2d(DType_t dtype, const void *input, void *output, size_t rows, size_t cols,
                  cudaStream_t stream);

void nchw_to_cnhw(DType_t dtype, const void *input, void *output, size_t n, size_t c, size_t h,
                  size_t w, cudaStream_t stream);

void cnhw_to_nchw(DType_t dtype, const void *input, void *output, size_t n, size_t c, size_t h,
                  size_t w, cudaStream_t stream);

}  // namespace cuda
}  // namespace tunx

#endif