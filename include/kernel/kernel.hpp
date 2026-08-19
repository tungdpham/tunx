#pragma once

#include "device/dptr.hpp"
#ifdef TUNX_USE_CUDA
#include "kernel/cuda/kernels.hpp"
#endif
#include "type/type.hpp"

namespace tunx {
namespace kernel {
void copy(const dptr src, dptr dst, size_t size, stream s = nullptr);

void bswap(DType_t dtype, const dptr input, dptr output, size_t size, stream s = nullptr);

}  // namespace kernel
}  // namespace tunx