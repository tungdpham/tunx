#ifdef TUNX_USE_CUDA
#include <cuda_runtime.h>

#include "cuda/helpers.cuh"

namespace tunx {
namespace cuda {
void synchronize() { cudaDeviceSynchronize(); }
}  // namespace cuda
}  // namespace tunx

#endif  // TUNX_USE_CUDA