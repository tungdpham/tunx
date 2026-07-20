#pragma once

#ifdef TUNX_USE_CUDNN

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_graph.h>

#include <cstring>

#include "type/type.hpp"

namespace tunx {
namespace cuda {
namespace cudnn {
cudnnDataType_t to_cudnn_datatype(DType_t dtype);
}
}  // namespace cuda
}  // namespace tunx
#endif