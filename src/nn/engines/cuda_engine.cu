#ifdef TUNX_USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <ctime>

#include "device/stream.hpp"
#include "nn/engines/cuda_engine.hpp"
#include "nn/engines/engine_handle.hpp"
namespace tunx {

engine_handle CUDAEngine::create_handle(stream s) {
  std::shared_ptr<IEngineHandle> handle = std::make_shared<CUDAEngineHandle>(s);
  return engine_handle(handle);
}

}  // namespace tunx

#endif
