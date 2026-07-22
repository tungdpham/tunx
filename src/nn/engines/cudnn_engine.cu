#ifdef TUNX_USE_CUDNN
#include <cuda_runtime.h>
#include <fmt/core.h>

#include <memory>

#include "device/stream.hpp"
#include "nn/engines/cudnn_engine.hpp"
#include "nn/engines/engine_handle.hpp"

namespace tunx {

engine_handle CuDNNEngine::create_handle(stream s) {
  std::shared_ptr<IEngineHandle> handle = std::make_shared<CuDNNEngineHandle>(s);
  return engine_handle(handle);
}

}  // namespace tunx

#endif
