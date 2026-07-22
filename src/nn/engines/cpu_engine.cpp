#include "nn/engines/cpu_engine.hpp"

#include <memory>

#include "device/stream.hpp"
#include "nn/engines/engine_handle.hpp"

namespace tunx {

engine_handle CPUEngine::create_handle(stream s) {
  std::shared_ptr<IEngineHandle> handle = std::make_shared<CPUEngineHandle>(s);
  return engine_handle(handle);
}

}  // namespace tunx
