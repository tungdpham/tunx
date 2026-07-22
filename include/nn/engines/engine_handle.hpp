#pragma once

#include <cassert>
#include <cstddef>
#include <memory>

#include "device/stream.hpp"
namespace tunx {

class IEngineHandle {
public:
  virtual ~IEngineHandle() = default;

  virtual stream get_stream() = 0;
};

class engine_handle {
public:
  engine_handle() = default;
  engine_handle(std::nullptr_t)
      : impl_(nullptr) {}

  engine_handle(std::shared_ptr<IEngineHandle> impl)
      : impl_(std::move(impl)) {}

  operator bool() const { return impl_.get() != nullptr; }

  stream get_stream() {
    assert(impl_.get() != nullptr && "Engine handle is null");
    return impl_->get_stream();
  }

  template <typename StreamType>
  StreamType *stream_as() {
    return get_stream().as<StreamType>();
  }

  template <typename T>
  T *as() const {
    return static_cast<T *>(impl_.get());
  }

private:
  std::shared_ptr<IEngineHandle> impl_;
};

}  // namespace tunx