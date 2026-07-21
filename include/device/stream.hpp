/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>
#include <memory>

#ifdef TUNX_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <stdexcept>

namespace tunx {
class Device;

class stream_impl {
public:
  stream_impl() = default;
  virtual ~stream_impl() = default;

  virtual void sync() = 0;

  virtual Device* device() const = 0;
};

class stream {
public:
  stream() = default;
  stream(std::nullptr_t)
      : impl_(nullptr) {}
  stream(std::shared_ptr<stream_impl> impl)
      : impl_(impl) {}

  operator bool() const { return impl_.get() != nullptr; }

  bool operator<(const stream& other) const { return this->impl_ < other.impl_; }

  void sync() {
    check();
    impl_->sync();
  }

  template <typename StreamType>
  StreamType* as() {
    return dynamic_cast<StreamType*>(impl_.get());
  }

private:
  void check() const {
    if (!impl_) {
      throw std::runtime_error("Stream is null");
    }
  }
  std::shared_ptr<stream_impl> impl_;
};

}  // namespace tunx
