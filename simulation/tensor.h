#pragma once

#include <atomic>
#include <cstddef>

#include "allocator.h"

class Tensor {
private:
  struct Impl {
    Allocator* allocator;
    std::atomic<size_t> ref_count;
    size_t size;

    void inc_ref() { ref_count.fetch_add(1, std::memory_order_relaxed); }

    bool dec_ref() { return ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1; }

    Impl()
        : allocator(nullptr),
          ref_count(1),
          size(0) {}

    Impl(Allocator* allocator, size_t size)
        : allocator(allocator),
          ref_count(1),
          size(size) {
      if (allocator && size > 0) {
        allocator->allocate(size);
      }
    }

    ~Impl() {
      if (allocator && size > 0) {
        allocator->free(size);
      }
    }
  };

  Impl* impl_;

  void release() {
    if (impl_ && impl_->dec_ref()) {
      delete impl_;
    }
  }

public:
  Tensor()
      : impl_(new Impl()) {}

  Tensor(Allocator* allocator, size_t size)
      : impl_(new Impl(allocator, size)) {}

  ~Tensor() { release(); }

  Tensor(const Tensor& other)
      : impl_(other.impl_) {
    if (impl_) {
      impl_->inc_ref();
    }
  }

  Tensor& operator=(const Tensor& other) {
    if (this != &other) {
      release();
      impl_ = other.impl_;
      if (impl_) {
        impl_->inc_ref();
      }
    }
    return *this;
  }

  Tensor(Tensor&& other) noexcept
      : impl_(other.impl_) {
    other.impl_ = nullptr;
  }

  Tensor& operator=(Tensor&& other) noexcept {
    if (this != &other) {
      release();
      impl_ = other.impl_;
      other.impl_ = nullptr;
    }
    return *this;
  }

  Tensor(std::nullptr_t)
      : impl_(nullptr) {}

  size_t size() const { return impl_ ? impl_->size : 0; }
  Allocator* allocator() const { return impl_ ? impl_->allocator : nullptr; }

  operator bool() const { return impl_ != nullptr; }
};