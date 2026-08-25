/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <mutex>

#include "device/iallocator.hpp"

namespace tunx {

class OffloadAllocator : public IAllocator {
public:
  OffloadAllocator(IAllocator *backend_alloc) : backend_alloc_(backend_alloc), allocated_(0) {}

  dptr allocate(size_t size) override {
    if (size == 0) return dptr(nullptr);

    dptr orig_ptr = backend_alloc_->allocate(size);
    void *ptr = orig_ptr.get<void>();

    allocated_ += size;
    for (auto &hook : hooks_) {
      hook.second(allocated_);
    }

    auto storage = std::make_shared<device_storage>(
        device(), ptr, size, [this, size, orig_ptr]() mutable {
          this->allocated_ -= size;
          orig_ptr = dptr(nullptr);
        });
    return dptr(storage, 0, size);
  }

  void clear() override { backend_alloc_->clear(); }

  void ensure(size_t size) override { backend_alloc_->ensure(size); }

  size_t reserved() const override { return backend_alloc_->reserved(); }

  size_t allocated() const override { return allocated_; }

  size_t unused() const override { return backend_alloc_->unused(); }

  void evict_unused() override { backend_alloc_->evict_unused(); }

  size_t add_allocation_hook(std::function<void(size_t)> hook) override {
    size_t id = next_hook_id_++;
    hooks_[id] = hook;
    return id;
  }

  bool remove_allocation_hook(size_t hook_id) override {
    return hooks_.erase(hook_id) > 0;
  }

  Device &device() const override { return backend_alloc_->device(); }

private:
  IAllocator *backend_alloc_;
  size_t allocated_;
  size_t next_hook_id_ = 0;
  std::map<size_t, std::function<void(size_t)>> hooks_;
};

}  // namespace tunx
