/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>
#include <map>
#include <mutex>

#include "device/dptr.hpp"
#include "device/iallocator.hpp"
#include "device/stream.hpp"
#include <utility>

namespace tunx {

// Allocates a device pointer that contains a storage block that can be shared and automatically
// reclaimed by allocator by installing a custom deleter in device_storage's shared_ptr.
// Ensures user don't do some bad memory management.
// Bounded to a specific device and flow, so that we can reuse memory across different tensors on
// the same device and flow, but not across different devices or flows.
class PoolAllocator : public IAllocator {
public:
  PoolAllocator(Device &device, stream s)
      : device_(device),
        stream_(s) {}
  ~PoolAllocator() { clear(); }

  PoolAllocator(const PoolAllocator &) = delete;
  PoolAllocator &operator=(const PoolAllocator &) = delete;

  static PoolAllocator &instance(Device &device, stream s) {
    static std::mutex registry_mutex;
    static std::map<std::pair<Device *, stream>, std::unique_ptr<PoolAllocator>> instances;
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto &pool = instances[{&device, s}];
    if (!pool) {
      pool = std::make_unique<PoolAllocator>(device, s);
    }
    return *pool;
  }

  dptr allocate(size_t size) override {
    if (size == 0) {
      auto storage = std::make_shared<device_storage>(device_, nullptr, 0, []() {});
      return dptr(storage, 0, 0);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    void *ptr = nullptr;
    auto it = free_blocks_.lower_bound(size);
    if (it != free_blocks_.end()) {
      ptr = it->second;
      free_blocks_.erase(it);
    } else {
      ptr = device_.allocate_aligned_memory(size, DEFAULT_ALIGNMENT);
    }
    set_allocated(allocated_ + size);

    auto storage = std::make_shared<device_storage>(device_, ptr, size,
                                                    [this, ptr, size]() { reclaim(ptr, size); });
    return dptr(storage, 0, size);
  }

  void clear() override {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &pair : free_blocks_) {
      device_.deallocate_aligned_memory(pair.second);
    }
    free_blocks_.clear();
  }

  void ensure(size_t size) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = free_blocks_.lower_bound(size);
    if (it != free_blocks_.end()) {
      return;
    }
    void *ptr = device_.allocate_aligned_memory(size, DEFAULT_ALIGNMENT);
    free_blocks_.emplace(size, ptr);
  }

  size_t reserved() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (const auto &pair : free_blocks_) {
      total += pair.first;
    }
    return total + allocated_;
  }

  size_t allocated() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocated_;
  }

  size_t unused() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (const auto &pair : free_blocks_) {
      total += pair.first;
    }
    return total;
  }

  void evict_unused() override {
    clear();  // clears all free blocks
  }

  size_t add_allocation_hook(std::function<void(size_t)> hook) override {
    std::lock_guard<std::mutex> lock(mutex_);
    allocation_hooks_.push_back(hook);
    return allocation_hooks_.size() - 1;
  }

  bool remove_allocation_hook(size_t hook_id) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (hook_id >= allocation_hooks_.size()) {
      return false;
    }
    allocation_hooks_.erase(allocation_hooks_.begin() + hook_id);
    return true;
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_blocks_.size();
  }

  size_t cached_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (const auto &pair : free_blocks_) {
      total += pair.first;
    }
    return total;
  }

  size_t peak_allocated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return peak_allocated_;
  }

  Device &device() const override { return device_; }

private:
  std::multimap<size_t, void *> free_blocks_;
  Device &device_;
  stream stream_;
  mutable std::mutex mutex_;
  size_t allocated_ = 0;
  size_t peak_allocated_ = 0;
  std::vector<std::function<void(size_t)>> allocation_hooks_;

  void set_allocated(size_t new_total) {
    allocated_ = new_total;
    if (allocated_ > peak_allocated_) {
      peak_allocated_ = allocated_;
    }
    for (auto &hook : allocation_hooks_) {
      hook(allocated_);
    }
  }

  void reclaim(void *ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    set_allocated(allocated_ - size);
    free_blocks_.emplace(size, ptr);
  }
};

}  // namespace tunx