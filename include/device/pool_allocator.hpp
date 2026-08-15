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
    static std::map<stream, std::unique_ptr<PoolAllocator>> instances;
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto &pool = instances[s];
    if (!pool) {
      pool = std::make_unique<PoolAllocator>(device, s);
    }
    return *pool;
  }

  dptr allocate(size_t size) override {
    device_storage *ptr = allocate_storage(size);
    auto storage =
        std::shared_ptr<device_storage>(ptr, [this](device_storage *ptr) { this->reclaim(ptr); });
    return dptr(storage, 0, size);
  }

  void clear() override {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &[size, storage] : free_blocks_) {
      if (storage) {
        device_.deallocate_aligned_memory(storage->data());
        delete storage;
      }
    }
    free_blocks_.clear();
  }

  void ensure(size_t size) override {
    auto it = free_blocks_.lower_bound(size);
    if (it != free_blocks_.end()) {
      return;
    }
    // if not enough space allocate new
    device_storage *ptr = allocate_storage(size);
    std::lock_guard<std::mutex> lock(mutex_);
    free_blocks_.emplace(ptr->capacity(), ptr);  // add to free blocks for future reuse
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

  Device &device() const override { return device_; }

private:
  std::multimap<size_t, device_storage *> free_blocks_;
  Device &device_;
  stream stream_;
  mutable std::mutex mutex_;
  size_t allocated_ = 0;
  std::vector<std::function<void(size_t)>> allocation_hooks_;

  void set_allocated(size_t new_total) {
    allocated_ = new_total;
    for (auto &hook : allocation_hooks_) {
      hook(allocated_);
    }
  }

  device_storage *allocate_storage(size_t size) {
    if (size == 0) {
      return new device_storage(device_, nullptr, 0, DEFAULT_ALIGNMENT);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = free_blocks_.lower_bound(size);
    if (it != free_blocks_.end()) {
      device_storage *block = it->second;
      free_blocks_.erase(it);
      return block;
    }
    void *ptr = device_.allocate_aligned_memory(size, DEFAULT_ALIGNMENT);

    set_allocated(allocated_ + size);

    return new device_storage(device_, ptr, size, DEFAULT_ALIGNMENT);
  }

  void reclaim(device_storage *storage) {
    std::lock_guard<std::mutex> lock(mutex_);

    set_allocated(allocated_ - storage->capacity());

    free_blocks_.emplace(storage->capacity(), storage);
  }
};

}  // namespace tunx