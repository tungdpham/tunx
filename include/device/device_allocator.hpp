/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <unordered_map>

#include "device/device_manager.hpp"
#include "device/dptr.hpp"
#include "device/iallocator.hpp"

namespace tunx {

// default allocator
class DeviceAllocator : public IAllocator {
public:
  DeviceAllocator(Device& device)
      : device_(device) {}

  static DeviceAllocator& instance(Device& device) {
    static std::mutex registry_mutex;
    static std::unordered_map<Device*, std::unique_ptr<DeviceAllocator>> registry;
    std::lock_guard<std::mutex> lock(registry_mutex);
    if (registry.find(&device) == registry.end()) {
      registry[&device] = std::make_unique<DeviceAllocator>(device);
    }
    return *registry[&device];
  }

  dptr allocate(size_t size) override {
    void* ptr = device_->allocate_aligned_memory(size, DEFAULT_ALIGNMENT);

    set_allocated(allocated_ + size);

    auto storage = std::make_shared<device_storage>(device_, ptr, size, [this, ptr, size]() {
      device_->deallocate_aligned_memory(ptr);
      std::lock_guard<std::mutex> lock(mutex_);
      set_allocated(allocated_ - size);
    });

    return dptr(storage, 0, size);
  }

  void clear() override {
    // no-op since we don't cache any memory blocks
  }

  void ensure(size_t size) override {
    // no-op since we don't cache any memory blocks
  }

  Device& device() const override { return *device_; }

  size_t reserved() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocated_;  // No extra reservation
  }

  size_t allocated() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocated_;
  }

  size_t unused() const override {
    return 0;  // No caching
  }

  void evict_unused() override {
    // No-op
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

private:
  sref<Device> device_;
  mutable std::mutex mutex_;
  size_t allocated_ = 0;
  std::vector<std::function<void(size_t)>> allocation_hooks_;

  void set_allocated(size_t new_total) {
    allocated_ = new_total;
    for (auto& hook : allocation_hooks_) {
      hook(allocated_);
    }
  }
};

inline DeviceAllocator& HostAllocator() { return DeviceAllocator::instance(getHost()); }

inline DeviceAllocator& GPUAllocator(int device_id = 0) {
  return DeviceAllocator::instance(getGPU(device_id));
}

}  // namespace tunx