/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/dptr.hpp"
#include "device/iallocator.hpp"

#ifdef TUNX_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace tunx {

class ManagedAllocator : public IAllocator {
public:
  ManagedAllocator(Device& device)
      : device_(device) {}
  ~ManagedAllocator() {}
  ManagedAllocator(const ManagedAllocator&) = delete;
  ManagedAllocator& operator=(const ManagedAllocator&) = delete;

  static ManagedAllocator& instance(Device& device, stream s = nullptr) {
    static std::mutex registry_mutex;
    static std::map<std::pair<Device*, stream>, std::unique_ptr<ManagedAllocator>> registry;
    std::lock_guard<std::mutex> lock(registry_mutex);
    std::pair<Device*, stream> key = {&device, s};
    auto it = registry.find(key);
    if (it == registry.end()) {
      registry.emplace(key, std::make_unique<ManagedAllocator>(device));
    }
    return *registry[key];
  }

  dptr allocate(size_t size) override {
    if (size == 0) {
      auto storage = std::make_shared<device_storage>(device_, nullptr, 0, []() {});
      return dptr(storage, 0, 0);
    }
#ifdef TUNX_USE_CUDA
    void* ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&ptr, size);
    if (err != cudaSuccess) {
      throw std::runtime_error("cudaMallocManaged failed");
    }

    set_allocated(allocated_ + size);

    auto storage = std::make_shared<device_storage>(device_, ptr, size, [this, ptr, size]() {
      cudaFree(ptr);
      std::lock_guard<std::mutex> lock(mutex_);
      set_allocated(allocated_ - size);
    });

    return dptr(storage, 0, size);
#else
    throw std::runtime_error("ManagedAllocator requires TUNX_USE_CUDA");
#endif
  }

  void clear() override {}
  void ensure(size_t size) override {}

  Device& device() const override { return *device_; }

  size_t reserved() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocated_;
  }

  size_t allocated() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocated_;
  }

  size_t unused() const override { return 0; }
  void evict_unused() override {}

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

}  // namespace tunx
