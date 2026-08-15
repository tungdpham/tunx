/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <functional>

#include "device/device.hpp"
#include "device/dptr.hpp"

namespace tunx {

// Allocators should return a dptr whose internal storage is reclaimed by the allocator itself
class IAllocator {
public:
  virtual ~IAllocator() = default;

  // allocate a ptr with byte size capacity == size
  virtual dptr allocate(size_t size) = 0;

  // clear all cached memory blocks in the allocator
  virtual void clear() = 0;

  // reserve some memory for future use, so that subsequent allocate() can reuse them without going
  // through device allocation path. This is optional and can be a no-op for some allocators.
  virtual void ensure(size_t size) = 0;

  virtual size_t reserved() const = 0;   // amount of memory the allocator is holding onto
  virtual size_t allocated() const = 0;  // amount of memory the allocator has allocated from device
  virtual size_t unused() const = 0;     // amount of free memory the allocator has
  virtual void evict_unused() = 0;       // evict all unused blocks that is evictable

  virtual size_t add_allocation_hook(std::function<void(size_t)> hook) = 0;
  virtual bool remove_allocation_hook(size_t hook_id) = 0;

  virtual Device &device() const = 0;
};
}  // namespace tunx