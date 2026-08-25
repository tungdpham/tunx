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
#include <string>
#include <vector>

#include "device/device.hpp"
#include "device/dptr.hpp"
#include "device/iallocator.hpp"

namespace tunx {

struct TensorAllocation {
  std::string iid;
  size_t size;
  int start_step;
  int end_step;
};

class MemoryPacker {
public:
  struct PackResult {
    std::map<std::string, size_t> offsets;
    size_t peak_memory = 0;
  };

  static PackResult pack(std::vector<TensorAllocation> allocations);
};

class PackedAllocator : public IAllocator, std::enable_shared_from_this<PackedAllocator> {
  PackedAllocator(size_t peak_memory, const std::map<std::string, size_t>& offsets);

public:
  static std::shared_ptr<PackedAllocator> create(size_t peak_memory,
                                                 const std::map<std::string, size_t>& offsets) {
    return std::shared_ptr<PackedAllocator>(new PackedAllocator(peak_memory, offsets));
  }

  ~PackedAllocator() override;

  void set_backend_allocator(IAllocator* allocator);
  void set_current_edge(const std::string& uid);

  dptr allocate(size_t size) override;
  void clear() override;
  void ensure(size_t size) override;
  size_t reserved() const override;
  size_t allocated() const override;
  size_t unused() const override;
  void evict_unused() override;
  size_t add_allocation_hook(std::function<void(size_t)> hook) override;
  bool remove_allocation_hook(size_t hook_id) override;
  Device& device() const override;
  size_t peak_memory() const { return peak_memory_; }

private:
  IAllocator* backend_alloc_;
  size_t peak_memory_;
  std::map<std::string, size_t> offsets_;

  std::string current_edge_uid_;
  int current_alloc_index_ = 0;
  size_t current_allocated_ = 0;

  dptr create_dptr(size_t offset, size_t size);

  dptr base_dptr_;
  bool is_ensured_ = false;
  size_t next_hook_id_ = 1;
  std::map<size_t, std::function<void(size_t)>> hooks_;
};

}  // namespace tunx
