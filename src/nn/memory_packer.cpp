/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/memory_packer.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "device/dptr.hpp"

namespace tunx {

TrackingAllocator::TrackingAllocator(IAllocator* backend_alloc)
    : backend_alloc_(backend_alloc) {}

void TrackingAllocator::set_current_edge(const std::string& uid) {
  current_edge_uid_ = uid;
  current_alloc_index_ = 0;
}

void TrackingAllocator::step() { current_step_++; }

dptr TrackingAllocator::allocate(size_t size) {
  if (size == 0) return dptr(nullptr);

  // align size to 256
  size_t aligned_size = (size + 255) & ~255;

  std::string iid = current_edge_uid_ + "_" + std::to_string(current_alloc_index_++);

  TensorAllocation alloc;
  alloc.iid = iid;
  alloc.size = aligned_size;
  alloc.start_step = current_step_;
  alloc.end_step = -1;  // -1 means not freed yet

  size_t alloc_idx = allocations_.size();
  allocations_.push_back(alloc);
  active_allocs_[iid] = alloc_idx;

  dptr orig_ptr = backend_alloc_->allocate(aligned_size);
  void* ptr = orig_ptr.get<void>();
  auto storage = std::make_shared<device_storage>(
      device(), ptr, aligned_size, [this, alloc_idx, orig_ptr]() mutable {
        this->allocations_[alloc_idx].end_step = this->current_step_;
        orig_ptr = dptr(nullptr);  // release orig_ptr
      });
  return dptr(storage, 0, aligned_size);
}

void TrackingAllocator::clear() { backend_alloc_->clear(); }

void TrackingAllocator::ensure(size_t size) { backend_alloc_->ensure(size); }

size_t TrackingAllocator::reserved() const { return backend_alloc_->reserved(); }

size_t TrackingAllocator::allocated() const { return backend_alloc_->allocated(); }

size_t TrackingAllocator::unused() const { return backend_alloc_->unused(); }

void TrackingAllocator::evict_unused() { backend_alloc_->evict_unused(); }

size_t TrackingAllocator::add_allocation_hook(std::function<void(size_t)> hook) {
  return backend_alloc_->add_allocation_hook(hook);
}

bool TrackingAllocator::remove_allocation_hook(size_t hook_id) {
  return backend_alloc_->remove_allocation_hook(hook_id);
}

Device& TrackingAllocator::device() const { return backend_alloc_->device(); }

const std::vector<TensorAllocation>& TrackingAllocator::get_allocations() const {
  return allocations_;
}

// --- MemoryPacker ---

MemoryPacker::PackResult MemoryPacker::pack(std::vector<TensorAllocation> allocations) {
  int max_step = 0;
  for (auto& a : allocations) {
    if (a.end_step == -1) {
      a.end_step = 1000000;
    }
    if (a.end_step > max_step && a.end_step != 1000000) max_step = a.end_step;
    if (a.start_step > max_step) max_step = a.start_step;
  }

  std::sort(allocations.begin(), allocations.end(),
            [](const TensorAllocation& a, const TensorAllocation& b) { return a.size > b.size; });

  std::map<std::string, size_t> offsets;
  size_t peak_memory = 0;

  struct PlacedAllocation {
    size_t offset;
    size_t size;
    int start_step;
    int end_step;
  };
  std::vector<PlacedAllocation> placed;

  for (const auto& alloc : allocations) {
    std::vector<size_t> candidate_offsets = {0};
    for (const auto& p : placed) {
      candidate_offsets.push_back(p.offset + p.size);
    }
    std::sort(candidate_offsets.begin(), candidate_offsets.end());
    candidate_offsets.erase(std::unique(candidate_offsets.begin(), candidate_offsets.end()),
                            candidate_offsets.end());

    size_t best_offset = (size_t)-1;
    size_t best_gap_size = (size_t)-1;

    for (size_t offset : candidate_offsets) {
      bool overlap = false;
      size_t gap_end = (size_t)-1;
      for (const auto& p : placed) {
        if (alloc.start_step <= p.end_step && alloc.end_step >= p.start_step) {
          if (offset < p.offset + p.size && offset + alloc.size > p.offset) {
            overlap = true;
            break;
          }
          if (p.offset >= offset) {
            if (p.offset < gap_end) {
              gap_end = p.offset;
            }
          }
        }
      }
      if (!overlap) {
        size_t gap_size = (gap_end == (size_t)-1) ? (size_t)-1 : (gap_end - offset);
        if (gap_size >= alloc.size) {
          if (best_gap_size == (size_t)-1 || gap_size < best_gap_size) {
            best_offset = offset;
            best_gap_size = gap_size;
          }
        }
      }
    }

    if (best_offset == (size_t)-1) {
      best_offset = 0;
    }

    offsets[alloc.iid] = best_offset;
    placed.push_back({best_offset, alloc.size, alloc.start_step, alloc.end_step});
    if (best_offset + alloc.size > peak_memory) {
      peak_memory = best_offset + alloc.size;
    }
  }

  PackResult result;
  result.offsets = offsets;
  result.peak_memory = peak_memory;

  std::ofstream out("memory_placement.csv");
  if (out.is_open()) {
    out << "edge_uid,alloc_step,free_step,size,ffd_offset\n";
    for (const auto& alloc : allocations) {
      std::string iid = alloc.iid;
      size_t last_underscore = iid.find_last_of('_');
      std::string edge_uid =
          (last_underscore != std::string::npos) ? iid.substr(0, last_underscore) : iid;
      out << edge_uid << "," << alloc.start_step << "," << alloc.end_step << "," << alloc.size
          << "," << offsets[alloc.iid] << "\n";
    }
  }

  return result;
}

PackedAllocator::PackedAllocator(IAllocator* backend_alloc, size_t peak_memory,
                                 const std::map<std::string, size_t>& offsets)
    : backend_alloc_(backend_alloc),
      peak_memory_(peak_memory),
      offsets_(offsets) {
  if (backend_alloc == nullptr) {
    throw std::runtime_error("Received null backend allocator in PackedAllocator constructor");
  }
}

PackedAllocator::~PackedAllocator() { clear(); }

void PackedAllocator::set_current_edge(const std::string& uid) {
  current_edge_uid_ = uid;
  current_alloc_index_ = 0;
}

void PackedAllocator::ensure(size_t size) {
  if (!is_ensured_ && peak_memory_ > 0) {
    backend_alloc_->evict_unused();
    base_dptr_ = backend_alloc_->allocate(peak_memory_);
    is_ensured_ = true;
  }
}

dptr PackedAllocator::create_dptr(size_t offset, size_t size) {
  void* ptr = base_dptr_.get<char>() + offset;
  std::shared_ptr<device_storage> storage = std::make_shared<device_storage>(
      base_dptr_.device(), ptr, size, [this, size]() { this->current_allocated -= size; });
  current_allocated += size;
  return dptr(storage, 0, size);
}

dptr PackedAllocator::allocate(size_t size) {
  if (size == 0) return dptr(nullptr);
  ensure(size);

  size_t aligned_size = (size + 255) & ~255;

  std::string iid = current_edge_uid_ + "_" + std::to_string(current_alloc_index_++);
  auto it = offsets_.find(iid);
  if (it == offsets_.end()) {
    std::cerr << "PackedAllocator: Missed IID " << iid
              << " during allocation. Falling back to backend allocator.\n";
    return backend_alloc_->allocate(aligned_size);
  }
  size_t offset = it->second;
  if (offset + aligned_size > base_dptr_.capacity()) {
    std::cerr << "PackedAllocator error: offset " << offset << " + aligned_size " << aligned_size
              << " > capacity " << base_dptr_.capacity() << " for iid " << iid << "\n";
  }
  return create_dptr(offset, aligned_size);
}

void PackedAllocator::clear() {
  base_dptr_ = dptr(nullptr);
  is_ensured_ = false;
}

size_t PackedAllocator::reserved() const { return is_ensured_ ? peak_memory_ : 0; }

size_t PackedAllocator::allocated() const { return is_ensured_ ? peak_memory_ : 0; }

size_t PackedAllocator::unused() const { return 0; }

void PackedAllocator::evict_unused() {}

size_t PackedAllocator::add_allocation_hook(std::function<void(size_t)> hook) { return 0; }

bool PackedAllocator::remove_allocation_hook(size_t hook_id) { return false; }

Device& PackedAllocator::device() const { return backend_alloc_->device(); }

}  // namespace tunx
