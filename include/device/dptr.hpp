/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "common/blob.hpp"
#include "device/device.hpp"
#include "device/device_type.hpp"
#include "device/sref.hpp"

namespace tunx {

constexpr size_t DEFAULT_ALIGNMENT = 256;

struct device_storage {
private:
  sref<Device> device_;
  void *ptr_;
  size_t capacity_;
  std::function<void()> deleter_;

public:
  device_storage(sref<Device> device, void *ptr, size_t capacity, std::function<void()> deleter)
      : device_(device),
        ptr_(ptr),
        capacity_(capacity),
        deleter_(deleter) {}

  device_storage(const device_storage &other) = delete;
  device_storage &operator=(const device_storage &other) = delete;
  device_storage(device_storage &&other) = default;
  device_storage &operator=(device_storage &&other) = default;

  ~device_storage() { deleter_(); }

  sref<Device> device() const { return device_; }
  void *data() const { return ptr_; }
  size_t capacity() const { return capacity_; }
};

// device pointer. Shares ownership of device storage. True owners have offset 0.
class dptr {
protected:
  std::shared_ptr<device_storage> storage_;
  size_t offset_;
  size_t capacity_;

public:
  dptr(std::shared_ptr<device_storage> storage = nullptr, size_t offset = 0, size_t capacity = 0)
      : storage_(storage),
        offset_(offset),
        capacity_(capacity) {
    if (storage_ && (offset_ + capacity_ > storage_->capacity())) {
      throw std::out_of_range("Bro what? dptr offset out of range in dptr constructor");
    }
  }

  dptr(void *ptr, size_t byte_size, Device &device) {
    if (ptr == nullptr && byte_size > 0) {
      throw std::invalid_argument("Cannot create dptr with null pointer and non-zero size");
    }
    auto storage = std::make_shared<device_storage>(device, ptr, byte_size, []() {});
    storage_ = storage;
    offset_ = 0;
    capacity_ = byte_size;
  }

  dptr(std::nullptr_t)
      : storage_(nullptr),
        offset_(0),
        capacity_(0) {}

  operator bool() const { return storage_ != nullptr && storage_->data() != nullptr; }

  Device &device() const { return storage_->device(); }

  DeviceType device_type() const {
    if (!storage_) {
      return DeviceType::UNKNOWN;
    }
    return storage_->device()->device_type();
  }

  size_t capacity() const { return capacity_; }

  template <typename T = void>
  T *get() {
    if (!storage_) {
      return nullptr;
    }
    return static_cast<T *>(
        static_cast<void *>(static_cast<uint8_t *>(storage_->data()) + offset_));
  }

  template <typename T = void>
  const T *get() const {
    if (!storage_) {
      return nullptr;
    }
    return static_cast<const T *>(
        static_cast<const void *>(static_cast<const uint8_t *>(storage_->data()) + offset_));
  }

  dptr span(size_t offset, size_t span_size) {
    if (offset + span_size > capacity_) {
      throw std::out_of_range("dptr span size out of range");
    }
    return dptr(storage_, offset_ + offset, span_size);
  }

  const dptr span(size_t offset, size_t span_size) const {
    if (offset + span_size > capacity_) {
      throw std::out_of_range("dptr span size out of range");
    }
    return dptr(storage_, offset_ + offset, span_size);
  }

  dptr operator+(size_t offset) const {
    if (offset > capacity_) {
      throw std::out_of_range("dptr offset out of range");
    }
    return dptr(storage_, offset_ + offset, capacity_ - offset);
  }
};

template <typename Archiver>
void archive(Archiver &archiver, const dptr &dptr) {
  archiver(static_cast<uint64_t>(dptr.capacity()));
  archiver(make_blob(dptr.get<unsigned char>(), dptr.capacity(), dptr.device()));
}

inline dptr make_dptr(sref<Device> device, size_t byte_size, size_t alignment = DEFAULT_ALIGNMENT) {
  if (byte_size == 0) {
    return dptr(nullptr);
  }
  void *ptr = device->allocate_aligned_memory(byte_size, alignment);
  if (!ptr) {
    throw std::runtime_error("Bad Alloc");
  }
  auto storage = std::make_shared<device_storage>(
      device, ptr, byte_size, [&device, &ptr]() { device->deallocate_aligned_memory(ptr); });
  return dptr(storage, 0, byte_size);
}

template <typename T>
inline dptr make_dptr_t(sref<Device> device, size_t count, size_t alignment = DEFAULT_ALIGNMENT) {
  return make_dptr(device, count * sizeof(T), alignment);
}

}  // namespace tunx