/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <fmt/core.h>

#include <stdexcept>
#include <type_traits>

#include "common/archiver.hpp"
#include "common/endian.hpp"
#include "device/device.hpp"
#include "device/dptr.hpp"
#include "device/stream.hpp"
#include "ops/ops.hpp"

namespace tunx {
class Sizer : public IArchiver<Sizer> {
private:
  size_t size_ = 0;

public:
  template <typename T>
  void archive_impl(const T* data, size_t count, Device& device) {
    size_ += sizeof(T) * count;
  }

  size_t size() const { return size_; }

  void reset() { size_ = 0; }
};

// For serialization
class Writer : public IArchiver<Writer> {
private:
  dptr buffer_;
  size_t offset_;

public:
  Writer(dptr& buffer)
      : buffer_(buffer),
        offset_(0) {}

  template <typename T>
  void archive_impl(const T* data, size_t count, Device& device) {
    static_assert(std::is_trivially_copyable<T>::value, "...");
    if (offset_ + sizeof(T) * count > buffer_.capacity()) {
      throw std::runtime_error(fmt::format("Writer overflow: Offset {}, Size: {}, Capacity: {}",
                                           offset_, sizeof(T) * count, buffer_.capacity()));
    }
    const dptr src(const_cast<T*>(data), sizeof(T) * count, device);
    ops::cd_copy<unsigned char>(src, buffer_ + offset_, sizeof(T) * count, nullptr);
    offset_ += sizeof(T) * count;
  }

  size_t bytes_written() const { return offset_; }
};

// Reader - For deserialization
class Reader : public IArchiver<Reader> {
private:
  const dptr& buffer_;
  size_t offset_;
  Endianness endianness_;

public:
  Reader(const dptr& buffer)
      : buffer_(buffer),
        offset_(0),
        endianness_(host_endianness) {}

  template <typename T>
  void archive_impl(T* data, size_t count, Device& device) {
    dptr dst(data, sizeof(T) * count, device);
    ops::cd_copy<unsigned char>(buffer_ + offset_, dst, sizeof(T) * count, nullptr);
    if (endianness_ != device.get_endianness()) {
      ops::bswap<T>(dst, dst, count, nullptr);
    }
    offset_ += sizeof(T) * count;
  }

  void set_endianess(Endianness endianness) { endianness_ = endianness; }

  size_t bytes_read() const { return offset_; }
};

}  // namespace tunx
