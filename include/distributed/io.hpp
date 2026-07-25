/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <fmt/core.h>

#include <cassert>
#include <type_traits>

#include "common/archiver.hpp"
#include "common/endian.hpp"
#include "device/device.hpp"
#include "device/dptr.hpp"
#include "device/stream.hpp"
#include "kernel/kernel.hpp"
#include "type/type.hpp"

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
  stream s_;

public:
  Writer(dptr& buffer)
      : buffer_(buffer),
        offset_(0),
        s_(nullptr) {}

  template <typename T>
  void archive_impl(const T* data, size_t count, Device& device) {
    static_assert(std::is_trivially_copyable<T>::value, "...");
    assert(offset_ + sizeof(T) * count <= buffer_.capacity() && "Writer overflow");
    const dptr src(const_cast<T*>(data), sizeof(T) * count, device);
    kernel::copy(src, buffer_ + offset_, sizeof(T) * count, s_);
    offset_ += sizeof(T) * count;
  }

  size_t bytes_written() const { return offset_; }

  void set_stream(stream s) { s_ = s; }

  stream get_stream() { return s_; }
};

// Reader - For deserialization
class Reader : public IArchiver<Reader> {
private:
  const dptr& buffer_;
  size_t offset_;
  Endianness endianness_;
  stream s_;

public:
  Reader(const dptr& buffer)
      : buffer_(buffer),
        offset_(0),
        endianness_(host_endianness),
        s_(nullptr) {}

  template <typename T>
  void archive_impl(T* data, size_t count, Device& device) {
    dptr dst(data, sizeof(T) * count, device);
    kernel::copy(buffer_ + offset_, dst, sizeof(T) * count, s_);
    if (endianness_ != device.get_endianness()) {
      kernel::bswap(dtype_of<T>(), dst, dst, count, s_);
    }
    offset_ += sizeof(T) * count;
  }

  void set_endianess(Endianness endianness) { endianness_ = endianness; }

  void set_stream(stream s) { s_ = s; }

  size_t bytes_read() const { return offset_; }
};

}  // namespace tunx
