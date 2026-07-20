/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstring>
#include <string>

#include "common/endian.hpp"
#include "device/flow.hpp"
#include "device_type.hpp"

namespace tunx {

class Device {
public:
  Device(int id);
  virtual ~Device();
  Device(Device &&other) noexcept;
  Device &operator=(Device &&other) noexcept;
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;

  bool operator==(const Device &other) const;

  int get_id() const;

  virtual DeviceType device_type() const = 0;
  virtual std::string get_name() const = 0;
  virtual size_t get_total_memory() const = 0;
  virtual size_t get_available_memory() const = 0;
  virtual void *allocate_memory(size_t size) const = 0;
  virtual void deallocate_memory(void *ptr) const = 0;
  virtual void *allocate_aligned_memory(size_t size, size_t alignment) const = 0;
  virtual void deallocate_aligned_memory(void *ptr) const = 0;
  virtual Endianness get_endianness() const = 0;
  virtual void create_flow(flowHandle_t handle) const = 0;
  virtual Flow *get_flow(flowHandle_t handle) const = 0;

private:
  int id_;
};

}  // namespace tunx