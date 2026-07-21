#pragma once

#include <functional>

#include "device/device.hpp"

namespace tunx {

class CPUDevice : public Device {
public:
  CPUDevice(int id);
  virtual ~CPUDevice();

  CPUDevice(CPUDevice &&other) noexcept = default;
  CPUDevice &operator=(CPUDevice &&other) noexcept = default;

  CPUDevice(const CPUDevice &) = delete;
  CPUDevice &operator=(const CPUDevice &) = delete;

  virtual DeviceType device_type() const override;
  virtual std::string get_name() const override;
  virtual size_t get_total_memory() const override;
  virtual size_t get_available_memory() const override;
  virtual void *allocate_memory(size_t size) const override;
  virtual void deallocate_memory(void *ptr) const override;
  virtual void *allocate_aligned_memory(size_t size, size_t alignment) const override;
  virtual void deallocate_aligned_memory(void *ptr) const override;
  virtual Endianness get_endianness() const override;
  virtual void create_stream(stream &s) override;
  virtual stream default_stream() const override;
  static void launch(Device &device, stream s, std::function<void()> func);

private:
  int id_;
  stream default_stream_;
};

// Does nothing for now, but task handler per device can put it into different threads in threadpool
// in the future
class cpu_stream : public stream_impl {
public:
  cpu_stream() = default;
  cpu_stream(CPUDevice &device)
      : device_(&device) {}

  void sync() override {
    // No-op for CPU
  }

  Device *device() const override { return device_; }

private:
  CPUDevice *device_;
};

}  // namespace tunx