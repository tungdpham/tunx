#pragma once

#include <memory>
#include <unordered_map>

#include "device/device.hpp"

namespace tunx {
class CPUDevice : public Device {
public:
  CPUDevice(int id);
  virtual ~CPUDevice();

  CPUDevice(CPUDevice &&other) noexcept;
  CPUDevice &operator=(CPUDevice &&other) noexcept;

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
  virtual void create_flow(flowHandle_t handle) const override;
  virtual Flow *get_flow(flowHandle_t handle) const override;

private:
  int id_;
  mutable std::unordered_map<flowHandle_t, std::unique_ptr<CPUFlow>> flows_;
};

}  // namespace tunx