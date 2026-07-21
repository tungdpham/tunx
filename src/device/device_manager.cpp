#include "device/device_manager.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>

#include "device/cpu_device.hpp"
#include "device/cuda_device.hpp"
#include "device/device_type.hpp"
#ifdef TUNX_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace tunx {
DeviceManager DeviceManager::instance_;

DeviceManager::DeviceManager() { discover(); }

DeviceManager::~DeviceManager() = default;

DeviceManager &DeviceManager::instance() { return instance_; }

void DeviceManager::discover() {
  clear();

  try {
    auto cpu_device = std::make_unique<CPUDevice>(0);
    add(std::move(cpu_device));
  } catch (const std::exception &e) {
    std::cerr << "Failed to create CPU device: " << e.what() << std::endl;
  }

#ifdef TUNX_USE_CUDA
  // Discover CUDA devices
  int cuda_device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&cuda_device_count);

  if (err == cudaSuccess && cuda_device_count > 0) {
    for (int i = 0; i < cuda_device_count; ++i) {
      try {
        // Get device properties for logging
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        auto cuda_device = std::make_unique<CUDADevice>(i);
        add(std::move(cuda_device));
        std::cout << "Discovered CUDA device with ID: " << i << " (CUDA Device " << i << ": "
                  << prop.name << ")" << std::endl;
      } catch (const std::exception &e) {
        std::cerr << "Failed to create CUDA device " << i << ": " << e.what() << std::endl;
      }
    }
  } else {
    std::cout << "No CUDA devices found or CUDA not available. Error: " << cudaGetErrorString(err)
              << " (" << err << ")" << std::endl;
  }
#else
  std::cout << "CUDA support not compiled in" << std::endl;
#endif
  std::cout << "Default devices initialized" << std::endl;
}

void DeviceManager::add(std::unique_ptr<Device> device) {
  DeviceID device_id = {device->device_type(), device->get_id()};
  devices_.emplace(device_id, std::move(device));
}

void DeviceManager::remove(DeviceType type, int id) { devices_.erase({type, id}); }

void DeviceManager::remove(DeviceID device_id) { devices_.erase(device_id); }

void DeviceManager::clear() { devices_.clear(); }

Device &DeviceManager::get(DeviceType type, int id) const {
  auto it = devices_.find(DeviceID{type, id});
  if (it != devices_.end()) {
    return *(it->second);
  }
  throw std::runtime_error("Device with the given ID not found");
}

Device &DeviceManager::get(DeviceID device_id) const {
  auto it = devices_.find(device_id);
  if (it != devices_.end()) {
    return *(it->second);
  }
  throw std::runtime_error("Device with the given ID not found");
}

Vec<DeviceID> DeviceManager::get_all() const {
  Vec<DeviceID> ids;
  ids.reserve(devices_.size());
  for (const auto &pair : devices_) {
    ids.push_back(pair.first);
  }
  return ids;
}

bool DeviceManager::has(DeviceType type, int id) const {
  return devices_.find({type, id}) != devices_.end();
}

bool DeviceManager::has(DeviceID device_id) const {
  return devices_.find(device_id) != devices_.end();
}

void initializeDefaultDevices() {
  DeviceManager &manager = DeviceManager::instance();
  manager.discover();
}

Device &getGPU(size_t id) {
  DeviceManager &manager = DeviceManager::instance();
  return manager.get(DeviceType::CUDA, id);
  throw std::runtime_error("Requested CUDA index not found");
}

Device &getHost() {
  DeviceManager &manager = DeviceManager::instance();
  return manager.get(DeviceType::CPU, 0);
}

}  // namespace tunx