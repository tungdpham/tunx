#include "device/cpu_device.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

namespace tunx {

CPUDevice::CPUDevice(int id)
    : Device(id) {
  create_flow(defaultFlowHandle);
}

CPUDevice::~CPUDevice() {}

CPUDevice::CPUDevice(CPUDevice &&other) noexcept
    : Device(std::move(other)),
      flows_(std::move(other.flows_)) {}

CPUDevice &CPUDevice::operator=(CPUDevice &&other) noexcept {
  if (this != &other) {
    Device::operator=(std::move(other));
    flows_ = std::move(other.flows_);
  }
  return *this;
}

DeviceType CPUDevice::device_type() const { return DeviceType::CPU; }

std::string CPUDevice::get_name() const { return "CPU Device " + std::to_string(id_); }

size_t CPUDevice::get_total_memory() const {
#ifdef __linux__
  std::ifstream meminfo("/proc/meminfo");
  if (!meminfo.is_open()) {
    return 0;
  }

  std::string line;
  while (std::getline(meminfo, line)) {
    if (line.substr(0, 9) == "MemTotal:") {
      std::istringstream iss(line);
      std::string key;
      size_t value;
      std::string unit;

      if (iss >> key >> value >> unit) {
        // Value is in kB, convert to bytes
        return value * 1024;
      }
    }
  }
  return 0;
#elif defined(_WIN32)
  // Windows implementation
  MEMORYSTATUSEX memStatus;
  memStatus.dwLength = sizeof(memStatus);
  if (GlobalMemoryStatusEx(&memStatus)) {
    return static_cast<size_t>(memStatus.ullTotalPhys);
  }
  return 0;
#elif defined(__APPLE__)
  // macOS implementation
  int64_t physical_memory;
  size_t length = sizeof(physical_memory);
  if (sysctlbyname("hw.memsize", &physical_memory, &length, nullptr, 0) == 0) {
    return static_cast<size_t>(physical_memory);
  }
  return 0;
#else
  // Fallback for other platforms
  return 0;
#endif
}

size_t CPUDevice::get_available_memory() const {
#ifdef __linux__
  std::ifstream meminfo("/proc/meminfo");
  if (!meminfo.is_open()) {
    return 0;
  }

  std::string line;
  while (std::getline(meminfo, line)) {
    if (line.substr(0, 13) == "MemAvailable:") {
      std::istringstream iss(line);
      std::string key;
      size_t value;
      std::string unit;

      if (iss >> key >> value >> unit) {
        // Value is in kB, convert to bytes
        return value * 1024;
      }
    }
  }
  return 0;
#elif defined(_WIN32)
  // Windows implementation
  MEMORYSTATUSEX memStatus;
  memStatus.dwLength = sizeof(memStatus);
  if (GlobalMemoryStatusEx(&memStatus)) {
    return static_cast<size_t>(memStatus.ullAvailPhys);
  }
  return 0;
#elif defined(__APPLE__)
  // macOS implementation
  vm_size_t page_size;
  vm_statistics64_data_t vm_stat;
  mach_msg_type_number_t host_size = sizeof(vm_stat) / sizeof(natural_t);

  host_page_size(mach_host_self(), &page_size);
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm_stat, &host_size) ==
      KERN_SUCCESS) {
    return static_cast<size_t>(vm_stat.free_count * page_size);
  }
  return 0;
#else
  // Fallback for other platforms
  return 0;
#endif
}

void *CPUDevice::allocate_memory(size_t size) const { return std::malloc(size); }

void CPUDevice::deallocate_memory(void *ptr) const { std::free(ptr); }

void *CPUDevice::allocate_aligned_memory(size_t size, size_t alignment) const {
#ifdef _WIN32
  return _aligned_malloc(size, alignment);
#else
#ifndef TUNX_USE_CUDA
  // POSIX aligned_alloc requires size to be a multiple of alignment
  size_t adjusted_size = ((size + alignment - 1) / alignment) * alignment;
  return std::aligned_alloc(alignment, adjusted_size);
#else
  void *ptr = nullptr;
  cudaError_t err = cudaMallocHost(&ptr, size);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to allocate pinned host memory: " +
                             std::string(cudaGetErrorString(err)));
  }
  return ptr;
  (void)alignment;  // Unused parameter
#endif
#endif
}

void CPUDevice::deallocate_aligned_memory(void *ptr) const {
#ifdef _WIN32
  _aligned_free(ptr);
#else
#ifndef TUNX_USE_CUDA
  std::free(ptr);
#else
  cudaError_t err = cudaFreeHost(ptr);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to free pinned host memory: " +
                             std::string(cudaGetErrorString(err)));
  }
#endif
#endif
}

Endianness CPUDevice::get_endianness() const {
  uint16_t num = 0x1;
  char *numPtr = reinterpret_cast<char *>(&num);
  return (numPtr[0] == 1) ? Endianness::LITTLE : Endianness::BIG;
}

void CPUDevice::create_flow(flowHandle_t handle) const {
  if (flows_.find(handle) == flows_.end()) {
    flows_[handle] = std::make_unique<CPUFlow>();
  }
}

Flow *CPUDevice::get_flow(flowHandle_t handle) const {
  if (flows_.find(handle) == flows_.end()) {
    std::cerr << "WARN: Creating new CPUFlow with ID: " << handle
              << ". Are we using the right flow?" << std::endl;
    flows_[handle] = std::make_unique<CPUFlow>();
  }
  return flows_[handle].get();
}

}  // namespace tunx