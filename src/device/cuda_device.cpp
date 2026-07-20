#ifdef TUNX_USE_CUDA

#include "device/cuda_device.hpp"

#include <cuda_runtime.h>
#include <cudnn_graph.h>
#include <nvml.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace tunx {

CUDADevice::CUDADevice(int id)
    : Device(id), id_(id) {
  cudaError_t err = cudaSetDevice(id);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to set CUDA device " + std::to_string(id) + ": " +
                             cudaGetErrorString(err));
  }
  nvmlInit_v2();
  create_flow(defaultFlowHandle);
}

CUDADevice::~CUDADevice() = default;

CUDADevice::CUDADevice(CUDADevice &&other) noexcept
    : Device(std::move(other)), id_(other.id_), flows_(std::move(other.flows_)) {
  other.id_ = -1;
}

CUDADevice &CUDADevice::operator=(CUDADevice &&other) noexcept {
  if (this != &other) {
    Device::operator=(std::move(other));
    id_ = other.id_;
    flows_ = std::move(other.flows_);
    other.id_ = -1;
  }
  return *this;
}

DeviceType CUDADevice::device_type() const {
  return DeviceType::CUDA;
}

std::string CUDADevice::get_name() const {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, id_);
  return std::string(prop.name);
}

size_t CUDADevice::get_total_memory() const {
  size_t total_mem = 0;
  cudaError_t err = cudaMemGetInfo(nullptr, &total_mem);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to get total CUDA memory: " +
                             std::string(cudaGetErrorString(err)));
  }
  return total_mem;
}

size_t CUDADevice::get_available_memory() const {
  size_t free_mem = 0;
  cudaError_t err = cudaMemGetInfo(&free_mem, nullptr);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to get available CUDA memory: " +
                             std::string(cudaGetErrorString(err)));
  }
  return free_mem;
}

void *CUDADevice::allocate_memory(size_t size) const {
  void *ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, size);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to allocate CUDA memory: " +
                             std::string(cudaGetErrorString(err)));
  }
  return ptr;
}

void CUDADevice::deallocate_memory(void *ptr) const {
  if (ptr != nullptr) {
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to free CUDA memory: " +
                               std::string(cudaGetErrorString(err)));
    }
  }
}

void *CUDADevice::allocate_aligned_memory(size_t size, size_t alignment) const {
  (void)alignment;
  return allocate_memory(size);
}

void CUDADevice::deallocate_aligned_memory(void *ptr) const {
  deallocate_memory(ptr);
}

Endianness CUDADevice::get_endianness() const {
  return Endianness::LITTLE;
}

void CUDADevice::create_flow(flowHandle_t handle) const {
  if (flows_.find(handle) == flows_.end()) {
    flows_[handle] = std::make_unique<CUDAFlow>();
  }
}

Flow *CUDADevice::get_flow(flowHandle_t handle) const {
  auto it = flows_.find(handle);
  if (it == flows_.end()) {
    std::cerr << "WARN: Creating new CUDAFlow with ID: " << handle
              << ". Are we using the right flow?" << std::endl;
    flows_[handle] = std::make_unique<CUDAFlow>();
    return flows_[handle].get();
  } else {
    return it->second.get();
  }
}

}  // namespace tunx

#endif  // TUNX_USE_CUDA
