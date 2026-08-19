#include <memory>

#include "device/stream.hpp"
#ifdef TUNX_USE_CUDA

#include <cuda_runtime.h>
#include <cudnn_graph.h>
#include <nvml.h>

#include <stdexcept>
#include <string>

#include "device/cuda_device.hpp"

namespace tunx {

CUDADevice::CUDADevice(int id)
    : Device(id),
      id_(id) {
  cudaError_t err = cudaSetDevice(id);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to set CUDA device " + std::to_string(id) + ": " +
                             cudaGetErrorString(err));
  }
  nvmlInit_v2();
  cudaStream_t default_cu_stream = cudaStreamDefault;
  auto impl = std::make_shared<cuda_stream>(*this, default_cu_stream);
  default_stream_ = stream(impl);
}

CUDADevice::~CUDADevice() = default;

CUDADevice::CUDADevice(CUDADevice &&other) noexcept
    : Device(std::move(other)),
      id_(other.id_) {
  other.id_ = -1;
}

CUDADevice &CUDADevice::operator=(CUDADevice &&other) noexcept {
  if (this != &other) {
    Device::operator=(std::move(other));
    id_ = other.id_;
    other.id_ = -1;
  }
  return *this;
}

DeviceType CUDADevice::device_type() const { return DeviceType::CUDA; }

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

void CUDADevice::deallocate_aligned_memory(void *ptr) const { deallocate_memory(ptr); }

Endianness CUDADevice::get_endianness() const { return Endianness::LITTLE; }

void CUDADevice::create_stream(stream &s) {
  auto impl = std::make_shared<cuda_stream>(*this);
  s = stream(impl);
}

stream CUDADevice::default_stream() const { return default_stream_; }

void CUDADevice::launch(Device &device, stream s, std::function<void(cudaStream_t)> func) {
  cudaStream_t cu_stream = *s.as<cuda_stream>();
  func(cu_stream);
}

}  // namespace tunx

#endif  // TUNX_USE_CUDA
