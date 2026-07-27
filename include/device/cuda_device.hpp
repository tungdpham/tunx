#pragma once

#include <functional>
#ifdef TUNX_USE_CUDA

#include "device/device.hpp"
#include "device/stream.hpp"

namespace tunx {

class CUDADevice : public Device {
public:
  CUDADevice(int id);
  virtual ~CUDADevice();

  CUDADevice(CUDADevice &&other) noexcept;
  CUDADevice &operator=(CUDADevice &&other) noexcept;

  CUDADevice(const CUDADevice &) = delete;
  CUDADevice &operator=(const CUDADevice &) = delete;

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
  static void launch(Device &device, stream s, std::function<void(cudaStream_t)> func);

private:
  int id_;
  stream default_stream_;
};

class cuda_stream : public stream_impl {
private:
  CUDADevice *device_;
  cudaStream_t stream_;

public:
  cuda_stream() = default;
  explicit cuda_stream(CUDADevice &device)
      : device_(&device) {
    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to create CUDA stream: " +
                               std::string(cudaGetErrorString(err)));
    }
  }
  cuda_stream(CUDADevice &device, cudaStream_t stream)
      : device_(&device),
        stream_(stream) {}

  ~cuda_stream() { cudaStreamDestroy(stream_); }

  operator cudaStream_t() { return stream_; }

  void sync() override {
    cudaError_t err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to synchronize CUDA stream: " +
                               std::string(cudaGetErrorString(err)));
    }
  }

  Device *device() const override { return device_; }
};

}  // namespace tunx

#endif
