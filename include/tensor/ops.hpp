#pragma once

#include <fmt/core.h>

#include <istream>
#include <ostream>

#include "cpu/ops.hpp"
#include "device/device_manager.hpp"
#include "device/stream.hpp"
#include "device/task.hpp"
#include "kernel/cpu/kernels.hpp"
#include "type/type.hpp"
#ifdef TUNX_USE_CUDA
#include "cuda/ops.hpp"
#include "kernel/cuda/kernels.hpp"
#endif
#include "tensor/tensor.hpp"

namespace tunx {

inline void add(const Tensor &a, const Tensor &b, Tensor &c, stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("add: All device pointers must be on the same device");
  }
  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::add, dtype, a.data_as<void>(),
                           b.data_as<void>(), c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::add, dtype, a.data_as<void>(),
                            b.data_as<void>(), c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void sub(const Tensor &a, const Tensor &b, Tensor &c, stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("sub: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::sub, dtype, a.data_as<void>(),
                           b.data_as<void>(), c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::sub, dtype, a.data_as<void>(),
                            b.data_as<void>(), c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void mul(const Tensor &a, const Tensor &b, Tensor &c, stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("mul: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::mul, dtype, a.data_as<void>(),
                           b.data_as<void>(), c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::mul, dtype, a.data_as<void>(),
                            b.data_as<void>(), c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void div(const Tensor &a, const Tensor &b, Tensor &c, stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("div: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::div, dtype, a.data_as<void>(),
                           b.data_as<void>(), c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::div, dtype, a.data_as<void>(),
                            b.data_as<void>(), c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void fmadd(const Tensor &a, const Tensor &b, Tensor &c, stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("fmadd: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::fmadd, dtype, a.data_as<void>(),
                           b.data_as<void>(), c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::fmadd, dtype, a.data_as<void>(),
                            b.data_as<void>(), c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void fmsub(const Tensor &a, const Tensor &b, Tensor &c, stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("fmsub: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::fmsub, dtype, a.data_as<void>(),
                           b.data_as<void>(), c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::fmsub, dtype, a.data_as<void>(),
                            b.data_as<void>(), c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void fnmadd(const Tensor &a, const Tensor &b, Tensor &c, stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("fnmadd: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::fnmadd, dtype, a.data_as<void>(),
                           b.data_as<void>(), c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::fnmadd, dtype, a.data_as<void>(),
                            b.data_as<void>(), c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void add_scalar(const Tensor &a, double scalar, Tensor &c, stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != c.device()) {
    throw std::runtime_error("add_scalar: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::add_scalar, dtype, a.data_as<void>(),
                           scalar, c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::add_scalar, dtype, a.data_as<void>(),
                            scalar, c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void sub_scalar(const Tensor &a, double scalar, Tensor &c, stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != c.device()) {
    throw std::runtime_error("sub_scalar: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::sub_scalar, dtype, a.data_as<void>(),
                           scalar, c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::sub_scalar, dtype, a.data_as<void>(),
                            scalar, c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void mul_scalar(const Tensor &a, double scalar, Tensor &c, stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != c.device()) {
    throw std::runtime_error("mul_scalar: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::mul_scalar, dtype, a.data_as<void>(),
                           scalar, c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::mul_scalar, dtype, a.data_as<void>(),
                            scalar, c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void div_scalar(const Tensor &a, double scalar, Tensor &c, stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != c.device()) {
    throw std::runtime_error("div_scalar: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::div_scalar, dtype, a.data_as<void>(),
                           scalar, c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::div_scalar, dtype, a.data_as<void>(),
                            scalar, c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void fill(Tensor &c, double scalar, stream stream = nullptr) {
  DType_t dtype = c.dtype();
  size_t size = c.size();
  auto &device = c.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::fill, dtype, c.data_as<void>(), scalar,
                           size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::fill, dtype, c.data_as<void>(), scalar,
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void axpy(double alpha, const Tensor &x, Tensor &y, stream stream = nullptr) {
  DType_t dtype = x.dtype();
  size_t size = x.size();
  if (x.device() != y.device()) {
    throw std::runtime_error("axpy: All device pointers must be on the same device");
  }

  auto &device = x.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::axpy, dtype, alpha, x.data_as<void>(),
                           y.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::axpy, dtype, alpha, x.data_as<void>(),
                            y.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void sqrt(const Tensor &a, Tensor &c, stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != c.device()) {
    throw std::runtime_error("sqrt: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::sqrt, dtype, a.data_as<void>(),
                           c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::sqrt, dtype, a.data_as<void>(),
                            c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void rsqrt(const Tensor &a, Tensor &c, stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != c.device()) {
    throw std::runtime_error("rsqrt: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::rsqrt, dtype, a.data_as<void>(),
                           c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::rsqrt, dtype, a.data_as<void>(),
                            c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void rcp(const Tensor &a, Tensor &c, stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != c.device()) {
    throw std::runtime_error("rcp: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::rcp, dtype, a.data_as<void>(),
                           c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::rcp, dtype, a.data_as<void>(),
                            c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void abs(const Tensor &a, Tensor &c, stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != c.device()) {
    throw std::runtime_error("abs: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::abs, dtype, a.data_as<void>(),
                           c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::abs, dtype, a.data_as<void>(),
                            c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void min(const Tensor &a, const Tensor &b, Tensor &c, stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("min: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::min, dtype, a.data_as<void>(),
                           b.data_as<void>(), c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::min, dtype, a.data_as<void>(),
                            b.data_as<void>(), c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void max(const Tensor &a, const Tensor &b, Tensor &c, stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("max: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::max, dtype, a.data_as<void>(),
                           b.data_as<void>(), c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::max, dtype, a.data_as<void>(),
                            b.data_as<void>(), c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void scalar_max(const Tensor &a, double scalar, Tensor &c, stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != c.device()) {
    throw std::runtime_error("scalar_max: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::scalar_max, dtype, a.data_as<void>(),
                           scalar, c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::scalar_max, dtype, a.data_as<void>(),
                            scalar, c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void clamp(const Tensor &a, double min_val, double max_val, Tensor &c,
                  stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != c.device()) {
    throw std::runtime_error("clamp: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::clamp, dtype, a.data_as<void>(), min_val,
                           max_val, c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::clamp, dtype, a.data_as<void>(), min_val,
                            max_val, c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void equal(const Tensor &a, const Tensor &b, Tensor &c, stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("equal: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::equal, dtype, a.data_as<void>(),
                           b.data_as<void>(), c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::equal, dtype, a.data_as<void>(),
                            b.data_as<void>(), c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void greater(const Tensor &a, const Tensor &b, Tensor &c, stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("greater: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::greater, dtype, a.data_as<void>(),
                           b.data_as<void>(), c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::greater, dtype, a.data_as<void>(),
                            b.data_as<void>(), c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void copy(const Tensor &a, Tensor &c, stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != c.device()) {
    throw std::runtime_error("copy: All device pointers must be on the same device");
  }

  if (a.data_as<void>() == nullptr || c.data_as<void>() == nullptr) {
    throw std::runtime_error("copy: Null pointer exception in copy operation");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::copy, dtype, a.data_as<void>(),
                           c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::copy, dtype, a.data_as<void>(),
                            c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

// Special copy for copying cross devices (resort to same device/host copy if applicable)
inline void cd_copy(const Tensor &a, Tensor &c, stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  auto &a_device = a.device();
  auto &c_device = c.device();
  if (a_device == c_device) {
    // same device copy
    return copy(a, c, stream);
  }
  auto a_device_type = a_device.device_type();
  auto c_device_type = c_device.device_type();

  if (a_device_type == DeviceType::CPU && c_device_type == DeviceType::CUDA) {
    // host to device copy
#ifdef TUNX_USE_CUDA
    return create_cuda_task(c_device, stream, kernel::cuda::h2d_copy, dtype, a.data_as<void>(),
                            c.data_as<void>(), size);
#else
    throw std::runtime_error("cd_copy: CUDA not enabled for CPU to CUDA copy");
#endif
  } else if (a_device_type == DeviceType::CUDA && c_device_type == DeviceType::CPU) {
    // device to host copy
#ifdef TUNX_USE_CUDA
    return create_cuda_task(a_device, stream, kernel::cuda::d2h_copy, dtype, a.data_as<void>(),
                            c.data_as<void>(), size);
#else
    throw std::runtime_error("cd_copy: CUDA not enabled for CUDA to CPU copy");
#endif
  } else {
    (void)dtype;
    (void)size;
    throw std::runtime_error("cd_copy: Unsupported device type combination");
  }
}

inline void bswap(const Tensor &a, Tensor &c, stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != c.device()) {
    throw std::runtime_error("bswap: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::bswap, dtype, a.data_as<void>(),
                           c.data_as<void>(), size);
  } else if (device_type == DeviceType::CUDA) {
#ifdef TUNX_USE_CUDA
    return create_cuda_task(device, stream, kernel::cuda::bswap, dtype, a.data_as<void>(),
                            c.data_as<void>(), size);
#else
    throw std::runtime_error("bswap: CUDA support not compiled in");
#endif
  } else {
    throw std::runtime_error("bswap: Unsupported device type");
  }
}

inline void zero(Tensor &c, stream stream = nullptr) {
  DType_t dtype = c.dtype();
  size_t size = c.size();
  auto &device = c.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::zero, dtype, c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::zero, dtype, c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline double sum(Tensor &a) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return kernel::cpu::sum(dtype, a.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return kernel::cuda::sum(dtype, a.data_as<void>(), size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline double dot_product(Tensor &a, Tensor &b) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != b.device()) {
    throw std::runtime_error("dot_product: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return kernel::cpu::dot_product(dtype, a.data_as<void>(), b.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return kernel::cuda::dot_product(dtype, a.data_as<void>(), b.data_as<void>(), size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline double norm_squared(Tensor &a) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return kernel::cpu::norm_squared(dtype, a.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return kernel::cuda::norm_squared(dtype, a.data_as<void>(), size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline double sum_squared_diff(const Tensor &a, double mean) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return kernel::cpu::sum_squared_diff(dtype, a.data_as<void>(), mean, size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return kernel::cuda::sum_squared_diff(dtype, a.data_as<void>(), mean, size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void sub_mul_scalar(const Tensor &a, double sub_scalar, double mul_scalar, Tensor &c,
                           stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != c.device()) {
    throw std::runtime_error("sub_mul_scalar: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::sub_mul_scalar, dtype, a.data_as<void>(),
                           sub_scalar, mul_scalar, c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::sub_mul_scalar, dtype, a.data_as<void>(),
                            sub_scalar, mul_scalar, c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void mul_add_scalar(const Tensor &a, double mul_scalar, double add_scalar, Tensor &c,
                           stream stream = nullptr) {
  DType_t dtype = a.dtype();
  size_t size = a.size();
  if (a.device() != c.device()) {
    throw std::runtime_error("mul_add_scalar: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::mul_add_scalar, dtype, a.data_as<void>(),
                           mul_scalar, add_scalar, c.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::mul_add_scalar, dtype, a.data_as<void>(),
                            mul_scalar, add_scalar, c.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void fill_uniform(Tensor &data, double min_val, double max_val, unsigned long long seed,
                         stream stream = nullptr) {
  DType_t dtype = data.dtype();
  size_t size = data.size();
  auto &device = data.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::fill_uniform, dtype, data.data_as<void>(),
                           size, min_val, max_val, seed);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::fill_uniform, dtype, data.data_as<void>(),
                            size, min_val, max_val, seed);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void fill_normal(Tensor &data, double mean, double stddev, unsigned long long seed,
                        stream stream = nullptr) {
  DType_t dtype = data.dtype();
  size_t size = data.size();
  auto &device = data.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::fill_normal, dtype, data.data_as<void>(),
                           size, mean, stddev, seed);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::fill_normal, dtype, data.data_as<void>(),
                            size, mean, stddev, seed);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void check_equals(const Tensor &a, const Tensor &b, bool &result, double eps = 1e-5,
                         stream stream = nullptr) {
  DType_t dtype = a.dtype();
  if (a.device() != b.device()) {
    throw std::runtime_error("check_equals: All device pointers must be on the same device");
  }
  if (a.size() != b.size()) {
    throw std::runtime_error("check_equals: All device pointers must have the same capacity");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::check_equals, dtype, a.data_as<void>(),
                           b.data_as<void>(), a.size(), result, eps);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::check_equals, dtype, a.data_as<void>(),
                            b.data_as<void>(), a.size(), result, eps);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void cast(const Tensor &a, Tensor &b, stream stream = nullptr) {
  DType_t a_dtype = a.dtype();
  DType_t b_dtype = b.dtype();
  size_t size = a.size();
  if (a.device() != b.device()) {
    throw std::runtime_error("cast: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, kernel::cpu::cast, a_dtype, b_dtype, a.data_as<void>(),
                           b.data_as<void>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, kernel::cuda::cast, a_dtype, b_dtype, a.data_as<void>(),
                            b.data_as<void>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline Tensor to_device(const Tensor &input, Device &device, stream s = nullptr) {
  Tensor result(input.shape(), input.dtype(), device);
  cd_copy(input, result, s);
  return result;
}

inline Tensor to_host(const Tensor &input, stream s = nullptr) {
  return to_device(input, getHost(), s);
}

inline Tensor clone(const Tensor &input, stream s = nullptr) {
  Tensor output = Tensor(input.shape(), input.dtype(), input.allocator());
  copy(input, output, s);
  return output;
}

inline void save(const Tensor &input, std::ostream &out) {
  if (!out) {
    throw std::runtime_error("Stream is not ready for writing");
  }

  // write dims, shape
  size_t dims = input.dims();
  DType_t dtype = input.dtype();
  out.write(reinterpret_cast<const char *>(&dtype), sizeof(DType_t));
  out.write(reinterpret_cast<const char *>(&dims), sizeof(size_t));
  out.write(reinterpret_cast<const char *>(input.shape().data()),
            input.shape().size() * sizeof(size_t));

  if (input.device_type() == DeviceType::CPU) {
    out.write(reinterpret_cast<const char *>(input.data_as<uchar>()),
              input.size() * get_dtype_size(dtype));
  } else {
    auto host_tensor = to_host(input);
    out.write(reinterpret_cast<const char *>(host_tensor.data_as<uchar>()),
              input.size() * get_dtype_size(dtype));
  }
}

inline void load(Tensor &input, std::istream &in) {
  if (!in) {
    throw std::runtime_error("Stream is not ready for reading");
  }

  // read dims, shape
  DType_t dtype;
  size_t dims;
  in.read(reinterpret_cast<char *>(&dtype), sizeof(DType_t));
  in.read(reinterpret_cast<char *>(&dims), sizeof(size_t));
  Vec<size_t> shape(dims);
  in.read(reinterpret_cast<char *>(shape.data()), dims * sizeof(size_t));

  input = Tensor(shape, dtype, input.allocator());

  if (input.device_type() == DeviceType::CPU) {
    in.read(reinterpret_cast<char *>(input.data_as<uchar>()), input.size() * get_dtype_size(dtype));
  } else {
    Tensor host_tensor(shape, dtype, DeviceAllocator::instance(getHost()));
    in.read(reinterpret_cast<char *>(host_tensor.data_as<uchar>()),
            input.size() * get_dtype_size(dtype));
    copy(host_tensor, input);
  }
}

inline void im2col(const Tensor &input, Tensor &col_data, size_t kernel_h, size_t kernel_w,
                   size_t stride_h = 1, size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0,
                   stream stream = nullptr) {
  if (col_data.device() != input.device()) {
    throw std::runtime_error("im2col: Mismatched device types between col_data and input");
  }

  auto &device = input.device();

  if (input.dtype() != col_data.dtype()) {
    throw std::runtime_error("im2col: Mismatched data types between col_data and input");
  }

  const auto &shape = input.shape();
  if (shape.size() != 4) {
    throw std::invalid_argument("im2col: Input input must be 4-dimensional (NCHW)");
  }

  DType_t dtype = input.dtype();

  size_t batch_size = shape[0];
  size_t channels = shape[1];
  size_t height = shape[2];
  size_t width = shape[3];

  size_t padded_h = height + 2 * pad_h;
  size_t padded_w = width + 2 * pad_w;
  size_t output_h = (padded_h - kernel_h) / stride_h + 1;
  size_t output_w = (padded_w - kernel_w) / stride_w + 1;

  if (input.device_type() == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::im2col, dtype, input.data_as(), col_data.data_as(),
                           batch_size, channels, height, width, kernel_h, kernel_w, stride_h,
                           stride_w, pad_h, pad_w, output_h, output_w);
  }
#ifdef TUNX_USE_CUDA
  else if (input.device_type() == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::im2col, dtype, input.data_as(),
                            col_data.data_as(), batch_size, channels, height, width, kernel_h,
                            kernel_w, stride_h, stride_w, pad_h, pad_w, output_h, output_w);
  }
#endif
  else {
    throw std::runtime_error("im2col: Unsupported device type");
  }
}

inline void col2im(const Tensor &col_data, Tensor &result_data, size_t batch_size, size_t channels,
                   size_t height, size_t width, size_t kernel_h, size_t kernel_w, size_t stride_h,
                   size_t stride_w, size_t pad_h, size_t pad_w, stream stream = nullptr) {
  if (col_data.device_type() != result_data.device_type()) {
    throw std::runtime_error("col2im: Mismatched device types between col_data and result_data");
  }

  if (col_data.dtype() != result_data.dtype()) {
    throw std::runtime_error("col2im: Mismatched data types between col_data and result_data");
  }

  DType_t dtype = col_data.dtype();
  size_t padded_h = height + 2 * pad_h;
  size_t padded_w = width + 2 * pad_w;
  size_t output_h = (padded_h - kernel_h) / stride_h + 1;
  size_t output_w = (padded_w - kernel_w) / stride_w + 1;

  auto &device = col_data.device();
  if (col_data.device_type() == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::col2im, dtype, col_data.data_as<void>(),
                           result_data.data_as<void>(), batch_size, channels, height, width,
                           kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, output_h,
                           output_w);
  }
#ifdef TUNX_USE_CUDA
  else if (col_data.device_type() == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::col2im, dtype, col_data.data_as(),
                            result_data.data_as(), batch_size, channels, height, width, kernel_h,
                            kernel_w, stride_h, stride_w, pad_h, pad_w, output_h, output_w);
  }
#endif
  else {
    throw std::runtime_error("col2im: Unsupported device type");
  }
}

inline void pad(const Tensor &input, Tensor &result, size_t pad_h, size_t pad_w, double value = 0.0,
                stream stream = nullptr) {
  if (input.device_type() != result.device_type()) {
    throw std::runtime_error("pad: Mismatched device types between input and result");
  }

  if (input.dtype() != result.dtype()) {
    throw std::runtime_error("pad: Mismatched data types between input and result");
  }

  const auto &shape = input.shape();
  if (shape.size() != 4) {
    throw std::invalid_argument("pad: Input input must be 4-dimensional (NCHW)");
  }

  DType_t dtype = input.dtype();
  size_t batch_size = shape[0];
  size_t channels = shape[1];
  size_t height = shape[2];
  size_t width = shape[3];

  auto &device = input.device();
  if (input.device_type() == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::pad, dtype, input.data_as(), result.data_as(),
                           batch_size, channels, height, width, pad_h, pad_w, value);
  }
#ifdef TUNX_USE_CUDA
  else if (input.device_type() == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::pad, dtype, input.data_as(), result.data_as(),
                            batch_size, channels, height, width, pad_h, pad_w, value);
  }
#endif
  else {
    throw std::runtime_error("pad: Unsupported device type");
  }
}

inline void unpad(const Tensor &input, Tensor &result, size_t pad_h, size_t pad_w,
                  stream stream = nullptr) {
  if (input.device_type() != result.device_type()) {
    throw std::runtime_error("unpad: Mismatched device types between input and result");
  }

  if (input.dtype() != result.dtype()) {
    throw std::runtime_error("unpad: Mismatched data types between input and result");
  }

  const auto &shape = input.shape();
  if (shape.size() != 4) {
    throw std::invalid_argument("unpad: Input input must be 4-dimensional (NCHW)");
  }

  size_t padded_height = shape[2];
  size_t padded_width = shape[3];

  if (padded_height <= 2 * pad_h || padded_width <= 2 * pad_w) {
    throw std::invalid_argument("Padding size too large for unpadding");
  }

  DType_t dtype = input.dtype();
  size_t batch_size = shape[0];
  size_t channels = shape[1];
  size_t height = padded_height - 2 * pad_h;
  size_t width = padded_width - 2 * pad_w;

  auto &device = input.device();
  if (input.device_type() == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::unpad, dtype, input.data_as(), result.data_as(),
                           batch_size, channels, height, width, pad_h, pad_w);
  }
#ifdef TUNX_USE_CUDA
  else if (input.device_type() == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::unpad, dtype, input.data_as(), result.data_as(),
                            batch_size, channels, height, width, pad_h, pad_w);
  }
#endif
  else {
    throw std::runtime_error("unpad: Unsupported device type");
  }
}

inline void crop(const Tensor &input, Tensor &result, size_t start_h, size_t start_w, size_t end_h,
                 size_t end_w, stream stream = nullptr) {
  if (input.device_type() != result.device_type()) {
    throw std::runtime_error("crop: Mismatched device types between input and result");
  }

  if (input.dtype() != result.dtype()) {
    throw std::runtime_error("crop: Mismatched data types between input and result");
  }

  const auto &shape = input.shape();
  if (shape.size() != 4) {
    throw std::invalid_argument("crop: Input input must be 4-dimensional (NCHW)");
  }

  size_t height = shape[2];
  size_t width = shape[3];

  if (end_h >= height || end_w >= width || start_h > end_h || start_w > end_w) {
    throw std::invalid_argument("Invalid crop dimensions");
  }

  DType_t dtype = input.dtype();
  size_t batch_size = shape[0];
  size_t channels = shape[1];
  size_t new_height = end_h - start_h + 1;
  size_t new_width = end_w - start_w + 1;

  auto &device = input.device();
  if (input.device_type() == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::crop, dtype, input.data_as(), result.data_as(),
                           batch_size, channels, height, width, start_h, start_w, new_height,
                           new_width);
  }
#ifdef TUNX_USE_CUDA
  else if (input.device_type() == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::crop, dtype, input.data_as(), result.data_as(),
                            batch_size, channels, height, width, start_h, start_w, new_height,
                            new_width);
  }
#endif
  else {
    throw std::runtime_error("crop: Unsupported device type");
  }
}

inline void slice_batch(const Tensor &input, Tensor &result, size_t start_batch, size_t end_batch,
                        stream stream = nullptr) {
  if (input.device_type() != result.device_type()) {
    throw std::runtime_error("slice_batch: Mismatched device types between input and result");
  }

  if (input.dtype() != result.dtype()) {
    throw std::runtime_error("slice_batch: Mismatched data types between input and result");
  }

  const auto &shape = input.shape();
  size_t batch_size = shape[0];

  if (end_batch > batch_size || start_batch > end_batch) {
    throw std::invalid_argument("Invalid batch slice range");
  }

  size_t batch_stride = 1;
  for (size_t i = 1; i < shape.size(); ++i) {
    batch_stride *= shape[i];
  }

  Vec<size_t> result_shape = shape;
  result_shape[0] = end_batch - start_batch;
  result = Tensor(result_shape, input.dtype(), input.allocator());

  size_t copy_size = (end_batch - start_batch) * batch_stride;
  DType_t dtype = input.dtype();

  auto &device = input.device();
  if (input.device_type() == DeviceType::CPU) {
    return create_cpu_task(
        device, stream, kernel::cpu::copy, dtype,
        input.data_as<char>() + start_batch * batch_stride * get_dtype_size(dtype),
        result.data_as(), copy_size);
  }
#ifdef TUNX_USE_CUDA
  else if (input.device_type() == DeviceType::CUDA) {
    return create_cuda_task(
        device, stream, kernel::cuda::copy, dtype,
        input.data_as<char>() + start_batch * batch_stride * get_dtype_size(dtype),
        result.data_as(), copy_size);
  }
#endif
  else {
    throw std::runtime_error("slice_batch: Unsupported device type");
  }
}

inline void split(const Tensor &input, Vec<Tensor> &results, size_t num_splits,
                  stream stream = nullptr) {
  const auto &shape = input.shape();
  size_t batch_size = shape[0];

  if (num_splits == 0 || num_splits > batch_size) {
    throw std::invalid_argument("Invalid number of splits");
  }

  DType_t dtype = input.dtype();

  results.clear();
  results.reserve(num_splits);
  size_t split_size = batch_size / num_splits;

  size_t batch_stride = 1;
  for (size_t j = 1; j < shape.size(); ++j) {
    batch_stride *= shape[j];
  }

  for (size_t i = 0; i < num_splits; ++i) {
    size_t start = i * split_size;
    size_t end = (i == num_splits - 1) ? batch_size : start + split_size;

    Vec<size_t> split_shape = shape;
    split_shape[0] = end - start;

    Tensor split_tensor = Tensor(split_shape, input.dtype(), input.allocator());

    size_t copy_size = (end - start) * batch_stride;

    auto &device = input.device();
    if (input.device_type() == DeviceType::CPU) {
      create_cpu_task(device, stream, kernel::cpu::copy, dtype,
                      input.data_as<char>() + start * batch_stride * get_dtype_size(dtype),
                      split_tensor.data_as(), copy_size);
    }
#ifdef TUNX_USE_CUDA
    else if (input.device_type() == DeviceType::CUDA) {
      create_cuda_task(device, stream, kernel::cuda::copy, dtype,
                       input.data_as<char>() + start * batch_stride * get_dtype_size(dtype),
                       split_tensor.data_as(), copy_size);
    }
#endif
    else {
      throw std::runtime_error("split: Unsupported device type");
    }
    results.push_back(split_tensor);
  }
}

inline void transpose_2d(const Tensor &input, Tensor &output, size_t rows, size_t cols,
                         stream stream = nullptr) {
  if (output.device() != input.device()) {
    throw std::runtime_error("transpose_2d: Input and output must be on the same device");
  }

  if (input.dtype() != output.dtype()) {
    throw std::runtime_error("transpose_2d: Mismatched data types between input and output");
  }

  DType_t dtype = input.dtype();

  auto &device = input.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::transpose_2d, dtype, input.data_as(),
                           output.data_as(), rows, cols);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::transpose_2d, dtype, input.data_as(),
                            output.data_as(), rows, cols);
  }
#endif
  else {
    throw std::runtime_error("transpose_2d: Unsupported device type");
  }
}

inline void nchw_to_cnhw(const Tensor &input, Tensor &output, size_t n, size_t c, size_t h,
                         size_t w, stream stream = nullptr) {
  if (output.device() != input.device()) {
    throw std::runtime_error("nchw_to_cnhw: Input and output must be on the same device");
  }

  if (input.dtype() != output.dtype()) {
    throw std::runtime_error("nchw_to_cnhw: Mismatched data types between input and output");
  }

  DType_t dtype = input.dtype();

  auto &device = input.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::nchw_to_cnhw, dtype, input.data_as(),
                           output.data_as(), n, c, h, w);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::nchw_to_cnhw, dtype, input.data_as(),
                            output.data_as(), n, c, h, w);
  }
#endif
  else {
    throw std::runtime_error("nchw_to_cnhw: Unsupported device type");
  }
}

inline void cnhw_to_nchw(const Tensor &input, Tensor &output, size_t n, size_t c, size_t h,
                         size_t w, stream stream = nullptr) {
  if (output.device() != input.device()) {
    throw std::runtime_error("cnhw_to_nchw: Input and output must be on the same device");
  }

  if (input.dtype() != output.dtype()) {
    throw std::runtime_error("cnhw_to_nchw: Mismatched data types between input and output");
  }

  DType_t dtype = input.dtype();

  auto &device = input.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::cnhw_to_nchw, dtype, input.data_as(),
                           output.data_as(), n, c, h, w);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::cnhw_to_nchw, dtype, input.data_as(),
                            output.data_as(), n, c, h, w);
  }
#endif
  else {
    throw std::runtime_error("cnhw_to_nchw: Unsupported device type");
  }
}

inline void print_tensor(const Tensor &input, size_t num_elements_, std::string_view label) {
  auto host_tensor = to_host(input);
  fmt::print(fmt::runtime("{}: "), label);
  for (size_t i = 0; i < num_elements_; i++) {
    DISPATCH_ANY_DTYPE(input.dtype(), T, {
      fmt::print(fmt::runtime("{:.4f} "), static_cast<double>(host_tensor.data_as<T>()[i]));
    })
  }
  fmt::print(fmt::runtime("\n"));
}

}  // namespace tunx

#include "tensor/arithmetic.hpp"  // IWYU pragma: export