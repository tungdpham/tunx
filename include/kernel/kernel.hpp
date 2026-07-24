#pragma once

#include "device/dptr.hpp"
#include "kernel/cpu/kernels.hpp"
#include "type/type.hpp"
#ifdef TUNX_USE_CUDA
#include "kernel/cuda/kernels.hpp"
#endif
#include <cstddef>
#include <stdexcept>

#include "device/task.hpp"

namespace tunx {
namespace kernel {

// TODO: remove template and add DType to all functions as first args (number depending on number of
// possible typed arg).
inline void add(DType_t dtype, const dptr a, const dptr b, dptr c, size_t size,
                stream stream = nullptr) {
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("add: All device pointers must be on the same device");
  }
  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::add, dtype, a.get(), b.get(), c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::add, dtype, a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void sub(DType_t dtype, const dptr a, const dptr b, dptr c, size_t size,
                stream stream = nullptr) {
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("sub: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::sub, dtype, a.get(), b.get(), c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::sub, dtype, a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void mul(DType_t dtype, const dptr a, const dptr b, dptr c, size_t size,
                stream stream = nullptr) {
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("mul: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::mul, dtype, a.get(), b.get(), c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::mul, dtype, a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void div(DType_t dtype, const dptr a, const dptr b, dptr c, size_t size,
                stream stream = nullptr) {
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("div: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::div, dtype, a.get(), b.get(), c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::div, dtype, a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void fmadd(DType_t dtype, const dptr a, const dptr b, dptr c, size_t size,
                  stream stream = nullptr) {
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("fmadd: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::fmadd, dtype, a.get(), b.get(), c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::fmadd, dtype, a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void fmsub(DType_t dtype, const dptr a, const dptr b, dptr c, size_t size,
                  stream stream = nullptr) {
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("fmsub: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::fmsub, dtype, a.get(), b.get(), c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::fmsub, dtype, a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void fnmadd(DType_t dtype, const dptr a, const dptr b, dptr c, size_t size,
                   stream stream = nullptr) {
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("fnmadd: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::fnmadd, dtype, a.get(), b.get(), c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::fnmadd, dtype, a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void add_scalar(DType_t dtype, const dptr a, double scalar, dptr c, size_t size,
                       stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("add_scalar: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::add_scalar, dtype, a.get(), scalar, c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::add_scalar, dtype, a.get(), scalar, c.get(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void sub_scalar(DType_t dtype, const dptr a, double scalar, dptr c, size_t size,
                       stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("sub_scalar: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::sub_scalar, dtype, a.get(), scalar, c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::sub_scalar, dtype, a.get(), scalar, c.get(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void mul_scalar(DType_t dtype, const dptr a, double scalar, dptr c, size_t size,
                       stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("mul_scalar: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::mul_scalar, dtype, a.get(), scalar, c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::mul_scalar, dtype, a.get(), scalar, c.get(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void div_scalar(DType_t dtype, const dptr a, double scalar, dptr c, size_t size,
                       stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("div_scalar: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::div_scalar, dtype, a.get(), scalar, c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::div_scalar, dtype, a.get(), scalar, c.get(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void set_scalar(DType_t dtype, dptr c, double scalar, size_t size, stream stream = nullptr) {
  auto &device = c.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::set_scalar, dtype, c.get(), scalar, size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::set_scalar, dtype, c.get(), scalar, size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void axpy(DType_t dtype, double alpha, const dptr x, dptr y, size_t size,
                 stream stream = nullptr) {
  if (x.device() != y.device()) {
    throw std::runtime_error("axpy: All device pointers must be on the same device");
  }

  auto &device = x.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::axpy, dtype, alpha, x.get(), y.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::axpy, dtype, alpha, x.get(), y.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void sqrt(DType_t dtype, const dptr a, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("sqrt: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::sqrt, dtype, a.get(), c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::sqrt, dtype, a.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void rsqrt(DType_t dtype, const dptr a, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("rsqrt: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::rsqrt, dtype, a.get(), c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::rsqrt, dtype, a.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void rcp(DType_t dtype, const dptr a, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("rcp: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::rcp, dtype, a.get(), c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::rcp, dtype, a.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void abs(DType_t dtype, const dptr a, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("abs: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::abs, dtype, a.get(), c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::abs, dtype, a.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void min(DType_t dtype, const dptr a, const dptr b, dptr c, size_t size,
                stream stream = nullptr) {
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("min: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::min, dtype, a.get(), b.get(), c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::min, dtype, a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void max(DType_t dtype, const dptr a, const dptr b, dptr c, size_t size,
                stream stream = nullptr) {
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("max: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::max, dtype, a.get(), b.get(), c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::max, dtype, a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void scalar_max(DType_t dtype, const dptr a, double scalar, dptr c, size_t size,
                       stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("scalar_max: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::scalar_max, dtype, a.get(), scalar, c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::scalar_max, dtype, a.get(), scalar, c.get(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void clamp(DType_t dtype, const dptr a, double min_val, double max_val, dptr c, size_t size,
                  stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("clamp: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::clamp, dtype, a.get(), min_val, max_val, c.get(),
                           size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::clamp, dtype, a.get(), min_val, max_val, c.get(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void equal(DType_t dtype, const dptr a, const dptr b, dptr c, size_t size,
                  stream stream = nullptr) {
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("equal: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::equal, dtype, a.get(), b.get(), c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::equal, dtype, a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void greater(DType_t dtype, const dptr a, const dptr b, dptr c, size_t size,
                    stream stream = nullptr) {
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("greater: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::greater, dtype, a.get(), b.get(), c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::greater, dtype, a.get(), b.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void copy(DType_t dtype, const dptr a, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("copy: All device pointers must be on the same device");
  }

  if (a.get() == nullptr || c.get() == nullptr) {
    throw std::runtime_error("copy: Null pointer exception in copy operation");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::copy, dtype, a.get(), c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::copy, dtype, a.get(), c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

// Special copy for copying cross devices (resort to same device/host copy if applicable)
inline void cd_copy(DType_t dtype, const dptr a, dptr c, size_t size, stream stream = nullptr) {
  auto &a_device = a.device();
  auto &c_device = c.device();
  if (a_device == c_device) {
    // same device copy
    return copy(dtype, a, c, size, stream);
  }
  auto a_device_type = a_device.device_type();
  auto c_device_type = c_device.device_type();

  if (a_device_type == DeviceType::CPU && c_device_type == DeviceType::CUDA) {
    // host to device copy
#ifdef TUNX_USE_CUDA
    return create_cuda_task(c_device, stream, cuda::h2d_copy, dtype, a.get(), c.get(), size);
#else
    throw std::runtime_error("cd_copy: CUDA not enabled for CPU to CUDA copy");
#endif
  } else if (a_device_type == DeviceType::CUDA && c_device_type == DeviceType::CPU) {
    // device to host copy
#ifdef TUNX_USE_CUDA
    return create_cuda_task(a_device, stream, cuda::d2h_copy, dtype, a.get(), c.get(), size);
#else
    throw std::runtime_error("cd_copy: CUDA not enabled for CUDA to CPU copy");
#endif
  } else {
    throw std::runtime_error("cd_copy: Unsupported device type combination");
  }
}

inline void bswap(DType_t dtype, const dptr a, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("bswap: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::bswap, dtype, a.get(), c.get(), size);
  } else if (device_type == DeviceType::CUDA) {
#ifdef TUNX_USE_CUDA
    return create_cuda_task(device, stream, cuda::bswap, dtype, a.get(), c.get(), size);
#else
    throw std::runtime_error("bswap: CUDA support not compiled in");
#endif
  } else {
    throw std::runtime_error("bswap: Unsupported device type");
  }
}

inline void zero(DType_t dtype, dptr c, size_t size, stream stream = nullptr) {
  auto &device = c.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::zero, dtype, c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::zero, dtype, c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline double sum(DType_t dtype, dptr a, size_t size) {
  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return cpu::sum(dtype, a.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return cuda::sum(dtype, a.get(), size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline double dot_product(DType_t dtype, dptr a, dptr b, size_t size) {
  if (a.device() != b.device()) {
    throw std::runtime_error("dot_product: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return cpu::dot_product(dtype, a.get(), b.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return cuda::dot_product(dtype, a.get(), b.get(), size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline double norm_squared(DType_t dtype, dptr a, size_t size) {
  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return cpu::norm_squared(dtype, a.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return cuda::norm_squared(dtype, a.get(), size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline double sum_squared_diff(DType_t dtype, const dptr a, double mean, size_t size) {
  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return cpu::sum_squared_diff(dtype, a.get(), mean, size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return cuda::sum_squared_diff(dtype, a.get(), mean, size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void sub_mul_scalar(DType_t dtype, const dptr a, double sub_scalar, double mul_scalar,
                           dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("sub_mul_scalar: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::sub_mul_scalar, dtype, a.get(), sub_scalar,
                           mul_scalar, c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::sub_mul_scalar, dtype, a.get(), sub_scalar,
                            mul_scalar, c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void mul_add_scalar(DType_t dtype, const dptr a, double mul_scalar, double add_scalar,
                           dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("mul_add_scalar: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::mul_add_scalar, dtype, a.get(), mul_scalar,
                           add_scalar, c.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::mul_add_scalar, dtype, a.get(), mul_scalar,
                            add_scalar, c.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void fill_random_uniform(DType_t dtype, dptr data, size_t size, double min_val,
                                double max_val, unsigned long long seed, stream stream = nullptr) {
  auto &device = data.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::fill_random_uniform, dtype, data.get(), size,
                           min_val, max_val, seed);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::fill_random_uniform, dtype, data.get(), size,
                            min_val, max_val, seed);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void fill_random_normal(DType_t dtype, dptr data, size_t size, double mean, double stddev,
                               unsigned long long seed, stream stream = nullptr) {
  auto &device = data.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::fill_random_normal, dtype, data.get(), size, mean,
                           stddev, seed);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::fill_random_normal, dtype, data.get(), size, mean,
                            stddev, seed);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void check_equals(DType_t dtype, const dptr a, const dptr b, bool &result, double eps = 1e-5,
                         stream stream = nullptr) {
  if (a.device() != b.device()) {
    throw std::runtime_error("check_equals: All device pointers must be on the same device");
  }
  if (a.capacity() != b.capacity()) {
    throw std::runtime_error("check_equals: All device pointers must have the same capacity");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::check_equals, dtype, a.get(), b.get(), a.capacity(),
                           result, eps);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::check_equals, dtype, a.get(), b.get(),
                            a.capacity(), result, eps);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

inline void cast(DType_t a_dtype, DType_t b_dtype, const dptr a, dptr b, size_t size,
                 stream stream = nullptr) {
  if (a.device() != b.device()) {
    throw std::runtime_error("cast: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::cast, a_dtype, b_dtype, a.get(), b.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::cast, a_dtype, b_dtype, a.get(), b.get(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

}  // namespace kernel
}  // namespace tunx