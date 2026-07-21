#pragma once

#include "device/dptr.hpp"
#include "ops/cpu/kernels.hpp"
#ifdef TUNX_USE_CUDA
#include "ops/cuda/kernels.hpp"
#endif
#include <cstddef>
#include <stdexcept>

#include "device/task.hpp"

namespace tunx {
namespace ops {

template <typename T>
void add(const dptr a, const dptr b, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("add: All device pointers must be on the same device");
  }
  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::add<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::add<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void sub(const dptr a, const dptr b, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("sub: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::sub<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::sub<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void mul(const dptr a, const dptr b, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("mul: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::mul<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::mul<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void div(const dptr a, const dptr b, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("div: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::div<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::div<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void fmadd(const dptr a, const dptr b, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("fmadd: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::fmadd<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::fmadd<T>, a.get<T>(), b.get<T>(), c.get<T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void fmsub(const dptr a, const dptr b, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("fmsub: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::fmsub<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::fmsub<T>, a.get<T>(), b.get<T>(), c.get<T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void fnmadd(const dptr a, const dptr b, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("fnmadd: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::fnmadd<T>, a.get<T>(), b.get<T>(), c.get<T>(),
                           size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::fnmadd<T>, a.get<T>(), b.get<T>(), c.get<T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void add_scalar(const dptr a, T scalar, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("add_scalar: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::add_scalar<T>, a.get<T>(), scalar, c.get<T>(),
                           size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::add_scalar<T>, a.get<T>(), scalar, c.get<T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void sub_scalar(const dptr a, T scalar, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("sub_scalar: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::sub_scalar<T>, a.get<T>(), scalar, c.get<T>(),
                           size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::sub_scalar<T>, a.get<T>(), scalar, c.get<T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void mul_scalar(const dptr a, T scalar, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("mul_scalar: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::mul_scalar<T>, a.get<T>(), scalar, c.get<T>(),
                           size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::mul_scalar<T>, a.get<T>(), scalar, c.get<T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void div_scalar(const dptr a, T scalar, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("div_scalar: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::div_scalar<T>, a.get<T>(), scalar, c.get<T>(),
                           size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::div_scalar<T>, a.get<T>(), scalar, c.get<T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void set_scalar(dptr c, T scalar, size_t size, stream stream = nullptr) {
  auto &device = c.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::set_scalar<T>, c.get<T>(), scalar, size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::set_scalar<T>, c.get<T>(), scalar, size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void axpy(T alpha, const dptr x, dptr y, size_t size, stream stream = nullptr) {
  if (x.device() != y.device()) {
    throw std::runtime_error("axpy: All device pointers must be on the same device");
  }

  auto &device = x.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::axpy<T>, alpha, x.get<T>(), y.get<T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::axpy<T>, alpha, x.get<T>(), y.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void sqrt(const dptr a, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("sqrt: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::sqrt<T>, a.get<T>(), c.get<T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::sqrt<T>, a.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
inline void rsqrt(const dptr a, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("rsqrt: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::rsqrt<T>, a.get<T>(), c.get<T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::rsqrt<T>, a.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
inline void rcp(const dptr a, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("rcp: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::rcp<T>, a.get<T>(), c.get<T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::rcp<T>, a.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void abs(const dptr a, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("abs: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::abs<T>, a.get<T>(), c.get<T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::abs<T>, a.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void min(const dptr a, const dptr b, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("min: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::min<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::min<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void max(const dptr a, const dptr b, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("max: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::max<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::max<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void scalar_max(const dptr a, T scalar, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("scalar_max: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::scalar_max<T>, a.get<T>(), scalar, c.get<T>(),
                           size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::scalar_max<T>, a.get<T>(), scalar, c.get<T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void clamp(const dptr a, T min_val, T max_val, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("clamp: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::clamp<T>, a.get<T>(), min_val, max_val, c.get<T>(),
                           size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::clamp<T>, a.get<T>(), min_val, max_val,
                            c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void equal(const dptr a, const dptr b, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("equal: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::equal<T>, a.get<T>(), b.get<T>(), c.get<T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::equal<T>, a.get<T>(), b.get<T>(), c.get<T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void greater(const dptr a, const dptr b, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != b.device() || a.device() != c.device()) {
    throw std::runtime_error("greater: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::greater<T>, a.get<T>(), b.get<T>(), c.get<T>(),
                           size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::greater<T>, a.get<T>(), b.get<T>(), c.get<T>(),
                            size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void copy(const dptr a, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("copy: All device pointers must be on the same device");
  }

  if (a.get<T>() == nullptr || c.get<T>() == nullptr) {
    throw std::runtime_error("copy: Null pointer exception in copy operation");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::copy<T>, a.get<T>(), c.get<T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::copy<T>, a.get<T>(), c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

// Special copy for copying cross devices (resort to same device/host copy if applicable)
template <typename T>
void cd_copy(const dptr a, dptr c, size_t size, stream stream = nullptr) {
  auto &a_device = a.device();
  auto &c_device = c.device();
  if (a_device == c_device) {
    // same device copy
    return copy<T>(a, c, size, stream);
  }
  auto a_device_type = a_device.device_type();
  auto c_device_type = c_device.device_type();

  if (a_device_type == DeviceType::CPU && c_device_type == DeviceType::CUDA) {
    // host to device copy
#ifdef TUNX_USE_CUDA
    return create_cuda_task(c_device, stream, cuda::h2d_copy<T>, a.get<T>(), c.get<T>(), size);
#else
    throw std::runtime_error("cd_copy: CUDA not enabled for CPU to CUDA copy");
#endif
  } else if (a_device_type == DeviceType::CUDA && c_device_type == DeviceType::CPU) {
    // device to host copy
#ifdef TUNX_USE_CUDA
    return create_cuda_task(a_device, stream, cuda::d2h_copy<T>, a.get<T>(), c.get<T>(), size);
#else
    throw std::runtime_error("cd_copy: CUDA not enabled for CUDA to CPU copy");
#endif
  } else {
    throw std::runtime_error("cd_copy: Unsupported device type combination");
  }
}

template <typename T>
void bswap(const dptr a, dptr c, size_t size, stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("bswap: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::bswap<T>, a.get<T>(), c.get<T>(), size);
  } else if (device_type == DeviceType::CUDA) {
#ifdef TUNX_USE_CUDA
    return create_cuda_task(device, stream, cuda::bswap<T>, a.get<T>(), c.get<T>(), size);
#else
    throw std::runtime_error("bswap: CUDA support not compiled in");
#endif
  } else {
    throw std::runtime_error("bswap: Unsupported device type");
  }
}

template <typename T>
void zero(dptr c, size_t size, stream stream = nullptr) {
  auto &device = c.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::zero<T>, c.get<T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::zero<T>, c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
T sum(dptr a, size_t size) {
  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return cpu::sum(a.get<T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return cuda::sum(a.get<T>(), size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
T dot_product(dptr a, dptr b, size_t size) {
  if (a.device() != b.device()) {
    throw std::runtime_error("dot_product: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return cpu::dot_product(a.get<T>(), b.get<T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return cuda::dot_product(a.get<T>(), b.get<T>(), size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
T norm_squared(dptr a, size_t size) {
  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return cpu::norm_squared(a.get<T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return cuda::norm_squared(a.get<T>(), size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
T sum_squared_diff(const dptr a, T mean, size_t size) {
  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return cpu::sum_squared_diff(a.get<T>(), mean, size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return cuda::sum_squared_diff(a.get<T>(), mean, size, 0);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void sub_mul_scalar(const dptr a, T sub_scalar, T mul_scalar, dptr c, size_t size,
                    stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("sub_mul_scalar: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::sub_mul_scalar<T>, a.get<T>(), sub_scalar,
                           mul_scalar, c.get<T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::sub_mul_scalar<T>, a.get<T>(), sub_scalar,
                            mul_scalar, c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void mul_add_scalar(const dptr a, T mul_scalar, T add_scalar, dptr c, size_t size,
                    stream stream = nullptr) {
  if (a.device() != c.device()) {
    throw std::runtime_error("mul_add_scalar: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::mul_add_scalar<T>, a.get<T>(), mul_scalar,
                           add_scalar, c.get<T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::mul_add_scalar<T>, a.get<T>(), mul_scalar,
                            add_scalar, c.get<T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void fill_random_uniform(dptr data, size_t size, T min_val, T max_val, unsigned long long seed,
                         stream stream = nullptr) {
  auto &device = data.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::fill_random_uniform<T>, data.get<T>(), size,
                           min_val, max_val, seed);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::fill_random_uniform<T>, data.get<T>(), size,
                            min_val, max_val, seed);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void fill_random_normal(dptr data, size_t size, T mean, T stddev, unsigned long long seed,
                        stream stream = nullptr) {
  auto &device = data.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::fill_random_normal<T>, data.get<T>(), size, mean,
                           stddev, seed);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::fill_random_normal<T>, data.get<T>(), size, mean,
                            stddev, seed);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename T>
void check_equals(const dptr a, const dptr b, bool &result, double eps = 1e-5,
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
    return create_cpu_task(device, stream, cpu::check_equals<T>, a.get<T>(), b.get<T>(),
                           a.capacity(), result, eps);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::check_equals<T>, a.get<T>(), b.get<T>(),
                            a.capacity(), result, eps);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

template <typename A_T, typename B_T>
void cast(const dptr a, dptr b, size_t size, stream stream = nullptr) {
  if (a.device() != b.device()) {
    throw std::runtime_error("cast: All device pointers must be on the same device");
  }

  auto &device = a.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, cpu::cast<A_T, B_T>, a.get<A_T>(), b.get<B_T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, cuda::cast<A_T, B_T>, a.get<A_T>(), b.get<B_T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type");
  }
}

}  // namespace ops
}  // namespace tunx