#include "kernel/kernel.hpp"

#include <stdexcept>

#include "kernel/cpu/kernels.hpp"
#ifdef TUNX_USE_CUDA
#include "kernel/cuda/kernels.hpp"
#endif  // TUNX_USE_CUDA
#include "device/task.hpp"

namespace tunx {
namespace kernel {

void copy(const dptr src, dptr dst, size_t size, stream s) {
  auto &src_device = src.device();
  auto &dst_device = dst.device();
  if (src_device == dst_device) {
    // same device copy
    if (src_device.device_type() == DeviceType::CPU) {
      return create_cpu_task(src_device, s, kernel::cpu::copy, DType_t::BYTE, src.get(), dst.get(),
                             size);
#ifdef TUNX_USE_CUDA
    } else if (src_device.device_type() == DeviceType::CUDA) {
      return create_cuda_task(src_device, s, kernel::cuda::copy, DType_t::BYTE, src.get(),
                              dst.get(), size);
    }
#endif
    else {
      throw std::runtime_error("copy: Unsupported device type");
    }
  }
  auto src_device_type = src_device.device_type();
  auto dst_device_type = dst_device.device_type();

  if (src_device_type == DeviceType::CPU && dst_device_type == DeviceType::CUDA) {
    // host to device copy
#ifdef TUNX_USE_CUDA
    return create_cuda_task(dst_device, s, kernel::cuda::h2d_copy, DType_t::BYTE, src.get(),
                            dst.get(), size);
#else
      throw std::runtime_error("copy: CUDA not enabled for CPU to CUDA copy");
#endif
  } else if (src_device_type == DeviceType::CUDA && dst_device_type == DeviceType::CPU) {
    // device to host copy
#ifdef TUNX_USE_CUDA
    return create_cuda_task(src_device, s, kernel::cuda::d2h_copy, DType_t::BYTE, src.get(),
                            dst.get(), size);
#else
      throw std::runtime_error("copy: CUDA not enabled for CUDA to CPU copy");
#endif
  } else {
    throw std::runtime_error("copy: Unsupported device type combination");
  }
}

void bswap(DType_t dtype, const dptr input, dptr output, size_t size, stream s) {
  auto &device = input.device();
  auto device_type = device.device_type();
  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, s, kernel::cpu::bswap, dtype, input.get(), output.get(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, s, kernel::cuda::bswap, dtype, input.get(), output.get(), size);
  }
#endif
  else {
    throw std::runtime_error("bswap: Unsupported device type");
  }
}

}  // namespace kernel
}  // namespace tunx
