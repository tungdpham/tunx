/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

/**
 * This file provides a wrapper header for gemm functions
 */
#include "cpu/gemm.hpp"
#include "cuda/gemm.hpp"
#include "device/device.hpp"
#include "device/device_manager.hpp"
#include "device/dptr.hpp"
#include "device/stream.hpp"
#include "device/task.hpp"
#include "type/type.hpp"

namespace tunx {

template <typename IO_T, typename Param_T = IO_T, typename Compute_T = IO_T>
void legacy_gemm(const dptr &A, const dptr &B, const dptr &C, size_t M, size_t N, size_t K,
                 const bool trans_A, const bool trans_B, const IO_T alpha, const IO_T beta,
                 size_t lda, size_t ldb, size_t ldc, Device &device = getHost(),
                 stream stream = nullptr) {
  if (A.device_type() != B.device_type() || A.device_type() != C.device_type()) {
    throw std::runtime_error("All device pointers must be on the same device type for gemm.");
  }
  if (A.device_type() == DeviceType::CPU) {
    if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
      throw std::runtime_error(
          "gemm mixed dtype dispatch not implemented for CPU (io/param/compute must match).");
    }
    create_cpu_task(device, stream, cpu::legacy_gemm<IO_T>, A.get<IO_T>(), B.get<Param_T>(),
                    C.get<IO_T>(), M, N, K, trans_A, trans_B, alpha, beta, lda, ldb, ldc);
  }
#ifdef TUNX_USE_CUDA
  else if (A.device_type() == DeviceType::CUDA) {
    create_cuda_task(device, stream, cuda::legacy_gemm_ex<IO_T, Param_T, Compute_T>, A.get<IO_T>(),
                     B.get<Param_T>(), C.get<IO_T>(), M, N, K, trans_A, trans_B, alpha, beta, lda,
                     ldb, ldc);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for gemm.");
  }
}

inline void gemm(const dptr &A, const dptr &B, dptr &C, size_t M, size_t N, size_t K,
                 const bool trans_A, const bool trans_B, float alpha, float beta, size_t lda,
                 size_t ldb, size_t ldc, DTypeDesc type_desc, Device &device = getHost(),
                 stream stream = nullptr) {
  if (A.device_type() != B.device_type() || A.device_type() != C.device_type()) {
    throw std::runtime_error("All device pointers must be on the same device type for gemm.");
  }
  if (A.device_type() == DeviceType::CPU) {
    create_cpu_task(device, stream, cpu::gemm, A.get(), B.get(), C.get(), M, N, K, trans_A, trans_B,
                    alpha, beta, type_desc);
  }
#ifdef TUNX_USE_CUDA
  else if (A.device_type() == DeviceType::CUDA) {
    create_cuda_task(device, stream, cuda::gemm_ex, A.get(), B.get(), C.get(), M, N, K, trans_A,
                     trans_B, alpha, beta, lda, ldb, ldc, type_desc.io_dtype, type_desc.param_dtype,
                     type_desc.io_dtype, type_desc.compute_dtype);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for gemm.");
  }
}
}  // namespace tunx