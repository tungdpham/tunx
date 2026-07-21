/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <system_error>
#include <tuple>
#include <utility>

#include "device/cpu_device.hpp"
#include "device/cuda_device.hpp"
#include "stream.hpp"

#ifdef TUNX_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace tunx {

using ErrorStatus = std::error_code;

template <typename Func, typename... Args>
void create_cpu_task(Device &device, stream s, Func &&func, Args &&...args) {
  if (!s) {
    s = device.default_stream();
  }

  auto launch_func = [f = std::forward<Func>(func),
                      args_tuple = std::tuple<Args...>(std::forward<Args>(args)...)]() mutable {
    std::apply(f, std::tuple_cat(std::move(args_tuple)));  // add arg later if needed.
  };

  CPUDevice::launch(device, s, launch_func);
}

#ifdef TUNX_USE_CUDA
// bundle the function and inject a stream based on the handle
template <typename Func, typename... Args>
void create_cuda_task(Device &device, stream s, Func &&func, Args &&...args) {
  if (!s) {
    s = device.default_stream();
  }

  auto launch_func = [f = std::forward<Func>(func),
                      args_tuple = std::tuple<Args...>(std::forward<Args>(args)...)](
                         cudaStream_t cu_stream) mutable {
    std::apply(f, std::tuple_cat(std::move(args_tuple), std::make_tuple(cu_stream)));
  };

  CUDADevice::launch(device, s, launch_func);
}
#endif

}  // namespace tunx