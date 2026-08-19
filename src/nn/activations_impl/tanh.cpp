/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/activations_impl/tanh.hpp"

#include <cassert>

#include "device/task.hpp"
#include "nn/activations_impl/cpu/tanh_kernels.hpp"
#include "tensor/tensor.hpp"
#ifdef TUNX_USE_CUDA
#include "nn/activations_impl/cuda/tanh_kernels.hpp"
#endif

namespace tunx {
namespace func {

void Tanh::apply(const Tensor &input, Tensor &output, stream s) const {
  if (input.shape() != output.shape()) {
    throw std::runtime_error("Input and output shapes must match for Tanh");
  }
  if (input.device() != output.device()) {
    throw std::runtime_error("Input and output must be on the same device for Tanh");
  }

  size_t size = input.size();
  auto &device = input.device();
  if (input.device_type() == DeviceType::CPU) {
    create_cpu_task(device, s, cpu::tanh, input.dtype(), input.data_as(), output.data_as(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (input.device_type() == DeviceType::CUDA) {
    create_cuda_task(device, s, cuda::tanh, input.dtype(), input.data_as(), output.data_as(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for Tanh apply");
  }
}

void Tanh::compute_gradient(const Tensor &input, const Tensor &grad_output, Tensor &grad_input,
                            stream s) const {
  assert(grad_output.shape() == grad_input.shape() &&
         "Shapes must match for in-place grad_output computation");
  if (grad_output.device() != grad_input.device()) {
    throw std::runtime_error("Input and upstream grad_output must be on the same device for Tanh");
  }

  size_t size = grad_output.size();
  auto &device = grad_output.device();
  if (grad_output.device_type() == DeviceType::CPU) {
    create_cpu_task(device, s, cpu::tanh_gradient, input.dtype(), input.data_as(),
                    grad_output.data_as(), grad_input.data_as(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (grad_output.device_type() == DeviceType::CUDA) {
    create_cuda_task(device, s, cuda::tanh_gradient, input.dtype(), input.data_as(),
                     grad_output.data_as(), grad_input.data_as(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for Tanh compute_gradient");
  }
}

std::string Tanh::name() const { return "tanh"; }

std::unique_ptr<ActivationFunction> Tanh::clone() const { return std::make_unique<Tanh>(); }

}  // namespace func
}  // namespace tunx