/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/activations_impl/tanh.hpp"

#include <cassert>

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

  DISPATCH_DTYPE(input.dtype(), T, return apply_impl<T>(input, output, s));
}

void Tanh::compute_gradient(const Tensor &input, const Tensor &grad_output, Tensor &grad_input,
                            stream s) const {
  assert(grad_output.shape() == grad_input.shape() &&
         "Shapes must match for in-place grad_output computation");
  if (grad_output.device() != grad_input.device()) {
    throw std::runtime_error("Input and upstream grad_output must be on the same device for Tanh");
  }
  DISPATCH_DTYPE(input.dtype(), T,
                 return compute_gradient_impl<T>(input, grad_output, grad_input, s));
}

std::string Tanh::name() const { return "tanh"; }

std::unique_ptr<ActivationFunction> Tanh::clone() const { return std::make_unique<Tanh>(); }

template <typename Compute_T>
void Tanh::apply_impl(const Tensor &input, Tensor &output, stream handle) const {
  if (input.dtype() != dtype_of<Compute_T>() || output.dtype() != dtype_of<Compute_T>()) {
    throw std::runtime_error("Tanh tensor dtype mismatch with dispatch type");
  }

  size_t size = input.size();
  auto &device = input.device();
  if (input.device_type() == DeviceType::CPU) {
    create_cpu_task(device, handle, cpu::tanh<Compute_T>, input.data_as<Compute_T>(),
                    output.data_as<Compute_T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (input.device_type() == DeviceType::CUDA) {
    create_cuda_task(device, handle, cuda::tanh<Compute_T>, input.data_as<Compute_T>(),
                     output.data_as<Compute_T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for Tanh apply");
  }
}

template <typename Compute_T>
void Tanh::compute_gradient_impl(const Tensor &input, const Tensor &grad_output, Tensor &grad_input,
                                 stream handle) const {
  if (input.dtype() != dtype_of<Compute_T>() || grad_output.dtype() != dtype_of<Compute_T>() ||
      grad_input.dtype() != dtype_of<Compute_T>()) {
    throw std::runtime_error("Tanh tensor dtype mismatch with dispatch type");
  }

  size_t size = grad_output.size();
  auto &device = grad_output.device();
  if (grad_output.device_type() == DeviceType::CPU) {
    create_cpu_task(device, handle, cpu::tanh_gradient<Compute_T>, input.data_as<Compute_T>(),
                    grad_output.data_as<Compute_T>(), grad_input.data_as<Compute_T>(), size);
  }
#ifdef TUNX_USE_CUDA
  else if (grad_output.device_type() == DeviceType::CUDA) {
    create_cuda_task(device, handle, cuda::tanh_gradient<Compute_T>, input.data_as<Compute_T>(),
                     grad_output.data_as<Compute_T>(), grad_input.data_as<Compute_T>(), size);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for Tanh compute_gradient");
  }
}

}  // namespace func
}  // namespace tunx