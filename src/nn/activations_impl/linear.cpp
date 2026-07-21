/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/activations_impl/linear.hpp"

#include <memory>
#include <stdexcept>
#include <string>

#include "nn/activations_impl/base_activation.hpp"
#include "tensor/tensor.hpp"

namespace tunx {
namespace func {

void Linear::apply(const Tensor &input, Tensor &output, stream s) const {
  if (input.shape() != output.shape()) {
    throw std::runtime_error("Input and output shapes must match for Linear");
  }
  if (input.device() != output.device()) {
    throw std::runtime_error("Input and output must be on the same device for Linear");
  }

}

void Linear::compute_gradient(const Tensor &input, const Tensor &grad_output, Tensor &grad_input, stream s) const {
  if (grad_input.shape() != grad_output.shape()) {
    throw std::invalid_argument(
        "Upstream grad_output must have the same "
        "shape as pre-activation values");
  }
  grad_output.copy_to(grad_input);

}

std::string Linear::name() const { return "linear"; }

std::unique_ptr<ActivationFunction> Linear::clone() const {
  return std::make_unique<Linear>(*this);
}

}  // namespace func
}  // namespace tunx