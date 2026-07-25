/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/sub.hpp"

#include <stdexcept>

#include "tensor/ops.hpp"
#include "type/type.hpp"

namespace tunx {
namespace internal {

Vec<Vec<size_t>> SubImpl::output_shapes(const Vec<Vec<size_t>> &input_shapes) const {
  if (input_shapes.size() != 2) {
    throw std::runtime_error("SubImpl: expected exactly 2 inputs");
  }
  if (input_shapes[0] != input_shapes[1]) {
    throw std::runtime_error("SubImpl: both inputs must have the same shape");
  }
  return {input_shapes[0]};
}

Vec<Tensor> SubImpl::forward_impl(const Vec<Tensor> &inputs, Residuals &residuals) {
  if (inputs.size() != 2) {
    throw std::runtime_error("SubImpl: expected exactly 2 inputs");
  }
  const Tensor &a = inputs[0];
  const Tensor &b = inputs[1];

  if (a.shape() != b.shape()) {
    throw std::runtime_error("SubImpl: both inputs must have the same shape");
  }

  Tensor output = make_tensor(a.shape(), io_dtype_);
  sub(a, b, output, engine_handle_.get_stream());

  return {output};
}

Vec<Tensor> SubImpl::backward_impl(const Vec<Tensor> &grad_outputs, Residuals &residuals) {
  if (grad_outputs.size() != 1) {
    throw std::runtime_error("SubImpl: expected exactly 1 grad output");
  }
  const Tensor &grad_out = grad_outputs[0];

  // grad_a = grad_out, grad_b = -grad_out
  Tensor grad_a = make_tensor(grad_out.shape(), this->io_dtype_);
  Tensor grad_b = make_tensor(grad_out.shape(), this->io_dtype_);

  copy(grad_out, grad_a, engine_handle_.get_stream());
  mul_scalar(grad_out, -1.0, grad_b, engine_handle_.get_stream());

  return {grad_a, grad_b};
}

LayerConfig SubImpl::get_config() const {
  LayerConfig config;
  config.type = TYPE_NAME;
  config.name = this->name_;
  return config;
}

std::shared_ptr<SubImpl> SubImpl::create_from_config(const LayerConfig &config) {
  return std::make_shared<SubImpl>(config.name.empty() ? "sub" : config.name);
}

}  // namespace internal
}  // namespace tunx
