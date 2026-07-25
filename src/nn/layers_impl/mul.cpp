/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/mul.hpp"

#include <stdexcept>

#include "tensor/ops.hpp"
#include "type/type.hpp"

namespace tunx {
namespace internal {

Vec<Vec<size_t>> MulImpl::output_shapes(const Vec<Vec<size_t>> &input_shapes) const {
  if (input_shapes.size() != 2) {
    throw std::runtime_error("MulImpl: expected exactly 2 inputs");
  }
  if (input_shapes[0] != input_shapes[1]) {
    throw std::runtime_error("MulImpl: both inputs must have the same shape");
  }
  return {input_shapes[0]};
}

Vec<Tensor> MulImpl::forward_impl(const Vec<Tensor> &inputs, Residuals &residuals) {
  if (inputs.size() != 2) {
    throw std::runtime_error("MulImpl: expected exactly 2 inputs");
  }
  const Tensor &a = inputs[0];
  const Tensor &b = inputs[1];

  if (a.shape() != b.shape()) {
    throw std::runtime_error("MulImpl: both inputs must have the same shape");
  }

  Tensor output = make_tensor(a.shape(), io_dtype_);

  if (this->is_training_) {
    residuals["a"] = a;
    residuals["b"] = b;
  }

  mul(a, b, output, engine_handle_.get_stream());

  return {output};
}

Vec<Tensor> MulImpl::backward_impl(const Vec<Tensor> &grad_outputs, Residuals &residuals) {
  if (grad_outputs.size() != 1) {
    throw std::runtime_error("MulImpl: expected exactly 1 grad output");
  }
  const Tensor &grad_out = grad_outputs[0];
  const Tensor &a = residuals["a"];
  const Tensor &b = residuals["b"];

  // grad_a = grad_out * b,  grad_b = grad_out * a
  Tensor grad_a = make_tensor(grad_out.shape(), this->io_dtype_);
  Tensor grad_b = make_tensor(grad_out.shape(), this->io_dtype_);

  mul(grad_out, b, grad_a, engine_handle_.get_stream());
  mul(grad_out, a, grad_b, engine_handle_.get_stream());

  return {grad_a, grad_b};
}

LayerConfig MulImpl::get_config() const {
  LayerConfig config;
  config.type = TYPE_NAME;
  config.name = this->name_;
  return config;
}

std::shared_ptr<MulImpl> MulImpl::create_from_config(const LayerConfig &config) {
  return std::make_shared<MulImpl>(config.name.empty() ? "mul" : config.name);
}

}  // namespace internal
}  // namespace tunx
