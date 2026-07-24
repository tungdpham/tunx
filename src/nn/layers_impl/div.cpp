/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/div.hpp"

#include <stdexcept>

#include "kernel/kernel.hpp"
#include "type/type.hpp"

namespace tunx {
namespace internal {

Vec<Vec<size_t>> DivImpl::output_shapes(const Vec<Vec<size_t>> &input_shapes) const {
  if (input_shapes.size() != 2) {
    throw std::runtime_error("DivImpl: expected exactly 2 inputs");
  }
  if (input_shapes[0] != input_shapes[1]) {
    throw std::runtime_error("DivImpl: both inputs must have the same shape");
  }
  return {input_shapes[0]};
}

Vec<Tensor> DivImpl::forward_impl(const Vec<Tensor> &inputs, Residuals &residuals) {
  if (inputs.size() != 2) {
    throw std::runtime_error("DivImpl: expected exactly 2 inputs");
  }
  const Tensor &a = inputs[0];
  const Tensor &b = inputs[1];

  if (a.shape() != b.shape()) {
    throw std::runtime_error("DivImpl: both inputs must have the same shape");
  }

  Tensor output = make_tensor(a.shape(), io_dtype_);
  size_t n = a.size();

  if (this->is_training_) {
    residuals["a"] = a;
    residuals["b"] = b;
  }

  kernel::div(a.dtype(), a.data_ptr(), b.data_ptr(), output.data_ptr(), n);

  return {output};
}

Vec<Tensor> DivImpl::backward_impl(const Vec<Tensor> &grad_outputs, Residuals &residuals) {
  if (grad_outputs.size() != 1) {
    throw std::runtime_error("DivImpl: expected exactly 1 grad output");
  }
  const Tensor &grad_out = grad_outputs[0];
  const Tensor &a = residuals["a"];
  const Tensor &b = residuals["b"];
  size_t n = grad_out.size();

  // grad_a = grad_out / b
  // grad_b = -(grad_out * a) / b^2
  Tensor grad_a = make_tensor(grad_out.shape(), this->io_dtype_);
  Tensor grad_b = make_tensor(grad_out.shape(), this->io_dtype_);

  // grad_a = grad_out / b
  kernel::div(grad_out.dtype(), grad_out.data_ptr(), b.data_ptr(), grad_a.data_ptr(), n,
              engine_handle_.get_stream());

  Tensor b_sq = make_tensor(grad_out.shape(), this->io_dtype_);
  kernel::mul(b.dtype(), b.data_ptr(), b.data_ptr(), b_sq.data_ptr(), n,
              engine_handle_.get_stream());

  Tensor numerator = make_tensor(grad_out.shape(), this->io_dtype_);
  kernel::mul(grad_out.dtype(), grad_out.data_ptr(), a.data_ptr(), numerator.data_ptr(), n,
              engine_handle_.get_stream());

  kernel::div(numerator.dtype(), numerator.data_ptr(), b_sq.data_ptr(), grad_b.data_ptr(), n,
              engine_handle_.get_stream());

  kernel::mul_scalar(grad_b.dtype(), grad_b.data_ptr(), -1, grad_b.data_ptr(), n,
                     engine_handle_.get_stream());

  return {grad_a, grad_b};
}

LayerConfig DivImpl::get_config() const {
  LayerConfig config;
  config.type = TYPE_NAME;
  config.name = this->name_;
  return config;
}

std::shared_ptr<DivImpl> DivImpl::create_from_config(const LayerConfig &config) {
  return std::make_shared<DivImpl>(config.name.empty() ? "div" : config.name);
}

}  // namespace internal
}  // namespace tunx
