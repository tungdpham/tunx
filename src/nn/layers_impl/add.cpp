/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/add.hpp"

#include <stdexcept>

#include "ops/ops.hpp"
#include "type/type.hpp"

namespace tunx {
namespace internal {

Vec<Vec<size_t>> AddImpl::output_shapes(const Vec<Vec<size_t>> &input_shapes) const {
  if (input_shapes.size() != 2) {
    throw std::runtime_error("AddImpl: expected exactly 2 inputs");
  }
  if (input_shapes[0] != input_shapes[1]) {
    throw std::runtime_error("AddImpl: both inputs must have the same shape");
  }
  return {input_shapes[0]};
}

Vec<Tensor> AddImpl::forward_impl(const Vec<Tensor> &inputs, Residuals &residuals) {
  if (inputs.size() != 2) {
    throw std::runtime_error("AddImpl: expected exactly 2 inputs");
  }
  const Tensor &a = inputs[0];
  const Tensor &b = inputs[1];

  if (a.shape() != b.shape()) {
    throw std::runtime_error("AddImpl: both inputs must have the same shape");
  }

  Tensor output = get_tensor(a.shape(), a.dtype());
  size_t n = a.size();
  DISPATCH_DTYPE(a.dtype(), T, { ops::add<T>(a.data_ptr(), b.data_ptr(), output.data_ptr(), n); });

  return {output};
}

Vec<Tensor> AddImpl::backward_impl(const Vec<Tensor> &grad_outputs, Residuals &residuals) {
  if (grad_outputs.size() != 1) {
    throw std::runtime_error("AddImpl: expected exactly 1 grad output");
  }
  const Tensor &grad_out = grad_outputs[0];
  Tensor grad_a = get_tensor(grad_out.shape(), this->io_dtype_);
  Tensor grad_b = get_tensor(grad_out.shape(), this->io_dtype_);

  grad_out.copy_to(grad_a);
  grad_out.copy_to(grad_b);

  return {grad_a, grad_b};
}

LayerConfig AddImpl::get_config() const {
  LayerConfig config;
  config.type = TYPE_NAME;
  config.name = this->name_;
  return config;
}

std::shared_ptr<AddImpl> AddImpl::create_from_config(const LayerConfig &config) {
  return std::make_shared<AddImpl>(config.name.empty() ? "add" : config.name);
}

}  // namespace internal
}  // namespace tunx
