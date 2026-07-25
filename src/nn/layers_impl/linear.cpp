/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/linear.hpp"

#include <memory>

namespace tunx {
namespace internal {

LinearImpl::LinearImpl(const std::string &name)
    : SISOLayerImpl(name),
      activation_(std::make_unique<func::Linear>()) {}

Tensor LinearImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  Tensor output = make_tensor(input.shape(), io_dtype_);
  copy(input, output, engine_handle_.get_stream());
  return output;
}

Tensor LinearImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  Tensor grad_input = make_tensor(grad_output.shape(), io_dtype_);
  copy(grad_output, grad_input, engine_handle_.get_stream());
  return grad_input;
}

LayerConfig LinearImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  return config;
}

std::shared_ptr<LinearImpl> LinearImpl::create_from_config(const LayerConfig &config) {
  return std::make_shared<LinearImpl>(config.name);
}

}  // namespace internal
}  // namespace tunx
