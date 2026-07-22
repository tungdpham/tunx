/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/elu.hpp"

#include <memory>
#include <stdexcept>

namespace tunx {
namespace internal {

ELUImpl::ELUImpl(float alpha, const std::string &name)
    : SISOLayerImpl(name),
      activation_(std::make_unique<func::ELU>(alpha)),
      alpha_(alpha) {}

Tensor ELUImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  if (this->is_training_) {
    residuals["input"] = input;
  }

  Tensor output = make_tensor(input.shape(), io_dtype_);
  activation_->apply(input, output);
  return output;
}

Tensor ELUImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  const Tensor &input = residuals["input"];
  if (!input) {
    throw std::runtime_error("No cached input found for backward pass in ELUImpl");
  }

  Tensor grad_input = make_tensor(input.shape(), io_dtype_);
  activation_->compute_gradient(input, grad_output, grad_input);
  return grad_input;
}

LayerConfig ELUImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("alpha", alpha_);
  return config;
}

std::shared_ptr<ELUImpl> ELUImpl::create_from_config(const LayerConfig &config) {
  float alpha = config.get<float>("alpha", 1.0f);
  return std::make_shared<ELUImpl>(alpha, config.name);
}

}  // namespace internal
}  // namespace tunx
