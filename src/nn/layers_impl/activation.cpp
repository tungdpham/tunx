/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/activation.hpp"

#include "nn/activations.hpp"
namespace tunx {
namespace internal {

ActivationImpl::ActivationImpl(std::unique_ptr<ActivationFunction> activation,
                               const std::string &name)
    : SISOLayerImpl(name),
      activation_(std::move(activation)) {
  if (!activation_) {
    throw std::invalid_argument("Activation function cannot be null");
  }
}

Tensor ActivationImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  if (this->is_training_) {
    residuals["input"] = input;
  }

  Tensor output = make_tensor(input.shape(), input.dtype());
  activation_->apply(input, output);
  return output;
}

Tensor ActivationImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  const Tensor &input = residuals["input"];
  if (!input) {
    throw std::runtime_error("No cached input found for backward pass in ActivationImpl");
  }
  Tensor grad_input = make_tensor(input.shape(), input.dtype());
  activation_->compute_gradient(input, grad_output, grad_input);
  return grad_input;
}

LayerConfig ActivationImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("activation", activation_->name());
  return config;
}

Vec<size_t> ActivationImpl::compute_output_shape(const Vec<size_t> &input_shape) const {
  return input_shape;
}

std::shared_ptr<ActivationImpl> ActivationImpl::create_from_config(const LayerConfig &config) {
  std::string activation_name = config.get<std::string>("activation", "relu");
  ActivationFactory::register_defaults();
  auto activation = ActivationFactory::create(activation_name);
  return std::make_shared<ActivationImpl>(std::move(activation), config.name);
}

}  // namespace internal
}  // namespace tunx
