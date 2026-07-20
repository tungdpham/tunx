/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/sigmoid.hpp"

#include <memory>
#include <stdexcept>

namespace tunx {
namespace internal {

SigmoidImpl::SigmoidImpl(const std::string &name)
    : SISOLayerImpl(name),
      activation_(std::make_unique<func::Sigmoid>()) {}

Tensor SigmoidImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  Tensor output = get_tensor(input.shape(), io_dtype_);
  activation_->apply(input, output);

  if (this->is_training_) {
    // Cache output for efficient backward pass
    // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    residuals["output"] = output;
  }

  return output;
}

Tensor SigmoidImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  Tensor &output = residuals["output"];
  if (!output) {
    throw std::runtime_error("No cached output found for backward pass in SigmoidImpl");
  }

  Tensor grad_input = get_tensor(grad_output.shape(), io_dtype_);

  // Gradient: grad_input = grad_output * output * (1 - output)
  size_t num_elements = grad_output.size();
  if (grad_output.device_type() == DeviceType::CPU) {
    const float *grad_out_data = grad_output.data_as<float>();
    const float *output_data = output.data_as<float>();
    float *grad_in_data = grad_input.data_as<float>();
    for (size_t i = 0; i < num_elements; ++i) {
      float sig = output_data[i];
      grad_in_data[i] = grad_out_data[i] * sig * (1.0f - sig);
    }
  }
#ifdef TUNX_USE_CUDA
  else if (grad_output.device_type() == DeviceType::CUDA) {
    throw std::runtime_error("SigmoidImpl: CUDA backward not yet implemented");
  }
#endif

  return grad_input;
}

LayerConfig SigmoidImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  return config;
}

std::shared_ptr<SigmoidImpl> SigmoidImpl::create_from_config(const LayerConfig &config) {
  return std::make_shared<SigmoidImpl>(config.name);
}

}  // namespace internal
}  // namespace tunx
