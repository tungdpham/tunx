/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "nn/activations_impl/sigmoid.hpp"
#include "nn/siso_layer.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

/**
 * funcgmoid LayerImpl with output caching
 * Caches the output activation during forward pass for efficient gradient computation.
 * funcgmoid gradient: grad_input = grad_output * output * (1 - output)
 */
namespace internal {
class SigmoidImpl : public SISOLayerImpl {
private:
  std::unique_ptr<func::Sigmoid> activation_;

protected:
  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  static constexpr const char *TYPE_NAME = "sigmoid";

  explicit SigmoidImpl(const std::string &name = "sigmoid");

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  static std::shared_ptr<SigmoidImpl> create_from_config(const LayerConfig &config);

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override {
    return input_shape;
  }
};

}  // namespace internal

class Sigmoid : public LayerRef<internal::SigmoidImpl> {
public:
  explicit Sigmoid(const std::string &name = "sigmoid")
      : LayerRef(std::make_shared<internal::SigmoidImpl>(name)) {}

  using LayerRef<internal::SigmoidImpl>::LayerRef;
};

}  // namespace tunx
