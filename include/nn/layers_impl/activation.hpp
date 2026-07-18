/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "nn/activations_impl/base_activation.hpp"
#include "nn/siso_layer.hpp"

namespace tunx {

namespace internal {
class ActivationImpl : public SISOLayerImpl {
private:
  std::unique_ptr<ActivationFunction> activation_;

  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  static constexpr const char *TYPE_NAME = "activation";

  explicit ActivationImpl(std::unique_ptr<ActivationFunction> activation,
                               const std::string &name = "activation");

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  static std::shared_ptr<ActivationImpl> create_from_config(const LayerConfig &config);
  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;
};

}  // namespace internal

class Activation : public LayerRef<internal::ActivationImpl> {
public:
  Activation(std::unique_ptr<ActivationFunction> activation,
                  const std::string &name = "activation")
      : LayerRef(std::make_shared<internal::ActivationImpl>(std::move(activation), name)) {}

  using LayerRef<internal::ActivationImpl>::LayerRef;
};

}  // namespace tunx
