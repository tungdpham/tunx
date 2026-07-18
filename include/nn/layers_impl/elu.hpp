/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "nn/activations_impl/elu.hpp"
#include "nn/siso_layer.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

namespace internal {
class ELUImpl : public SISOLayerImpl {
private:
  std::unique_ptr<func::ELU> activation_;
  float alpha_;

protected:
  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  static constexpr const char *TYPE_NAME = "elu";

  explicit ELUImpl(float alpha = 1.0f, const std::string &name = "elu");

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  static std::shared_ptr<ELUImpl> create_from_config(const LayerConfig &config);

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override {
    return input_shape;
  }

  float get_alpha() const { return alpha_; }
};

}  // namespace internal

class ELU : public LayerRef<internal::ELUImpl> {
public:
  explicit ELU(float alpha = 1.0f, const std::string &name = "elu")
      : LayerRef(std::make_shared<internal::ELUImpl>(alpha, name)) {}

  using LayerRef<internal::ELUImpl>::LayerRef;
};

}  // namespace tunx
