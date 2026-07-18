/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "nn/activations_impl/linear.hpp"
#include "nn/siso_layer.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

namespace internal {
class LinearImpl : public SISOLayerImpl {
private:
  std::unique_ptr<func::Linear> activation_;

protected:
  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  static constexpr const char *TYPE_NAME = "linear";

  explicit LinearImpl(const std::string &name = "linear");

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  static std::shared_ptr<LinearImpl> create_from_config(const LayerConfig &config);

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override {
    return input_shape;
  }
};

}  // namespace internal

class Linear : public LayerRef<internal::LinearImpl> {
public:
  explicit Linear(const std::string &name = "linear")
      : LayerRef(std::make_shared<internal::LinearImpl>(name)) {}

  using LayerRef<internal::LinearImpl>::LayerRef;
};

}  // namespace tunx
