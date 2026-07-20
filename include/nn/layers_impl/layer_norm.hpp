/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "nn/siso_layer.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace tunx {

namespace internal {
class LayerNormImpl : public SISOLayerImpl {
private:
  size_t normalized_shape_;  // Size of C (channels)
  float epsilon_;
  bool affine_;  // Whether to use learnable affine parameters

  Param gamma_;
  Param beta_;

  void init_impl() override;
  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  explicit LayerNormImpl(size_t normalized_shape, float epsilon = 1e-5f, bool affine = true,
                         const std::string &name = "layer_norm");

  ~LayerNormImpl();

  static constexpr const char *TYPE_NAME = "layer_norm";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override {
    return input_shape;
  }


  static std::shared_ptr<LayerNormImpl> create_from_config(const LayerConfig &config);
};

}  // namespace internal

class LayerNorm : public LayerRef<internal::LayerNormImpl> {
public:
  explicit LayerNorm(size_t normalized_shape, float epsilon = 1e-5f, bool affine = true,
                     const std::string &name = "layer_norm")
      : LayerRef(
            std::make_shared<internal::LayerNormImpl>(normalized_shape, epsilon, affine, name)) {}

  using LayerRef<internal::LayerNormImpl>::LayerRef;
};

}  // namespace tunx
