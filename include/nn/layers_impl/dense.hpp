/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "nn/param.hpp"
#include "nn/siso_layer.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

namespace internal {
class DenseImpl : public SISOLayerImpl {
private:
  size_t input_features_;
  size_t output_features_;
  bool use_bias_;
  Param weights_;
  Param bias_;

  void init_impl() override;
  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  DenseImpl(size_t input_features, size_t output_features, bool use_bias = true,
            const std::string &name = "dense");

  ~DenseImpl();

  static constexpr const char *TYPE_NAME = "dense";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;

  static std::shared_ptr<DenseImpl> create_from_config(const LayerConfig &config);
};

}  // namespace internal

class Dense : public LayerRef<internal::DenseImpl> {
public:
  Dense(size_t input_features, size_t output_features, bool use_bias = true,
        const std::string &name = "dense")
      : LayerRef(std::make_shared<internal::DenseImpl>(input_features, output_features, use_bias,
                                                       name)) {}

  using LayerRef<internal::DenseImpl>::LayerRef;
};

}  // namespace tunx
