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

namespace tunx {

namespace internal {
class LegacyBatchNormImpl : public SISOLayerImpl {
private:
  size_t num_features_;
  float epsilon_;
  float momentum_;
  bool affine_;

  Tensor gamma_;
  Tensor beta_;
  Tensor grad_gamma_;
  Tensor grad_beta_;

  Tensor running_mean_;
  Tensor running_var_;
  Tensor grad_dummy_mean_;
  Tensor grad_dummy_var_;

  Tensor def_forward(const Tensor &input, Residuals &residuals);
  Tensor def_backward(const Tensor &grad_output, Residuals &residuals);

  void init_impl() override;
  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  explicit LegacyBatchNormImpl(size_t num_features, float epsilon = 1e-5f,
                                    float momentum = 0.1f, bool affine = true,
                                    const std::string &name = "batchnorm");

  static constexpr const char *TYPE_NAME = "legacy_batchnorm";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;


  static std::shared_ptr<LegacyBatchNormImpl> create_from_config(const LayerConfig &config);
};

}  // namespace internal

class LegacyBatchNorm : public LayerRef<internal::LegacyBatchNormImpl> {
public:
  explicit LegacyBatchNorm(size_t num_features, float epsilon = 1e-5f, float momentum = 0.1f,
                                bool affine = true, const std::string &name = "batchnorm")
      : LayerRef(std::make_shared<internal::LegacyBatchNormImpl>(num_features, epsilon, momentum, affine,
                                                            name)) {}

  using LayerRef<internal::LegacyBatchNormImpl>::LayerRef;
};

}  // namespace tunx
