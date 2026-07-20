/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "nn/param.hpp"
#include "nn/siso_layer.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace tunx {

namespace internal {
class BatchNormImpl : public SISOLayerImpl {
private:
  size_t num_features_;
  float epsilon_;
  float momentum_;
  bool affine_;
  bool use_relu_;

  Param gamma_;
  Param beta_;

  Param running_mean_;
  Param running_var_;

  void init_impl() override;
  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  explicit BatchNormImpl(size_t num_features, float epsilon = 1e-5f, float momentum = 0.1f,
                         bool affine = true, bool use_relu = false,
                         const std::string &name = "batchnorm");
  ~BatchNormImpl() override;

  static constexpr const char *TYPE_NAME = "batchnorm";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  static std::shared_ptr<BatchNormImpl> create_from_config(const LayerConfig &config);
  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;
};

}  // namespace internal

class BatchNorm : public LayerRef<internal::BatchNormImpl> {
public:
  BatchNorm(size_t num_features, float epsilon = 1e-5f, float momentum = 0.1f, bool affine = true,
            bool use_relu = false, const std::string &name = "batchnorm")
      : LayerRef(std::make_shared<internal::BatchNormImpl>(num_features, epsilon, momentum, affine,
                                                           use_relu, name)) {}

  using LayerRef<internal::BatchNormImpl>::LayerRef;
};

}  // namespace tunx
