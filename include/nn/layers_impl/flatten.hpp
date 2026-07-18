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

namespace tunx {

namespace internal {
class FlattenImpl : public SISOLayerImpl {
private:
  int start_dim_;
  int end_dim_;

  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  explicit FlattenImpl(int start_dim = 1, int end_dim = -1,
                            const std::string &name = "flatten");

  static constexpr const char *TYPE_NAME = "flatten";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;
  static std::shared_ptr<FlattenImpl> create_from_config(const LayerConfig &config);
};

}  // namespace internal

class Flatten : public LayerRef<internal::FlattenImpl> {
public:
  explicit Flatten(int start_dim = 1, int end_dim = -1, const std::string &name = "flatten")
      : LayerRef(std::make_shared<internal::FlattenImpl>(start_dim, end_dim, name)) {}

  using LayerRef<internal::FlattenImpl>::LayerRef;
};

}  // namespace tunx
