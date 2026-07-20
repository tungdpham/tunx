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
class ClassTokenImpl : public SISOLayerImpl {
private:
  size_t embed_dim_;
  Param class_token_;

  void init_impl() override;
  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  explicit ClassTokenImpl(size_t embed_dim, const std::string &name = "class_token");

  static constexpr const char *TYPE_NAME = "class_token";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;

  static std::shared_ptr<ClassTokenImpl> create_from_config(const LayerConfig &config);
};

}  // namespace internal

class ClassToken : public LayerRef<internal::ClassTokenImpl> {
public:
  explicit ClassToken(size_t embed_dim, const std::string &name = "class_token")
      : LayerRef(std::make_shared<internal::ClassTokenImpl>(embed_dim, name)) {}

  using LayerRef<internal::ClassTokenImpl>::LayerRef;
};

}  // namespace tunx
