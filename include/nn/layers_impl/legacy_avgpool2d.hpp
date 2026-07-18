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

#include "nn/siso_layer.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

namespace internal {
class LegacyAvgPool2DImpl : public SISOLayerImpl {
private:
  size_t pool_h_;
  size_t pool_w_;
  size_t stride_h_;
  size_t stride_w_;
  size_t pad_h_;
  size_t pad_w_;

  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  LegacyAvgPool2DImpl(size_t pool_h, size_t pool_w, size_t stride_h = 1, size_t stride_w = 1,
                           size_t pad_h = 0, size_t pad_w = 0,
                           const std::string &name = "avgpool2d");

  static constexpr const char *TYPE_NAME = "legacy_avgpool2d";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;
  static std::shared_ptr<LegacyAvgPool2DImpl> create_from_config(const LayerConfig &config);
};

}  // namespace internal

class LegacyAvgPool2D : public LayerRef<internal::LegacyAvgPool2DImpl> {
public:
  LegacyAvgPool2D(size_t pool_h, size_t pool_w, size_t stride_h = 1, size_t stride_w = 1,
                       size_t pad_h = 0, size_t pad_w = 0, const std::string &name = "avgpool2d")
      : LayerRef(std::make_shared<internal::LegacyAvgPool2DImpl>(pool_h, pool_w, stride_h, stride_w,
                                                            pad_h, pad_w, name)) {}

  using LayerRef<internal::LegacyAvgPool2DImpl>::LayerRef;
};

}  // namespace tunx
