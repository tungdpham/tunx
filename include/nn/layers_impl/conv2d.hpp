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
#include "type/type.hpp"

namespace tunx {

// [N, H, W, C] input
namespace internal {
class Conv2DImpl : public SISOLayerImpl {
private:
  size_t in_channels_;
  size_t out_channels_;
  size_t kernel_h_;
  size_t kernel_w_;
  size_t stride_h_;
  size_t stride_w_;
  size_t pad_h_;
  size_t pad_w_;
  bool use_bias_;

  Param weights_;
  Param bias_;

  void init_impl() override;
  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  static constexpr const char *TYPE_NAME = "conv2d";

  Conv2DImpl(size_t in_channels, size_t out_channels, size_t kernel_h, size_t kernel_w,
             size_t stride_h = 1, size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0,
             bool use_bias = true, const std::string &name = "conv2d");

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;
  static std::shared_ptr<Conv2DImpl> create_from_config(const LayerConfig &config);
};

}  // namespace internal

class Conv2D : public LayerRef<internal::Conv2DImpl> {
public:
  Conv2D(size_t in_channels, size_t out_channels, size_t kernel_h, size_t kernel_w,
         size_t stride_h = 1, size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0,
         bool use_bias = true, const std::string &name = "conv2d")
      : LayerRef(std::make_shared<internal::Conv2DImpl>(in_channels, out_channels, kernel_h,
                                                        kernel_w, stride_h, stride_w, pad_h, pad_w,
                                                        use_bias, name)) {}

  using LayerRef<internal::Conv2DImpl>::LayerRef;
};

}  // namespace tunx
