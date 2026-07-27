/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <string>

#include "nn/functional_layer.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

struct LegacyConv2DOp {
  static constexpr const char *TYPE_NAME = "legacy_conv2d";

  struct Config {
    size_t in_channels;
    size_t out_channels;
    size_t kernel_h;
    size_t kernel_w;
    size_t stride_h = 1;
    size_t stride_w = 1;
    size_t pad_h = 0;
    size_t pad_w = 0;
    bool use_bias = true;
  };

  static Tensor forward(OpContext &ctx, const Tensor &input, const Param &weights, const Param &bias, const Config &config);
  static Tensor backward(OpContext &ctx, const Tensor &grad_output, Param &weights, Param &bias, const Config &config);

  static LayerConfig get_config(const Config &config, const std::string &name);
  static Config parse_config(const LayerConfig &config);
  static Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config);
};

class LegacyConv2D : public FunctionalLayer<LegacyConv2DOp> {
public:
  LegacyConv2D(size_t in_channels, size_t out_channels, size_t kernel_h, size_t kernel_w,
               size_t stride_h = 1, size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0,
               bool use_bias = true, const std::string &name = "legacy_conv2d");
};

}  // namespace tunx
