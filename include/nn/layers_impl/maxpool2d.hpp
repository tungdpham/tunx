/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>
#include <string>

#include "nn/functional_layer.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

struct MaxPool2DOp {
  static constexpr const char *TYPE_NAME = "maxpool2d";

  struct Config {
    size_t pool_h;
    size_t pool_w;
    size_t stride_h = 1;
    size_t stride_w = 1;
    size_t pad_h = 0;
    size_t pad_w = 0;
  };

  static Tensor forward(OpContext &ctx, const Tensor &input, const Config &config);
  static Tensor backward(OpContext &ctx, const Tensor &grad_output, const Config &config);

  static LayerConfig get_config(const Config &config, const std::string &name);
  static Config parse_config(const LayerConfig &config);
  static Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config);
};

class MaxPool2D : public FunctionalLayer<MaxPool2DOp> {
public:
  MaxPool2D(size_t pool_h, size_t pool_w, size_t stride_h = 0, size_t stride_w = 0,
            size_t pad_h = 0, size_t pad_w = 0, const std::string &name = "maxpool2d")
      : FunctionalLayer(
            MaxPool2DOp::Config{pool_h, pool_w, stride_h == 0 ? pool_h : stride_h,
                                          stride_w == 0 ? pool_w : stride_w, pad_h, pad_w},
            name) {}
};

}  // namespace tunx
