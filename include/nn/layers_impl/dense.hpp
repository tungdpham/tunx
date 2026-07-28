/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <string>

#include "nn/functional_layer.hpp"
#include "nn/param.hpp"
#include "tensor/tensor.hpp"

namespace tunx {
struct DenseOp {
  static constexpr const char *TYPE_NAME = "dense";

  struct Config {
    size_t input_features;
    size_t output_features;
    bool use_bias = true;
  };

  static void init(OpContext &ctx, const Config &config);
  static Tensor forward(OpContext &ctx, const Tensor &input, const Param &weights,
                        const Param &bias, const Config &config);
  static Tensor backward(OpContext &ctx, const Tensor &grad_output, Param &weights, Param &bias,
                         const Config &config);
  static LayerConfig get_config(const Config &config, const std::string &name);
  static Config parse_config(const LayerConfig &config);
  static Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config);
};

class Dense : public FunctionalLayer<DenseOp> {
public:
  Dense(size_t input_features, size_t output_features, bool use_bias = true,
        const std::string &name = "dense")
      : FunctionalLayer(DenseOp::Config{input_features, output_features, use_bias},
                        name) {}
};
}  // namespace tunx
