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

struct ActivationOp {
  static constexpr const char *TYPE_NAME = "activation";

  struct Config {
    std::string activation_type = "relu";
  };

  static Tensor forward(OpContext &ctx, const Tensor &input, const Config &config);
  static Tensor backward(OpContext &ctx, const Tensor &grad_output, const Config &config);

  static LayerConfig get_config(const Config &config, const std::string &name);
  static Config parse_config(const LayerConfig &config) {
    return Config{config.get<std::string>("activation", "relu")};
  }
  static Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config);
};

using Activation = FunctionalLayer<ActivationOp>;

}  // namespace tunx
