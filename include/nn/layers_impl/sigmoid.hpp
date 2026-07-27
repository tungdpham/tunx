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

struct SigmoidOp {
  static constexpr const char *TYPE_NAME = "sigmoid";

  struct Config {};

  static Tensor forward(OpContext &ctx, const Tensor &input);
  static Tensor backward(OpContext &ctx, const Tensor &grad_output);

  static LayerConfig get_config(const Config &config, const std::string &name);
  static Config parse_config(const LayerConfig &config) { return Config{}; }
  static Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config);
};

using Sigmoid = FunctionalLayer<SigmoidOp>;

}  // namespace tunx
