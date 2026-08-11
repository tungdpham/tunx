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

struct ConcatOp {
  static constexpr const char *TYPE_NAME = "concat";

  struct Config {
    int axis;
  };

  static Tensor forward(OpContext &ctx, const Vec<Tensor> &inputs, const Config &config);
  static Vec<Tensor> backward(OpContext &ctx, const Tensor &grad_output, const Config &config);
  static LayerConfig get_config(const Config &config, const std::string &name);
  static Config parse_config(const LayerConfig &config);
  static Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config);
};

class Concat : public FunctionalLayer<ConcatOp> {
public:
  Concat(int axis, const std::string &name = "concat")
      : FunctionalLayer(ConcatOp::Config{axis}, name) {}
};

}  // namespace tunx
