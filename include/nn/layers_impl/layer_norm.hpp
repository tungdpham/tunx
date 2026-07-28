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
#include "nn/param.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace tunx {

struct LayerNormOp {
  static constexpr const char *TYPE_NAME = "layer_norm";

  struct Config {
    size_t normalized_shape;
    float epsilon = 1e-5f;
    bool affine = true;
  };

  static void init(OpContext &ctx, const Config &config);
  static Tensor forward(OpContext &ctx, const Tensor &input, const Param &gamma, const Param &beta,
                        const Config &config);
  static Tensor backward(OpContext &ctx, const Tensor &grad_output, Param &gamma, Param &beta,
                         const Config &config);

  static LayerConfig get_config(const Config &config, const std::string &name);
  static Config parse_config(const LayerConfig &config);
  static Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config);
};

class LayerNorm : public FunctionalLayer<LayerNormOp> {
public:
  LayerNorm(size_t normalized_shape, float epsilon = 1e-5f, bool affine = true,
            const std::string &name = "layer_norm")
      : FunctionalLayer(LayerNormOp::Config{normalized_shape, epsilon, affine}, name) {}
};

}  // namespace tunx
