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

struct LegacyBatchNormOp {
  static constexpr const char *TYPE_NAME = "legacy_batchnorm";

  struct Config {
    size_t num_features;
    float epsilon = 1e-5f;
    float momentum = 0.1f;
    bool affine = true;
  };

  static Tensor forward(OpContext &ctx, const Tensor &input, Param &gamma, Param &beta,
                        Param &running_mean, Param &running_var, const Config &config);
  static Tensor backward(OpContext &ctx, const Tensor &grad_output, Param &gamma, Param &beta,
                         const Config &config);

  static LayerConfig get_config(const Config &config, const std::string &name);
  static Config parse_config(const LayerConfig &config);
  static Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config);
};

class LegacyBatchNorm : public FunctionalLayer<LegacyBatchNormOp> {
public:
  explicit LegacyBatchNorm(size_t num_features, float epsilon = 1e-5f, float momentum = 0.1f,
                           bool affine = true, const std::string &name = "batchnorm");
};

}  // namespace tunx
