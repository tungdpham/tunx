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

struct BatchNormOp {
  static constexpr const char *TYPE_NAME = "batchnorm";

  struct Config {
    size_t num_features;
    float epsilon = 1e-5f;
    float momentum = 0.1f;
    bool affine = true;
    bool use_relu = false;
  };

  static void init(OpContext &ctx, const Config &config);
  static Tensor forward(OpContext &ctx, const Tensor &input, const Param &gamma, const Param &beta,
                        Param &running_mean, Param &running_var, const Config &config);
  static Tensor backward(OpContext &ctx, const Tensor &grad_output, Param &gamma, Param &beta,
                         const Config &config);

  static LayerConfig get_config(const Config &config, const std::string &name);
  static Config parse_config(const LayerConfig &config);
  static Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config);
};

class BatchNorm : public FunctionalLayer<BatchNormOp> {
public:
  BatchNorm(size_t num_features, float epsilon = 1e-5f, float momentum = 0.1f, bool affine = true,
            bool use_relu = false, const std::string &name = "batchnorm")
      : FunctionalLayer(BatchNormOp::Config{num_features, epsilon, momentum, affine, use_relu},
                        name) {}
};

}  // namespace tunx
