/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <chrono>
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
                        name) {
    impl_->register_param(
        "weights", {input_features, output_features}, [input_features](Param &p, OpContext &ctx) {
          float stddev = static_cast<float>(1.0 / std::sqrt(static_cast<double>(input_features)));
          long long seed = ctx.use_seed
                               ? ctx.srand_seed
                               : std::chrono::system_clock::now().time_since_epoch().count();
          fill_normal(p.data(), 0, stddev, seed);
        });
    if (use_bias) {
      impl_->register_param("bias", {output_features}, [input_features](Param &p, OpContext &ctx) {
        float stddev = static_cast<float>(1.0 / std::sqrt(static_cast<double>(input_features)));
        long long seed = ctx.use_seed ? ctx.srand_seed
                                      : std::chrono::system_clock::now().time_since_epoch().count();
        fill_normal(p.data(), 0, stddev, seed);
      });
    } else {
      impl_->register_param("bias_dummy", {0}, nullptr);
    }
  }
};
}  // namespace tunx
