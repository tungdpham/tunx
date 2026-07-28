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
struct ClassTokenOp {
  static constexpr const char *TYPE_NAME = "class_token";

  struct Config {
    size_t embed_dim;
  };

  static void init(OpContext &ctx, const Config &config);
  static Tensor forward(OpContext &ctx, const Tensor &input, const Param &class_token,
                        const Config &config);
  static Tensor backward(OpContext &ctx, const Tensor &grad_output, Param &class_token,
                         const Config &config);
  static LayerConfig get_config(const Config &config, const std::string &name);
  static Config parse_config(const LayerConfig &config);
  static Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config);
};

class ClassToken : public FunctionalLayer<ClassTokenOp> {
public:
  ClassToken(size_t embed_dim, const std::string &name = "class_token")
      : FunctionalLayer(ClassTokenOp::Config{embed_dim}, name) {}
};
}  // namespace tunx
