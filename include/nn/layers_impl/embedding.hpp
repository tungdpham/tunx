/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cmath>
#include <string>

#include "nn/functional_layer.hpp"
#include "nn/param.hpp"
#include "tensor/tensor.hpp"

namespace tunx {
struct EmbeddingOp {
  static constexpr const char *TYPE_NAME = "embedding";

  struct Config {
    size_t vocab_size;
    size_t embed_dim;
    size_t padding_idx = static_cast<size_t>(-1);
  };

  static void init(OpContext &ctx, const Config &config);
  static Tensor forward(OpContext &ctx, const Tensor &input, const Param &weight,
                        const Config &config);
  static Tensor backward(OpContext &ctx, const Tensor &grad_output, Param &weight,
                         const Config &config);
  static LayerConfig get_config(const Config &config, const std::string &name);
  static Config parse_config(const LayerConfig &config);
  static Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config);
};

class Embedding : public FunctionalLayer<EmbeddingOp> {
public:
  Embedding(size_t vocab_size, size_t embed_dim, size_t padding_idx = static_cast<size_t>(-1),
            const std::string &name = "embedding")
      : FunctionalLayer(EmbeddingOp::Config{vocab_size, embed_dim, padding_idx}, name) {}
};
}  // namespace tunx
