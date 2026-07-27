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
struct PositionalEmbeddingOp {
  static constexpr const char *TYPE_NAME = "pos_embedding";

  struct Config {
    size_t embed_dim;
    size_t seq_len;
  };

  static Tensor forward(OpContext &ctx, const Tensor &input, const Param &pos_embedding,
                        const Config &config);
  static Tensor backward(OpContext &ctx, const Tensor &grad_output, Param &pos_embedding,
                         const Config &config);
  static LayerConfig get_config(const Config &config, const std::string &name);
  static Config parse_config(const LayerConfig &config);
  static Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config);
};

class PositionalEmbedding : public FunctionalLayer<PositionalEmbeddingOp> {
public:
  PositionalEmbedding(size_t embed_dim, size_t seq_len, const std::string &name = "pos_embedding")
      : FunctionalLayer(PositionalEmbeddingOp::Config{embed_dim, seq_len}, name) {
    impl_->register_param(
        "pos_embedding", {seq_len, embed_dim}, [embed_dim](Param &p, OpContext &ctx) {
          float bound = static_cast<float>(1.0 / std::sqrt(static_cast<double>(embed_dim)));
          long long seed = ctx.use_seed
                               ? ctx.srand_seed
                               : std::chrono::system_clock::now().time_since_epoch().count();
          fill_normal(p.data(), 0, bound, seed);
        });
  }
};
}  // namespace tunx
