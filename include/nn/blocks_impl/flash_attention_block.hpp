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
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

struct FlashAttentionBlockOp {
  static constexpr const char *TYPE_NAME = "flash_attention_block";

  struct Config {
    size_t embed_dim;
    size_t num_heads;
    bool is_causal = true;
  };

  static Tensor forward(OpContext &ctx, const Tensor &input, Layer q_proj, Layer k_proj,
                        Layer v_proj, Layer out_proj, const Config &config);

  static Vec<Tensor> backward(OpContext &ctx, const Tensor &grad_output, Layer q_proj, Layer k_proj,
                              Layer v_proj, Layer out_proj, const Config &config);

  static LayerConfig get_config(const Config &config, const std::string &name);
  static Config parse_config(const LayerConfig &config);
  static Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config);
};

class FlashAttentionBlock : public FunctionalLayer<FlashAttentionBlockOp> {
public:
  FlashAttentionBlock(size_t embed_dim, size_t num_heads, bool is_causal = true,
                      const std::string &name = "flash_attention_block");

  static Layer create_from_config(const LayerConfig &config);
};

}  // namespace tunx
