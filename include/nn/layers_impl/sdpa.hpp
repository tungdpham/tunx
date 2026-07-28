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
#include "tensor/tensor.hpp"

namespace tunx {

struct SDPAOp {
  static constexpr const char *TYPE_NAME = "sdpa";

  struct Config {
    float attn_scale = 1.0f;
    bool is_causal = false;
  };

  static Tensor forward(OpContext &ctx, const Tensor &q, const Tensor &k, const Tensor &v,
                        const Config &config);

  static Vec<Tensor> backward(OpContext &ctx, const Tensor &grad_output, const Config &config);

  static LayerConfig get_config(const Config &config, const std::string &name);
  static Config parse_config(const LayerConfig &config);
  static Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config);
};

class SDPA : public FunctionalLayer<SDPAOp> {
public:
  static constexpr const char *TYPE_NAME = "sdpa";

  SDPA(float attn_scale = 1.0f, bool is_causal = false, const std::string &name = "sdpa");
};

}  // namespace tunx
