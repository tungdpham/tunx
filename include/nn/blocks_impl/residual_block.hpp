/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include "nn/functional_layer.hpp"
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

struct ResidualBlockOp {
  static constexpr const char *TYPE_NAME = "residual_block";
  struct Config {
    std::string activation = "relu";
    bool has_shortcut = false;
  };

  static Vec<Tensor> forward(OpContext &ctx, const Vec<Tensor> &inputs, Vec<Layer> layers, const Config &config);
  static Vec<Tensor> backward(OpContext &ctx, const Vec<Tensor> &grad_outputs, Vec<Layer> layers, const Config &config);

  static LayerConfig get_config(const Config &config, const std::string &name, const Vec<std::shared_ptr<LayerImpl>> &registered_layers);
  static Config parse_config(const LayerConfig &config);
  static Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config, const Vec<std::shared_ptr<LayerImpl>> &registered_layers);
};

class ResidualBlock : public FunctionalLayer<ResidualBlockOp> {
public:
  /**
   * @brief Constructs a residual block
   * @param main_path The main transformation path F(x) as a vector of layers
   * @param shortcut_path Optional projection path for dim matching (empty for identity)
   * @param final_activation Activation applied after addition (e.g., "relu")
   * @param name LayerImpl name
   */
  ResidualBlock(Vec<Layer> main_path, Vec<Layer> shortcut_path,
                const std::string &final_activation = "relu",
                const std::string &name = "residual_block");

  ResidualBlock(tunx::Layer main_path, tunx::Layer shortcut_path,
                const std::string &final_activation = "relu",
                const std::string &name = "residual_block");

  static Layer create_from_config(const LayerConfig &config);
};

}  // namespace tunx
