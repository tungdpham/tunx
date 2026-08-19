/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <fmt/core.h>

#include <cstddef>
#include <memory>
#include <nlohmann/json.hpp>

#include "nn/functional_layer.hpp"
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

struct SequentialOp {
  static constexpr const char *TYPE_NAME = "sequential";
  struct Config {};

  static Vec<Tensor> forward(OpContext &ctx, const Vec<Tensor> &inputs, Vec<Layer> layers,
                             const Config &config);
  static Vec<Tensor> backward(OpContext &ctx, const Vec<Tensor> &grad_outputs, Vec<Layer> layers,
                              const Config &config);

  static LayerConfig get_config(const Config &config, const std::string &name);
  static Config parse_config(const LayerConfig &config) { return Config{}; }
  static Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config);
};

class Sequential : public FunctionalLayer<SequentialOp> {
public:
  explicit Sequential(Vec<Layer> layers = {}, const std::string &name = "sequential");

  explicit Sequential(std::initializer_list<Layer> layers, const std::string &name = "sequential")
      : Sequential(Vec<Layer>(layers), name) {}

  Vec<Layer> layers() const {
    Vec<Layer> res;
    for (const auto &l : impl_->layers()) {
      res.push_back(Layer(l));
    }
    return res;
  }

  void print_summary(const Vec<size_t> &input_shape) const;

  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const;
  LayerConfig get_config() const;

  static Layer create_from_config(const LayerConfig &config);
};

}  // namespace tunx