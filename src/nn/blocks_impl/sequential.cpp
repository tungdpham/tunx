/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/blocks_impl/sequential.hpp"

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "nlohmann/json_fwd.hpp"
#include "nn/layer.hpp"
#include "nn/layer_factory.hpp"
#include "type/type.hpp"

namespace tunx {
Vec<Tensor> SequentialOp::forward(OpContext &ctx, const Vec<Tensor> &inputs, Vec<Layer> layers, const Config &config) {
  if (layers.empty()) {
    throw std::runtime_error("Cannot forward through empty sequential model");
  }
  Vec<Tensor> current_inputs = inputs;
  Vec<Tensor> current_outputs;
  if (layers.size() % 2 == 0) {
    // assuming we are on the reverse side of input, flip so output of last layer is always opposite
    // side of input.
    ctx.ws_allocator->flip();
  }
  for (size_t i = 0; i < layers.size(); ++i) {
    current_outputs = layers[i].forward(current_inputs, ctx.residuals["layer_" + std::to_string(i)]);
    current_inputs = Vec<Tensor>(current_outputs.begin(), current_outputs.end());
    if (i != layers.size() - 1) {
      ctx.ws_allocator->flip();
    }
  }
  return current_outputs;
}

Vec<Tensor> SequentialOp::backward(OpContext &ctx, const Vec<Tensor> &grad_outputs, Vec<Layer> layers, const Config &config) {
  if (layers.empty()) {
    throw std::runtime_error("Cannot backward through empty sequential model");
  }
  Vec<Tensor> current_gradients = grad_outputs;
  Vec<Tensor> grad_inputs;
  if (layers.size() % 2 == 0) {
    // flip so grad output of last layer is always opposite side of input.
    ctx.ws_allocator->flip();
  }
  for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
    grad_inputs = layers[i].backward(current_gradients, ctx.residuals["layer_" + std::to_string(i)]);
    current_gradients = Vec<Tensor>(grad_inputs.begin(), grad_inputs.end());
    if (i != 0) {
      ctx.ws_allocator->flip();  // algorithm 1 definitely applies
    }
  }
  return grad_inputs;
}

Sequential::Sequential(Vec<Layer> layers, const std::string &name)
    : FunctionalLayer(SequentialOp::Config{}, name) {
  for (auto &layer : layers) {
    impl_->register_layer(std::move(layer));
  }
}

Vec<Vec<size_t>> SequentialOp::output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config) {
  // We can't implement output_shapes statically easily without layers list.
  // Wait, FunctionalLayer overrides output_shapes, but Sequential needs it directly.
  return input_shapes; 
}

Vec<Vec<size_t>> Sequential::output_shapes(const Vec<Vec<size_t>> &input_shapes) const {
  if (impl_->layers().empty()) {
    return input_shapes;
  }

  Vec<Vec<size_t>> current_shapes = input_shapes;
  for (const auto &layer : impl_->layers()) {
    current_shapes = layer->output_shapes(current_shapes);
  }

  return current_shapes;
}

void Sequential::print_summary(const Vec<size_t> &input_shape) const {
  if (impl_->layers().empty()) {
    std::cout << "Empty model.\n";
    return;
  }

  auto format_shape = [](const Vec<size_t> &shape) {
    std::string shape_str = "(";
    for (size_t j = 0; j < shape.size(); ++j) {
      if (j > 0) shape_str += ",";
      shape_str += std::to_string(shape[j]);
    }
    shape_str += ")";
    return shape_str;
  };

  std::cout << std::string(100, '=') << "\n";
  std::cout << "Model Summary: " << impl_->name() << "\n";
  std::cout << std::string(100, '=') << "\n";
  std::cout << std::left << std::setw(20) << "Layer (Type)" << std::setw(20) << "Input Shape"
            << std::setw(20) << "Output Shape" << "\n";

  Vec<size_t> current_shape = input_shape;
  for (size_t i = 0; i < impl_->layers().size(); ++i) {
    const auto &layer = impl_->layers()[i];
    std::cout << std::left << std::setw(20)
              << (layer->get_config().name.empty() ? layer->type() : layer->get_config().name);

    std::cout << std::setw(20) << format_shape(current_shape);

    auto output_shape = layer->output_shapes({current_shape})[0];
    std::cout << std::setw(20) << format_shape(output_shape) << "\n";
    current_shape = layer->output_shapes({current_shape})[0];
  }
  std::cout << std::string(100, '-') << "\n";
}

LayerConfig SequentialOp::get_config(const Config &config, const std::string &name) {
  LayerConfig cfg;
  cfg.name = name;
  cfg.type = TYPE_NAME;
  return cfg;
}

LayerConfig Sequential::get_config() const {
  LayerConfig config = impl_->get_config();
  nlohmann::json layers_config = nlohmann::json::array();
  for (const auto &layer : impl_->layers()) {
    auto layer_config = layer->get_config();
    layers_config.push_back(layer_config.to_json());
  }
  config.set("layers", layers_config);
  return config;
}

Layer Sequential::create_from_config(const LayerConfig &config) {
  Vec<Layer> layers;
  nlohmann::json layers_json = config.get<nlohmann::json>("layers", nlohmann::json::array());
  if (!layers_json.is_array()) {
    throw std::runtime_error("Sequential layer config 'layers' parameter must be an array");
  }
  LayerFactory::register_defaults();
  for (const auto &layer_json : layers_json) {
    LayerConfig layer_config = LayerConfig::from_json(layer_json);
    auto layer = LayerFactory::create(layer_config);
    layers.push_back(std::move(layer));
  }
  return tunx::Layer(Sequential(std::move(layers), config.name));
}

}  // namespace tunx
