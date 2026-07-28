/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/blocks_impl/residual_block.hpp"

#include <alloca.h>

#include <cstddef>


#include "nn/layer.hpp"
#include "nn/layer_factory.hpp"
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

ResidualBlock::ResidualBlock(Vec<Layer> main_path, Vec<Layer> shortcut_path,
                             const std::string &final_activation, const std::string &name)
    : FunctionalLayer(ResidualBlockOp::Config{final_activation, !shortcut_path.empty()}, name) {
  if (main_path.empty()) {
    throw std::runtime_error("Main path of ResidualBlock cannot be empty.");
  }
  impl_->register_layer(static_cast<std::shared_ptr<LayerImpl>>(Layer(tunx::Sequential(std::move(main_path), name + "_main_path"))));

  if (!shortcut_path.empty()) {
    impl_->register_layer(static_cast<std::shared_ptr<LayerImpl>>(Layer(tunx::Sequential(std::move(shortcut_path), name + "_shortcut_path"))));
  }
  
  if (final_activation != "none" && final_activation != "linear") {
    impl_->register_layer(static_cast<std::shared_ptr<LayerImpl>>(Layer(tunx::Activation(ActivationOp::Config{final_activation}, name + "_act"))));
  }
}

ResidualBlock::ResidualBlock(tunx::Layer main_path, tunx::Layer shortcut_path,
                             const std::string &final_activation, const std::string &name)
    : FunctionalLayer(ResidualBlockOp::Config{final_activation, static_cast<bool>(shortcut_path)}, name) {
  if (!main_path) {
    throw std::runtime_error("Main path of ResidualBlock cannot be null.");
  }

  impl_->register_layer(static_cast<std::shared_ptr<LayerImpl>>(main_path));
  if (shortcut_path) {
    impl_->register_layer(static_cast<std::shared_ptr<LayerImpl>>(shortcut_path));
  }

  if (final_activation != "none" && final_activation != "linear") {
    impl_->register_layer(static_cast<std::shared_ptr<LayerImpl>>(Layer(tunx::Activation(ActivationOp::Config{final_activation}, name + "_act"))));
  }
}

Vec<Tensor> ResidualBlockOp::forward(OpContext &ctx, const Vec<Tensor> &inputs, Vec<Layer> layers, const Config &config) {
  // layers[0] is main_path
  Vec<Tensor> main_outputs = layers[0].forward(inputs, ctx.residuals["main_path"]);

  // Forward through shortcut path if present
  Vec<Tensor> shortcut_outputs = inputs;
  if (config.has_shortcut) {
    Vec<Tensor> shortcut_outputs_vec = layers[1].forward(inputs, ctx.residuals["shortcut_path"]);
    for (size_t i = 0; i < shortcut_outputs_vec.size(); ++i) {
      shortcut_outputs[i] = shortcut_outputs_vec[i];
    }
  }

  Vec<Tensor> outputs = main_outputs;  // reuse main path outputs for final output to save memory

  // Add outputs and apply final activation
  bool has_activation = config.activation != "none" && config.activation != "linear";
  Layer* act_layer = has_activation ? &layers.back() : nullptr;

  for (size_t i = 0; i < outputs.size(); ++i) {
    if (act_layer) {
      std::string pre_act_key = "pre_activation_" + std::to_string(i);
      Tensor pre_act = ctx.make_tensor(main_outputs[i].shape(), ctx.io_dtype);
      add(main_outputs[i], shortcut_outputs[i], pre_act);
      ctx.residuals[pre_act_key] = pre_act;
      outputs[i] = act_layer->forward({pre_act}, ctx.residuals[pre_act_key + "_act"])[0];
    } else {
      add(main_outputs[i], shortcut_outputs[i], outputs[i]);
    }
  }
  return outputs;
}

Vec<Tensor> ResidualBlockOp::backward(OpContext &ctx, const Vec<Tensor> &grad_outputs, Vec<Layer> layers, const Config &config) {
  bool has_activation = config.activation != "none" && config.activation != "linear";
  Layer* act_layer = has_activation ? &layers.back() : nullptr;

  Vec<Tensor> grads_to_propagate = grad_outputs;
  if (act_layer) {
    for (size_t i = 0; i < grad_outputs.size(); ++i) {
      std::string pre_act_key = "pre_activation_" + std::to_string(i);
      grads_to_propagate[i] = act_layer->backward({grad_outputs[i]}, ctx.residuals[pre_act_key + "_act"])[0];
      ctx.residuals[pre_act_key] = Tensor();  // release pre-activation cache
    }
    ctx.ws_allocator->flip();  // flip workspace allocator between main and shortcut backward
  }

  // Backward through main path
  Vec<Tensor> main_grad_inputs = layers[0].backward(grads_to_propagate, ctx.residuals["main_path"]);

  // Backward through shortcut path
  Vec<Tensor> shortcut_grad_inputs = grads_to_propagate;
  if (config.has_shortcut) {
    auto temp = layers[1].backward(grads_to_propagate, ctx.residuals["shortcut_path"]);
    shortcut_grad_inputs = Vec<Tensor>(temp.begin(), temp.end());
  }

  Vec<Tensor> grad_inputs(main_grad_inputs.size());
  for (size_t i = 0; i < grad_inputs.size(); ++i) {
    grad_inputs[i] = ctx.make_tensor(main_grad_inputs[i].shape(), main_grad_inputs[i].dtype());
    add(main_grad_inputs[i], shortcut_grad_inputs[i], grad_inputs[i]);
  }
  return grad_inputs;
}

Vec<Vec<size_t>> ResidualBlockOp::output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config, const Vec<std::shared_ptr<LayerImpl>> &registered_layers) {
  return registered_layers[0]->output_shapes(input_shapes);
}

ResidualBlockOp::Config ResidualBlockOp::parse_config(const LayerConfig &config) {
  Config cfg;
  cfg.activation = config.get<std::string>("activation", "relu");
  nlohmann::json shortcut_json = config.get<nlohmann::json>("shortcut_path", nlohmann::json::object());
  if (!shortcut_json.is_null() && !shortcut_json.empty()) {
    LayerConfig shortcut_config = LayerConfig::from_json(shortcut_json);
    nlohmann::json layers_json = shortcut_config.get<nlohmann::json>("layers", nlohmann::json::array());
    cfg.has_shortcut = (layers_json.is_array() && !layers_json.empty());
  }
  return cfg;
}

LayerConfig ResidualBlockOp::get_config(const Config &config, const std::string &name, const Vec<std::shared_ptr<LayerImpl>> &registered_layers) {
  LayerConfig cfg;
  cfg.name = name;
  cfg.type = TYPE_NAME;
  cfg.set("activation", config.activation);

  LayerConfig main_config = registered_layers[0]->get_config();
  cfg.set("main_path", main_config.to_json());
  if (config.has_shortcut) {
    LayerConfig shortcut_config = registered_layers[1]->get_config();
    cfg.set("shortcut_path", shortcut_config.to_json());
  } else {
    cfg.set("shortcut_path", nlohmann::json::object());
  }
  return cfg;
}

Layer ResidualBlock::create_from_config(const LayerConfig &config) {
  Layer main_path;
  Layer shortcut_path;
  nlohmann::json main_json = config.get<nlohmann::json>("main_path", nlohmann::json::object());
  LayerFactory::register_defaults();
  main_path = Sequential::create_from_config(LayerConfig::from_json(main_json));
  
  nlohmann::json shortcut_json =
      config.get<nlohmann::json>("shortcut_path", nlohmann::json::object());
  if (!shortcut_json.is_null() && !shortcut_json.empty()) {
    LayerConfig shortcut_config = LayerConfig::from_json(shortcut_json);
    nlohmann::json layers_json =
        shortcut_config.get<nlohmann::json>("layers", nlohmann::json::array());
    if (layers_json.is_array() && !layers_json.empty()) {
      shortcut_path = Sequential::create_from_config(shortcut_config);
    }
  }

  std::string activation = config.get<std::string>("activation", "relu");
  return tunx::Layer(ResidualBlock(std::move(main_path), std::move(shortcut_path), activation, config.name));
}

}  // namespace tunx
