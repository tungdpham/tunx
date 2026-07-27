/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/activation.hpp"

#include <stdexcept>

#include "nn/activations.hpp"

namespace tunx {

Vec<Vec<size_t>> ActivationOp::output_shapes(const Vec<Vec<size_t>> &input_shapes,
                                             const Config &config) {
  if (input_shapes.size() != 1) {
    throw std::runtime_error("ActivationOp: expected exactly 1 input");
  }
  return {input_shapes[0]};
}

Tensor ActivationOp::forward(OpContext &ctx, const Tensor &input, const Config &config) {
  if (ctx.is_training) {
    ctx.residuals["input"] = input;
  }

  Tensor output = ctx.make_tensor(input.shape(), input.dtype());

  ActivationFactory::register_defaults();
  auto activation = ActivationFactory::create(config.activation_type);
  if (!activation) {
    throw std::invalid_argument("Activation function cannot be null");
  }
  activation->apply(input, output);

  return output;
}

Tensor ActivationOp::backward(OpContext &ctx, const Tensor &grad_output, const Config &config) {
  const Tensor &input = ctx.residuals["input"];

  Tensor grad_input = ctx.make_tensor(input.shape(), input.dtype());

  ActivationFactory::register_defaults();
  auto activation = ActivationFactory::create(config.activation_type);
  if (!activation) {
    throw std::invalid_argument("Activation function cannot be null");
  }
  activation->compute_gradient(input, grad_output, grad_input);

  return grad_input;
}

LayerConfig ActivationOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.type = TYPE_NAME;
  lcfg.name = name;
  lcfg.set("activation", config.activation_type);
  return lcfg;
}

}  // namespace tunx
