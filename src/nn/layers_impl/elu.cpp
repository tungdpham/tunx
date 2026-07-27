/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/elu.hpp"

#include <stdexcept>

#include "nn/activations_impl/elu.hpp"

namespace tunx {

Vec<Vec<size_t>> ELUOp::output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config) {
  if (input_shapes.size() != 1) {
    throw std::runtime_error("ELUOp: expected exactly 1 input");
  }
  return {input_shapes[0]};
}

Tensor ELUOp::forward(OpContext &ctx, const Tensor &input, const Config &config) {
  if (ctx.is_training) {
    ctx.residuals["input"] = input;
  }

  Tensor output = ctx.make_tensor(input.shape(), ctx.io_dtype);
  func::ELU(config.alpha).apply(input, output);
  return output;
}

Tensor ELUOp::backward(OpContext &ctx, const Tensor &grad_output, const Config &config) {
  const Tensor &input = ctx.residuals["input"];

  Tensor grad_input = ctx.make_tensor(input.shape(), ctx.io_dtype);
  func::ELU(config.alpha).compute_gradient(input, grad_output, grad_input);
  return grad_input;
}

LayerConfig ELUOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.type = TYPE_NAME;
  lcfg.name = name;
  lcfg.set("alpha", config.alpha);
  return lcfg;
}

}  // namespace tunx
