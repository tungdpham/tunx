/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/tanh.hpp"

#include <stdexcept>

#include "nn/activations_impl/tanh.hpp"

namespace tunx {

Vec<Vec<size_t>> TanhOp::output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config) {
  if (input_shapes.size() != 1) {
    throw std::runtime_error("TanhOp: expected exactly 1 input");
  }
  return {input_shapes[0]};
}

Tensor TanhOp::forward(OpContext &ctx, const Tensor &input) {
  if (ctx.is_training) {
    ctx.residuals["input"] = input;
  }

  Tensor output = ctx.make_tensor(input.shape(), ctx.io_dtype);
  func::Tanh().apply(input, output, ctx.handle.get_stream());

  return output;
}

Tensor TanhOp::backward(OpContext &ctx, const Tensor &grad_output) {
  const Tensor &input = ctx.residuals["input"];

  Tensor grad_input = ctx.make_tensor(grad_output.shape(), ctx.io_dtype);

  func::Tanh().compute_gradient(input, grad_output, grad_input, ctx.handle.get_stream());

  return grad_input;
}

LayerConfig TanhOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.type = TYPE_NAME;
  lcfg.name = name;
  return lcfg;
}

}  // namespace tunx
