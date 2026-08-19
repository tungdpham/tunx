/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/linear.hpp"

#include "tensor/ops.hpp"

namespace tunx {

Vec<Vec<size_t>> LinearOp::output_shapes(const Vec<Vec<size_t>> &input_shapes,
                                         const Config &config) {
  return input_shapes;
}

Tensor LinearOp::forward(OpContext &ctx, const Tensor &input, const Config &config) {
  Tensor output = ctx.make_tensor(input.shape(), ctx.io_dtype);
  copy(input, output, ctx.handle.get_stream());
  return output;
}

Tensor LinearOp::backward(OpContext &ctx, const Tensor &grad_output, const Config &config) {
  Tensor grad_input = ctx.make_tensor(grad_output.shape(), ctx.io_dtype);
  copy(grad_output, grad_input, ctx.handle.get_stream());
  return grad_input;
}

LayerConfig LinearOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.name = name;
  lcfg.type = TYPE_NAME;
  return lcfg;
}

LinearOp::Config LinearOp::parse_config(const LayerConfig &config) { return Config{}; }

}  // namespace tunx
