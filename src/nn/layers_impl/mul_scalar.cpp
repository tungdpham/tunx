/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/mul_scalar.hpp"

#include <stdexcept>

#include "tensor/ops.hpp"
#include "type/type.hpp"

namespace tunx {

Vec<Vec<size_t>> MulScalarOp::output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config) {
  if (input_shapes.size() != 1) {
    throw std::runtime_error("MulScalarOp: expected exactly 1 input");
  }
  return {input_shapes[0]};
}

Tensor MulScalarOp::forward(OpContext &ctx, const Tensor &input, const Config &config) {
  Tensor output = ctx.make_tensor(input.shape(), input.dtype());
  mul_scalar(input, config.scalar, output, ctx.handle.get_stream());
  return output;
}

Tensor MulScalarOp::backward(OpContext &ctx, const Tensor &grad_out, const Config &config) {
  Tensor grad_input = ctx.make_tensor(grad_out.shape(), ctx.io_dtype);
  mul_scalar(grad_out, config.scalar, grad_input, ctx.handle.get_stream());
  return grad_input;
}

LayerConfig MulScalarOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.type = TYPE_NAME;
  lcfg.name = name;
  lcfg.set("scalar", config.scalar);
  return lcfg;
}

}  // namespace tunx
