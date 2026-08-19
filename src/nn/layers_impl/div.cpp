/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/div.hpp"

#include <stdexcept>

#include "tensor/ops.hpp"
#include "type/type.hpp"

namespace tunx {

Vec<Vec<size_t>> DivOp::output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config) {
  if (input_shapes.size() != 2) {
    throw std::runtime_error("DivOp: expected exactly 2 inputs");
  }
  if (input_shapes[0] != input_shapes[1]) {
    throw std::runtime_error("DivOp: both inputs must have the same shape");
  }
  return {input_shapes[0]};
}

Tensor DivOp::forward(OpContext &ctx, const Tensor &a, const Tensor &b) {
  if (a.shape() != b.shape()) {
    throw std::runtime_error("DivOp: both inputs must have the same shape");
  }

  if (ctx.is_training) {
    ctx.residuals["a"] = a;
    ctx.residuals["b"] = b;
  }

  Tensor output = ctx.make_tensor(a.shape(), a.dtype());
  div(a, b, output, ctx.handle.get_stream());

  return output;
}

Vec<Tensor> DivOp::backward(OpContext &ctx, const Tensor &grad_out) {
  const Tensor &a = ctx.residuals["a"];
  const Tensor &b = ctx.residuals["b"];

  // grad_a = grad_out / b
  // grad_b = -(grad_out * a) / b^2
  Tensor grad_a = ctx.make_tensor(grad_out.shape(), ctx.io_dtype);
  Tensor grad_b = ctx.make_tensor(grad_out.shape(), ctx.io_dtype);

  // grad_a = grad_out / b
  div(grad_out, b, grad_a, ctx.handle.get_stream());

  Tensor b_sq = ctx.make_tensor(grad_out.shape(), ctx.io_dtype);
  mul(b, b, b_sq, ctx.handle.get_stream());

  Tensor numerator = ctx.make_tensor(grad_out.shape(), ctx.io_dtype);
  mul(grad_out, a, numerator, ctx.handle.get_stream());
  div(numerator, b_sq, grad_b, ctx.handle.get_stream());
  mul_scalar(grad_b, -1, grad_b, ctx.handle.get_stream());

  return {grad_a, grad_b};
}

LayerConfig DivOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.type = TYPE_NAME;
  lcfg.name = name;
  return lcfg;
}

}  // namespace tunx
