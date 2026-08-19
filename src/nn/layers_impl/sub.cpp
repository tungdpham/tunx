/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/sub.hpp"

#include <stdexcept>

#include "tensor/ops.hpp"
#include "type/type.hpp"

namespace tunx {

Vec<Vec<size_t>> SubOp::output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config) {
  if (input_shapes.size() != 2) {
    throw std::runtime_error("SubOp: expected exactly 2 inputs");
  }
  if (input_shapes[0] != input_shapes[1]) {
    throw std::runtime_error("SubOp: both inputs must have the same shape");
  }
  return {input_shapes[0]};
}

Tensor SubOp::forward(OpContext &ctx, const Tensor &a, const Tensor &b) {
  if (a.shape() != b.shape()) {
    throw std::runtime_error("SubOp: both inputs must have the same shape");
  }

  Tensor output = ctx.make_tensor(a.shape(), a.dtype());
  sub(a, b, output, ctx.handle.get_stream());

  return output;
}

Vec<Tensor> SubOp::backward(OpContext &ctx, const Tensor &grad_out) {
  Tensor grad_a = ctx.make_tensor(grad_out.shape(), ctx.io_dtype);
  Tensor grad_b = ctx.make_tensor(grad_out.shape(), ctx.io_dtype);

  copy(grad_out, grad_a, ctx.handle.get_stream());
  mul_scalar(grad_out, -1.0, grad_b, ctx.handle.get_stream());

  return {grad_a, grad_b};
}

LayerConfig SubOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.type = TYPE_NAME;
  lcfg.name = name;
  return lcfg;
}

}  // namespace tunx
