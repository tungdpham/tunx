/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/add.hpp"

#include <stdexcept>

#include "tensor/ops.hpp"
#include "type/type.hpp"

namespace tunx {

Vec<Vec<size_t>> AddOp::output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config) {
  if (input_shapes.size() < 2) {
    throw std::runtime_error("AddOp: expected at least 2 inputs");
  }
  for (size_t i = 1; i < input_shapes.size(); ++i) {
    if (input_shapes[0] != input_shapes[i]) {
      throw std::runtime_error("AddOp: all inputs must have the same shape");
    }
  }
  return {input_shapes[0]};
}

Tensor AddOp::forward(OpContext &ctx, const Vec<Tensor> &inputs) {
  if (inputs.size() < 2) {
    throw std::runtime_error("AddOp: expected at least 2 inputs");
  }
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (inputs[0].shape() != inputs[i].shape()) {
      throw std::runtime_error("AddOp: all inputs must have the same shape");
    }
  }

  if (ctx.is_training) {
    Tensor num_inputs_tensor({inputs.size()}, dptr(nullptr), DType_t::SIZE_T);
    ctx.residuals["num_inputs"] = num_inputs_tensor;
  }

  Tensor output = ctx.make_tensor(inputs[0].shape(), inputs[0].dtype());
  add(inputs[0], inputs[1], output);
  for (size_t i = 2; i < inputs.size(); ++i) {
    add(output, inputs[i], output);
  }
  return output;
}

Vec<Tensor> AddOp::backward(OpContext &ctx, const Tensor &grad_out) {
  const Tensor &num_inputs_tensor = ctx.residuals["num_inputs"];
  size_t num_inputs = num_inputs_tensor.shape()[0];

  Vec<Tensor> grad_inputs;
  grad_inputs.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    Tensor grad = ctx.make_tensor(grad_out.shape(), ctx.io_dtype);
    copy(grad_out, grad, ctx.handle.get_stream());
    grad_inputs.push_back(grad);
  }

  return grad_inputs;
}

LayerConfig AddOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.type = TYPE_NAME;
  lcfg.name = name;
  return lcfg;
}

}  // namespace tunx
