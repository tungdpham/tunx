/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/flatten.hpp"

#include <stdexcept>

#include "nn/engines/engine_handle.hpp"
#include "tensor/ops.hpp"

namespace tunx {

Vec<Vec<size_t>> FlattenOp::output_shapes(const Vec<Vec<size_t>> &input_shapes,
                                          const Config &config) {
  if (input_shapes.empty() || input_shapes[0].empty()) {
    throw std::invalid_argument("FlattenOp expects non-empty input shape");
  }
  const auto &input_shape = input_shapes[0];
  Vec<size_t> output_shape;
  output_shape.push_back(input_shape[0]);
  int start = std::max(1, config.start_dim);
  int end = (config.end_dim < 0)
                ? static_cast<int>(input_shape.size())
                : std::min(config.end_dim + 1, static_cast<int>(input_shape.size()));
  for (int i = 1; i < start && i < static_cast<int>(input_shape.size()); ++i) {
    output_shape.push_back(input_shape[i]);
  }
  size_t flat_dim = 1;
  for (int i = start; i < end; ++i) {
    flat_dim *= input_shape[i];
  }
  output_shape.push_back(flat_dim);
  for (int i = end; i < static_cast<int>(input_shape.size()); ++i) {
    output_shape.push_back(input_shape[i]);
  }
  return {output_shape};
}

Tensor FlattenOp::forward(OpContext &ctx, const Tensor &input, const Config &config) {
  if (ctx.is_training) {
    Tensor shape_tensor({input.shape().size()}, DType_t::SIZE_T);
    std::copy(input.shape().begin(), input.shape().end(), shape_tensor.data_as<size_t>());
    ctx.residuals["original_shape"] = shape_tensor;
  }
  Vec<size_t> output_shape = output_shapes({input.shape()}, config)[0];
  Tensor output = ctx.make_tensor(output_shape, ctx.io_dtype);
  copy(input, output, ctx.handle.get_stream());
  return output;
}

Tensor FlattenOp::backward(OpContext &ctx, const Tensor &grad_output, const Config &config) {
  const Tensor &shape_tensor = ctx.residuals["original_shape"];
  if (!shape_tensor) {
    throw std::runtime_error("No cached original shape found for backward pass in FlattenOp");
  }
  Vec<size_t> original_shape(shape_tensor.size());
  std::copy(shape_tensor.data_as<size_t>(), shape_tensor.data_as<size_t>() + shape_tensor.size(),
            original_shape.begin());
  Tensor grad_input = ctx.make_tensor(original_shape, ctx.io_dtype);
  copy(grad_output, grad_input, ctx.handle.get_stream());
  return grad_input;
}

LayerConfig FlattenOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.name = name;
  lcfg.type = TYPE_NAME;
  lcfg.set("start_dim", config.start_dim);
  lcfg.set("end_dim", config.end_dim);
  return lcfg;
}

FlattenOp::Config FlattenOp::parse_config(const LayerConfig &config) {
  Config c;
  c.start_dim = config.get<int>("start_dim", 1);
  c.end_dim = config.get<int>("end_dim", -1);
  return c;
}

}  // namespace tunx
