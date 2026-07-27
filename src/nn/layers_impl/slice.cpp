/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/slice.hpp"

#include <stdexcept>

namespace tunx {

Vec<Vec<size_t>> SliceOp::output_shapes(const Vec<Vec<size_t>> &input_shapes,
                                        const Config &config) {
  if (input_shapes.empty()) throw std::invalid_argument("SliceOp expects input shapes");
  auto out_shape = input_shapes[0];
  if (config.axis >= out_shape.size()) {
    throw std::invalid_argument("Slice axis out of bounds");
  }
  if (config.start + config.length > out_shape[config.axis]) {
    throw std::invalid_argument("Slice range out of bounds");
  }
  out_shape[config.axis] = config.length;
  return {out_shape};
}

Tensor SliceOp::forward(OpContext &ctx, const Tensor &input, const Config &config) {
  if (ctx.is_training) {
    Tensor shape_tensor = ctx.make_tensor({input.shape().size()}, DType_t::SIZE_T);
    std::copy(input.shape().begin(), input.shape().end(), shape_tensor.data_as<size_t>());
    ctx.residuals["original_shape"] = shape_tensor;
  }

  Vec<size_t> out_shape = output_shapes({input.shape()}, config)[0];
  Tensor output = ctx.make_tensor(out_shape, ctx.io_dtype);

  throw std::runtime_error("SliceOp: unimplemented");
  return output;
}

Tensor SliceOp::backward(OpContext &ctx, const Tensor &grad_output, const Config &config) {
  const Tensor &shape_tensor = ctx.residuals["original_shape"];
  Vec<size_t> original_shape(shape_tensor.size());
  std::copy(shape_tensor.data_as<size_t>(), shape_tensor.data_as<size_t>() + shape_tensor.size(),
            original_shape.begin());

  Tensor grad_input = ctx.make_tensor(original_shape, ctx.io_dtype);

  throw std::runtime_error("SliceOp: unimplemented");
  return grad_input;
}

LayerConfig SliceOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.name = name;
  lcfg.type = TYPE_NAME;
  lcfg.set("axis", (int)config.axis);
  lcfg.set("start", (int)config.start);
  lcfg.set("length", (int)config.length);
  return lcfg;
}

SliceOp::Config SliceOp::parse_config(const LayerConfig &config) {
  Config c;
  c.axis = (size_t)config.get<int>("axis", 0);
  c.start = (size_t)config.get<int>("start", 0);
  c.length = (size_t)config.get<int>("length", 1);
  return c;
}

}  // namespace tunx
