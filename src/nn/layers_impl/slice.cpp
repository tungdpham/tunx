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
  int rank = static_cast<int>(out_shape.size());
  int axis = config.axis < 0 ? config.axis + rank : config.axis;
  if (axis < 0 || axis >= rank) {
    throw std::invalid_argument("Slice axis out of bounds");
  }
  size_t final_axis = static_cast<size_t>(axis);

  if (config.start + config.length > out_shape[final_axis]) {
    throw std::invalid_argument("Slice range out of bounds");
  }
  out_shape[final_axis] = config.length;
  return {out_shape};
}

Tensor SliceOp::forward(OpContext &ctx, const Tensor &input, const Config &config) {
  if (ctx.is_training) {
    Tensor shape_tensor(input.shape(), dptr(nullptr), DType_t::SIZE_T);
    ctx.residuals["original_shape"] = shape_tensor;
  }

  Vec<size_t> out_shape = output_shapes({input.shape()}, config)[0];
  Tensor output = ctx.make_tensor(out_shape, ctx.io_dtype);

  int rank = static_cast<int>(input.shape().size());
  size_t final_axis = config.axis < 0 ? config.axis + rank : config.axis;

  size_t outer_size = 1;
  for (size_t i = 0; i < final_axis; ++i) outer_size *= input.shape()[i];
  size_t inner_size = 1;
  for (size_t i = final_axis + 1; i < input.shape().size(); ++i) inner_size *= input.shape()[i];

  SliceStats stats{
      .outer_size = outer_size,
      .inner_size = inner_size,
      .axis_size = input.shape()[final_axis],
      .start = config.start,
      .length = config.length,
  };

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  WorkspaceReq req = ctx.engine->query_slice_graph(ctx.handle, stats, type_desc);
  size_t ws_size = req.fwd_workspace > 0 ? req.fwd_workspace : 1;
  Tensor workspace = ctx.make_tensor({ws_size}, DType_t::BYTE);

  ctx.engine->slice_fwd(ctx.handle, stats, input.data_as<void>(), output.data_as<void>(),
                        workspace.data_as<void>(), type_desc);

  return output;
}

Tensor SliceOp::backward(OpContext &ctx, const Tensor &grad_output, const Config &config) {
  const Tensor &shape_tensor = ctx.residuals["original_shape"];
  Vec<size_t> original_shape = shape_tensor.shape();

  Tensor grad_input = ctx.make_tensor(original_shape, ctx.io_dtype);
  tunx::fill(grad_input, 0.0f, ctx.handle.get_stream());

  int rank = static_cast<int>(original_shape.size());
  size_t final_axis = config.axis < 0 ? config.axis + rank : config.axis;

  size_t outer_size = 1;
  for (size_t i = 0; i < final_axis; ++i) outer_size *= original_shape[i];
  size_t inner_size = 1;
  for (size_t i = final_axis + 1; i < original_shape.size(); ++i) inner_size *= original_shape[i];

  SliceStats stats{
      .outer_size = outer_size,
      .inner_size = inner_size,
      .axis_size = original_shape[final_axis],
      .start = config.start,
      .length = config.length,
  };

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  WorkspaceReq req = ctx.engine->query_slice_graph(ctx.handle, stats, type_desc);
  size_t ws_size = req.bwd_workspace > 0 ? req.bwd_workspace : 1;
  Tensor workspace = ctx.make_tensor({ws_size}, DType_t::BYTE);

  ctx.engine->slice_bwd(ctx.handle, stats, grad_output.data_as<void>(), grad_input.data_as<void>(),
                        workspace.data_as<void>(), type_desc);

  return grad_input;
}

LayerConfig SliceOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.name = name;
  lcfg.type = TYPE_NAME;
  lcfg.set("axis", config.axis);
  lcfg.set("start", (int)config.start);
  lcfg.set("length", (int)config.length);
  return lcfg;
}

SliceOp::Config SliceOp::parse_config(const LayerConfig &config) {
  Config c;
  c.axis = config.get<int>("axis", 0);
  c.start = (size_t)config.get<int>("start", 0);
  c.length = (size_t)config.get<int>("length", 1);
  return c;
}

}  // namespace tunx
