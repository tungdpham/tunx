/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/concat.hpp"

#include <stdexcept>

namespace tunx {

Vec<Vec<size_t>> ConcatOp::output_shapes(const Vec<Vec<size_t>> &input_shapes,
                                         const Config &config) {
  if (input_shapes.empty()) throw std::invalid_argument("ConcatOp expects at least one input");

  auto out_shape = input_shapes[0];
  int rank = static_cast<int>(out_shape.size());
  int axis = config.axis < 0 ? config.axis + rank : config.axis;
  if (axis < 0 || axis >= rank) {
    throw std::invalid_argument("Concat axis out of bounds");
  }
  size_t final_axis = static_cast<size_t>(axis);

  size_t total_dim = 0;
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    if (input_shapes[i].size() != out_shape.size()) {
      throw std::invalid_argument("ConcatOp: All inputs must have the same number of dimensions");
    }
    for (size_t j = 0; j < out_shape.size(); ++j) {
      if (j != final_axis && input_shapes[i][j] != out_shape[j]) {
        throw std::invalid_argument(
            "ConcatOp: All inputs must have the same shape except for the concat axis");
      }
    }
    total_dim += input_shapes[i][final_axis];
  }

  out_shape[final_axis] = total_dim;
  return {out_shape};
}

Tensor ConcatOp::forward(OpContext &ctx, const Vec<Tensor> &inputs, const Config &config) {
  if (inputs.empty()) throw std::invalid_argument("ConcatOp expects at least one input");

  Vec<Vec<size_t>> input_shapes;
  for (const auto &input : inputs) {
    input_shapes.push_back(input.shape());
  }

  int rank = static_cast<int>(inputs.empty() ? 0 : inputs[0].shape().size());
  size_t final_axis = config.axis < 0 ? config.axis + rank : config.axis;

  if (ctx.is_training) {
    // We need to save the sizes of the concatenated dimension for the backward pass
    Vec<size_t> sizes;
    for (size_t i = 0; i < inputs.size(); ++i) {
      sizes.push_back(inputs[i].shape()[final_axis]);
    }
    Tensor shape_tensor(sizes, dptr(nullptr), DType_t::SIZE_T);
    ctx.residuals["input_axis_sizes"] = shape_tensor;
  }

  Vec<size_t> out_shape = output_shapes(input_shapes, config)[0];
  Tensor output = ctx.make_tensor(out_shape, ctx.io_dtype);

  // Initialize output with zeroes just in case
  tunx::fill(output, 0.0f, ctx.handle.get_stream());

  size_t outer_size = 1;
  for (size_t i = 0; i < final_axis; ++i) outer_size *= out_shape[i];
  size_t inner_size = 1;
  for (size_t i = final_axis + 1; i < out_shape.size(); ++i) inner_size *= out_shape[i];

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  size_t current_start = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    size_t length = inputs[i].shape()[final_axis];

    SliceStats stats{
        .outer_size = outer_size,
        .inner_size = inner_size,
        .axis_size = out_shape[final_axis],
        .start = current_start,
        .length = length,
    };

    WorkspaceReq req = ctx.engine->query_slice_graph(ctx.handle, stats, type_desc);
    size_t ws_size = req.bwd_workspace > 0 ? req.bwd_workspace : 1;
    Tensor workspace = ctx.make_tensor({ws_size}, DType_t::BYTE);

    // Using slice_bwd to insert the input into the concatenated output
    ctx.engine->slice_bwd(ctx.handle, stats, inputs[i].data_as<void>(), output.data_as<void>(),
                          workspace.data_as<void>(), type_desc);

    current_start += length;
  }

  return output;
}

Vec<Tensor> ConcatOp::backward(OpContext &ctx, const Tensor &grad_output, const Config &config) {
  const Tensor &shape_tensor = ctx.residuals["input_axis_sizes"];
  const auto &sizes = shape_tensor.shape();
  size_t num_inputs = sizes.size();

  int rank = static_cast<int>(grad_output.shape().size());
  size_t final_axis = config.axis < 0 ? config.axis + rank : config.axis;

  Vec<Tensor> grad_inputs;
  grad_inputs.reserve(num_inputs);

  size_t outer_size = 1;
  for (size_t i = 0; i < final_axis; ++i) outer_size *= grad_output.shape()[i];
  size_t inner_size = 1;
  for (size_t i = final_axis + 1; i < grad_output.shape().size(); ++i)
    inner_size *= grad_output.shape()[i];

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  size_t current_start = 0;
  for (size_t i = 0; i < num_inputs; ++i) {
    size_t length = sizes[i];

    Vec<size_t> input_shape = grad_output.shape();
    input_shape[final_axis] = length;

    Tensor grad_input = ctx.make_tensor(input_shape, ctx.io_dtype);

    SliceStats stats{
        .outer_size = outer_size,
        .inner_size = inner_size,
        .axis_size = grad_output.shape()[final_axis],
        .start = current_start,
        .length = length,
    };

    WorkspaceReq req = ctx.engine->query_slice_graph(ctx.handle, stats, type_desc);
    size_t ws_size = req.fwd_workspace > 0 ? req.fwd_workspace : 1;
    Tensor workspace = ctx.make_tensor({ws_size}, DType_t::BYTE);

    ctx.engine->slice_fwd(ctx.handle, stats, grad_output.data_as<void>(),
                          grad_input.data_as<void>(), workspace.data_as<void>(), type_desc);

    grad_inputs.push_back(grad_input);
    current_start += length;
  }

  return grad_inputs;
}

LayerConfig ConcatOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.name = name;
  lcfg.type = TYPE_NAME;
  lcfg.set("axis", config.axis);
  return lcfg;
}

ConcatOp::Config ConcatOp::parse_config(const LayerConfig &config) {
  Config c;
  c.axis = config.get<int>("axis", -1);
  return c;
}

}  // namespace tunx
