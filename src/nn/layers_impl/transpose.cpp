/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/transpose.hpp"
#include <stdexcept>
#include "nn/stats/stats.hpp"
#include "nn/engines/iengine.hpp"

namespace tunx {

Vec<Vec<size_t>> TransposeOp::output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config) {
  if (input_shapes.empty()) throw std::invalid_argument("TransposeOp expects input shapes");
  auto out_shape = input_shapes[0];
  if (config.dim0 >= out_shape.size() || config.dim1 >= out_shape.size()) {
    throw std::runtime_error("TransposeOp: dim0 or dim1 out of bounds");
  }
  std::swap(out_shape[config.dim0], out_shape[config.dim1]);
  return {out_shape};
}

Tensor TransposeOp::forward(OpContext &ctx, const Tensor &input, const Config &config) {
  if (config.dim0 >= input.dims() || config.dim1 >= input.dims()) {
    throw std::runtime_error("TransposeOp: dim0 or dim1 out of bounds");
  }

  Vec<size_t> out_shape = input.shape();
  std::swap(out_shape[config.dim0], out_shape[config.dim1]);
  Tensor output = ctx.make_tensor(out_shape, input.dtype());

  TransposeStats stats;
  stats.ndim = input.dims();
  stats.dim0 = config.dim0;
  stats.dim1 = config.dim1;
  for (size_t i = 0; i < input.dims(); ++i) stats.shape[i] = input.dim(i);

  DTypeDesc type_desc{ctx.io_dtype, ctx.param_dtype, ctx.compute_dtype};

  WorkspaceReq ws_req = ctx.engine->query_transpose_graph(ctx.handle, stats, type_desc);

  Tensor ws = ctx.make_tensor({ws_req.fwd_workspace}, ctx.io_dtype);

  ctx.engine->transpose(ctx.handle, stats, input.data_as<void>(), output.data_as<void>(),
                     ws.data_as<void>(), type_desc);

  return output;
}

Tensor TransposeOp::backward(OpContext &ctx, const Tensor &grad_output, const Config &config) {
  if (config.dim0 >= grad_output.dims() || config.dim1 >= grad_output.dims()) {
    throw std::runtime_error("TransposeOp: dim0 or dim1 out of bounds");
  }

  Vec<size_t> in_shape = grad_output.shape();
  std::swap(in_shape[config.dim0], in_shape[config.dim1]);
  Tensor grad_input = ctx.make_tensor(in_shape, grad_output.dtype());

  TransposeStats stats;
  stats.ndim = grad_output.dims();
  stats.dim0 = config.dim0;
  stats.dim1 = config.dim1;
  for (size_t i = 0; i < grad_output.dims(); ++i) stats.shape[i] = grad_output.dim(i);

  DTypeDesc type_desc{ctx.io_dtype, ctx.param_dtype, ctx.compute_dtype};

  WorkspaceReq ws_req = ctx.engine->query_transpose_graph(ctx.handle, stats, type_desc);

  Tensor ws = ctx.make_tensor({ws_req.bwd_workspace}, ctx.io_dtype);

  ctx.engine->transpose(ctx.handle, stats, grad_output.data_as<void>(), grad_input.data_as<void>(),
                     ws.data_as<void>(), type_desc);

  return grad_input;
}

LayerConfig TransposeOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.name = name;
  lcfg.type = TYPE_NAME;
  lcfg.set("dim0", config.dim0);
  lcfg.set("dim1", config.dim1);
  return lcfg;
}

TransposeOp::Config TransposeOp::parse_config(const LayerConfig &config) {
  Config c;
  c.dim0 = config.get<size_t>("dim0");
  c.dim1 = config.get<size_t>("dim1");
  return c;
}

}  // namespace tunx
