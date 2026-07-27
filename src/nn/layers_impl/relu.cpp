/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/relu.hpp"

#include <stdexcept>

#include "nn/engines/iengine.hpp"
#include "type/type.hpp"

namespace tunx {

Vec<Vec<size_t>> ReLUOp::output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config) {
  if (input_shapes.size() != 1) {
    throw std::runtime_error("ReLUOp: expected exactly 1 input");
  }
  return {input_shapes[0]};
}

Tensor ReLUOp::forward(OpContext &ctx, const Tensor &input) {
  Tensor output = ctx.make_tensor(input.shape(), ctx.io_dtype);

  size_t batch_size = input.dims() > 0 ? input.dim(0) : 1;
  size_t spatial_size = input.dims() > 0 ? input.stride(0) : 1;

  ReLUStats stats{batch_size, spatial_size};
  DTypeDesc type_desc{ctx.io_dtype, ctx.param_dtype, ctx.compute_dtype};

  WorkspaceReq ws_req = ctx.engine->query_relu_graph(ctx.handle, stats, type_desc);
  Tensor ws = ctx.make_tensor({ws_req.fwd_workspace}, DType_t::BYTE);

  if (ctx.is_training) {
    ctx.residuals["output"] = output;
    ctx.engine->relu_fwd(ctx.handle, stats, input.data_as<void>(), output.data_as<void>(),
                         ws.data_as<void>(), type_desc);
  } else {
    ctx.engine->relu_inf(ctx.handle, stats, input.data_as<void>(), output.data_as<void>(),
                         ws.data_as<void>(), type_desc);
  }

  return output;
}

Tensor ReLUOp::backward(OpContext &ctx, const Tensor &grad_output) {
  const Tensor &output = ctx.residuals["output"];

  Tensor grad_input = ctx.make_tensor(grad_output.shape(), ctx.io_dtype);

  size_t batch_size = grad_output.dims() > 0 ? grad_output.dim(0) : 1;
  size_t spatial_size = grad_output.dims() > 0 ? grad_output.stride(0) : 1;

  ReLUStats stats{batch_size, spatial_size};
  DTypeDesc type_desc{ctx.io_dtype, ctx.param_dtype, ctx.compute_dtype};

  WorkspaceReq ws_req = ctx.engine->query_relu_graph(ctx.handle, stats, type_desc);
  Tensor ws = ctx.make_tensor({ws_req.bwd_workspace}, DType_t::BYTE);

  ctx.engine->relu_bwd(ctx.handle, stats, grad_output.data_as<void>(), grad_input.data_as<void>(),
                       output.data_as<void>(), ws.data_as<void>(), type_desc);

  return grad_input;
}

LayerConfig ReLUOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.type = TYPE_NAME;
  lcfg.name = name;
  return lcfg;
}

}  // namespace tunx
