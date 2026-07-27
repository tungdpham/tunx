/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/dropout.hpp"

#include <stdexcept>

#include "nn/engines/iengine.hpp"
#include "tensor/ops.hpp"
#include "type/type.hpp"

namespace tunx {

Vec<Vec<size_t>> DropoutOp::output_shapes(const Vec<Vec<size_t>> &input_shapes,
                                          const Config &config) {
  return input_shapes;
}

Tensor DropoutOp::forward(OpContext &ctx, const Tensor &input, const Config &config) {
  if (config.dropout_rate < 0.0f || config.dropout_rate >= 1.0f) {
    throw std::invalid_argument("Dropout rate must be in [0, 1)");
  }

  if (!ctx.is_training) {
    Tensor output = ctx.make_tensor(input.shape(), ctx.io_dtype);
    copy(input, output, ctx.handle.get_stream());
    return output;
  }

  size_t batch_size = input.dim(0);
  size_t channels = input.dims() > 1 ? input.dim(1) : 1;
  size_t spatial_size = input.dims() > 1 ? input.stride(1) : input.stride(0);

  DropoutStats stats{
      .batch_size = batch_size,
      .channels = channels,
      .spatial_size = spatial_size,
      .dropout_rate = config.dropout_rate,
  };

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  WorkspaceReq ws_req = ctx.engine->query_dropout_graph(ctx.handle, stats, type_desc);

  Tensor mask = ctx.make_tensor(input.shape(), DType_t::BOOL);
  ctx.residuals["mask"] = mask;

  Tensor output = ctx.make_tensor(input.shape(), ctx.io_dtype);
  Tensor ws = ctx.make_tensor({ws_req.fwd_workspace}, DType_t::BYTE);

  ctx.engine->dropout_fwd(ctx.handle, stats, input.data_as<void>(), output.data_as<void>(),
                          mask.data_as<bool>(), ws.data_as<void>(), type_desc);

  return output;
}

Tensor DropoutOp::backward(OpContext &ctx, const Tensor &grad_output, const Config &config) {
  Tensor &mask = ctx.residuals["mask"];
  if (!mask) {
    throw std::runtime_error("No cached mask found in DropoutOp backward pass");
  }

  size_t batch_size = grad_output.dim(0);
  size_t channels = grad_output.dims() > 1 ? grad_output.dim(1) : 1;
  size_t spatial_size = grad_output.dims() > 1 ? grad_output.stride(1) : grad_output.stride(0);

  DropoutStats stats{
      .batch_size = batch_size,
      .channels = channels,
      .spatial_size = spatial_size,
      .dropout_rate = config.dropout_rate,
  };

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  Tensor grad_input = ctx.make_tensor(grad_output.shape(), ctx.io_dtype);
  WorkspaceReq ws_req = ctx.engine->query_dropout_graph(ctx.handle, stats, type_desc);
  Tensor ws = ctx.make_tensor({ws_req.bwd_workspace}, DType_t::BYTE);

  double scale = 1.0 / (1.0 - config.dropout_rate);

  ctx.engine->dropout_bwd(ctx.handle, stats, grad_output.data_as<void>(),
                          grad_input.data_as<void>(), mask.data_as<bool>(), scale,
                          ws.data_as<void>(), type_desc);

  return grad_input;
}

LayerConfig DropoutOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.name = name;
  lcfg.type = TYPE_NAME;
  lcfg.set("dropout_rate", config.dropout_rate);
  return lcfg;
}

DropoutOp::Config DropoutOp::parse_config(const LayerConfig &config) {
  Config c;
  c.dropout_rate = config.get<float>("dropout_rate");
  return c;
}

}  // namespace tunx
