/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/layer_norm.hpp"

#include <stdexcept>

#include "nn/engines/iengine.hpp"
#include "type/type.hpp"

namespace tunx {

Vec<Vec<size_t>> LayerNormOp::output_shapes(const Vec<Vec<size_t>> &input_shapes,
                                            const Config &config) {
  if (input_shapes.size() != 1) {
    throw std::runtime_error("LayerNormOp: expected exactly 1 input");
  }
  return {input_shapes[0]};
}

Tensor LayerNormOp::forward(OpContext &ctx, const Tensor &input, const Param &gamma,
                            const Param &beta, const Config &config) {
  if (ctx.is_training) {
    ctx.residuals["input"] = input;
  }

  const auto &shape = input.shape();
  size_t last_dim = shape.back();
  size_t channels = last_dim;
  size_t batch_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    batch_size *= shape[i];
  }

  LayerNormStats stats{
      .batch_size = batch_size,
      .seq_len = 1,
      .channels = channels,
      .epsilon = config.epsilon,
  };

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  WorkspaceReq ws_req = ctx.engine->query_layernorm_graph(ctx.handle, stats, type_desc);

  if (ctx.is_training) {
    Tensor batch_mean = ctx.make_tensor({batch_size}, DType_t::FP32);
    Tensor batch_invar = ctx.make_tensor({batch_size}, DType_t::FP32);
    ctx.residuals["batch_mean"] = batch_mean;
    ctx.residuals["batch_invar"] = batch_invar;

    Tensor output = ctx.make_tensor(shape, ctx.io_dtype);

    Tensor ws = ctx.make_tensor({ws_req.fwd_workspace}, DType_t::BYTE);

    ctx.engine->layernorm_fwd(ctx.handle, stats, input.data_as<void>(), gamma.data_as<void>(),
                              beta.data_as<void>(), output.data_as<void>(),
                              batch_mean.data_as<void>(), batch_invar.data_as<void>(),
                              ws.data_as<void>(), type_desc);

    return output;
  } else {
    Tensor output = ctx.make_tensor(shape, ctx.io_dtype);

    Tensor ws = ctx.make_tensor({ws_req.inf_workspace}, DType_t::BYTE);

    ctx.engine->layernorm_infer(ctx.handle, stats, input.data_as<void>(), gamma.data_as<void>(),
                                beta.data_as<void>(), output.data_as<void>(), ws.data_as<void>(),
                                type_desc);
    return output;
  }
}

Tensor LayerNormOp::backward(OpContext &ctx, const Tensor &grad_output, Param &gamma, Param &beta,
                             const Config &config) {
  const Tensor &input = ctx.residuals["input"];
  if (!input) {
    throw std::runtime_error("LayerNorm backward called without forward for this micro-batch");
  }

  const auto &shape = input.shape();
  Tensor grad_input = ctx.make_tensor(shape, ctx.io_dtype);

  size_t last_dim = shape.back();
  size_t channels = last_dim;
  size_t batch_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    batch_size *= shape[i];
  }

  LayerNormStats stats{
      .batch_size = batch_size,
      .seq_len = 1,
      .channels = channels,
      .epsilon = config.epsilon,
  };

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  WorkspaceReq ws_req = ctx.engine->query_layernorm_graph(ctx.handle, stats, type_desc);
  Tensor ws = ctx.make_tensor({ws_req.bwd_workspace}, DType_t::BYTE);

  const Tensor &batch_mean = ctx.residuals["batch_mean"];
  const Tensor &batch_invar = ctx.residuals["batch_invar"];

  if (config.affine) {
    ctx.engine->layernorm_bwd(ctx.handle, stats, grad_output.data_as<void>(), input.data_as<void>(),
                              gamma.data_as<void>(), batch_mean.data_as<void>(),
                              batch_invar.data_as<void>(), grad_input.data_as<void>(),
                              gamma.grad_as<void>(), beta.grad_as<void>(), ws.data_as<void>(),
                              type_desc);
  } else {
    ctx.engine->layernorm_bwd(ctx.handle, stats, grad_output.data_as<void>(), input.data_as<void>(),
                              gamma.data_as<void>(), batch_mean.data_as<void>(),
                              batch_invar.data_as<void>(), grad_input.data_as<void>(), nullptr,
                              nullptr, ws.data_as<void>(), type_desc);
  }

  return grad_input;
}

LayerConfig LayerNormOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.name = name;
  lcfg.type = TYPE_NAME;
  lcfg.set("normalized_shape", config.normalized_shape);
  lcfg.set("epsilon", config.epsilon);
  lcfg.set("affine", config.affine);
  return lcfg;
}

LayerNormOp::Config LayerNormOp::parse_config(const LayerConfig &config) {
  Config c;
  c.normalized_shape = config.get<size_t>("normalized_shape");
  c.epsilon = config.get<float>("epsilon", 1e-5f);
  c.affine = config.get<bool>("affine", true);
  return c;
}

}  // namespace tunx
