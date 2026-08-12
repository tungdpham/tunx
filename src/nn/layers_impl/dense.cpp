/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/dense.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>

#include "nn/engines/engine_handle.hpp"
#include "nn/stats/stats.hpp"
#include "tensor/ops.hpp"
#include "type/type.hpp"

namespace tunx {

void DenseOp::init(OpContext &ctx, const Config &config) {
  size_t input_features = config.input_features;
  size_t output_features = config.output_features;

  float stddev = static_cast<float>(std::sqrt(2.0 / static_cast<double>(input_features + output_features)));
  long long seed =
      ctx.use_seed ? ctx.srand_seed : std::chrono::system_clock::now().time_since_epoch().count();

  Param weights = ctx.make_param({input_features, output_features});
  fill_normal(weights.data(), 0, stddev, seed);

  if (config.use_bias) {
    Param bias = ctx.make_param({output_features});
    fill_normal(bias.data(), 0, stddev, seed);
  } else {
    ctx.make_param({0});
  }
}

Vec<Vec<size_t>> DenseOp::output_shapes(const Vec<Vec<size_t>> &input_shapes,
                                        const Config &config) {
  if (input_shapes.empty()) throw std::runtime_error("DenseOp: Input shape is empty.");
  Vec<size_t> out_shape = input_shapes[0];
  out_shape.back() = config.output_features;
  return {out_shape};
}

Tensor DenseOp::forward(OpContext &ctx, const Tensor &input, const Param &weights,
                        const Param &bias, const Config &config) {
  const Vec<size_t> &in_shape = input.shape();
  size_t last_dim = in_shape.back();

  if (last_dim != config.input_features) {
    throw std::invalid_argument("Input feature size mismatch in DenseOp");
  }

  if (ctx.is_training) {
    ctx.residuals["input"] = input;
  }

  size_t batch_size = 1;
  for (size_t i = 0; i < in_shape.size() - 1; ++i) {
    batch_size *= in_shape[i];
  }

  DenseStats stats{
      .batch_size = batch_size,
      .in_features = config.input_features,
      .out_features = config.output_features,
      .use_bias = config.use_bias,
  };

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  WorkspaceReq ws_req = ctx.engine->query_dense_graph(ctx.handle, stats, type_desc);

  Vec<size_t> out_shape = in_shape;
  out_shape.back() = config.output_features;
  Tensor output = ctx.make_tensor(out_shape, ctx.io_dtype);
  Tensor ws = ctx.make_tensor({ws_req.fwd_workspace}, DType_t::BYTE);

  ctx.engine->dense_fwd(ctx.handle, stats, input.data_as<void>(), weights.data_as<void>(),
                        config.use_bias ? bias.data_as<void>() : nullptr, output.data_as<void>(),
                        ws.data_as<void>(), type_desc);

  return output;
}

Tensor DenseOp::backward(OpContext &ctx, const Tensor &grad_output, Param &weights, Param &bias,
                         const Config &config) {
  if (grad_output.shape().back() != config.output_features) {
    throw std::invalid_argument("Gradient feature size mismatch in DenseOp.");
  }
  const Tensor &input = ctx.residuals["input"];
  const Vec<size_t> &in_shape = input.shape();
  size_t batch_size = 1;
  for (size_t i = 0; i < in_shape.size() - 1; ++i) {
    batch_size *= in_shape[i];
  }

  DenseStats stats{
      .batch_size = batch_size,
      .in_features = config.input_features,
      .out_features = config.output_features,
      .use_bias = config.use_bias,
  };

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  Tensor grad_input = ctx.make_tensor(in_shape, ctx.io_dtype);
  WorkspaceReq ws_req = ctx.engine->query_dense_graph(ctx.handle, stats, type_desc);
  Tensor ws = ctx.make_tensor({ws_req.bwd_workspace}, DType_t::BYTE);

  ctx.engine->dense_wgrad(ctx.handle, stats, grad_output.data_as<void>(), input.data_as<void>(),
                          weights.grad_as<void>(), ws.data_as<void>(), type_desc);

  if (config.use_bias) {
    ctx.engine->dense_bgrad(ctx.handle, stats, grad_output.data_as<void>(), bias.grad_as<void>(),
                            ws.data_as<void>(), type_desc);
  }

  ctx.engine->dense_dgrad(ctx.handle, stats, grad_output.data_as<void>(), weights.data_as<void>(),
                          grad_input.data_as<void>(), ws.data_as<void>(), type_desc);

  return grad_input;
}

LayerConfig DenseOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.name = name;
  lcfg.type = TYPE_NAME;
  lcfg.set("input_features", config.input_features);
  lcfg.set("output_features", config.output_features);
  lcfg.set("use_bias", config.use_bias);
  return lcfg;
}

DenseOp::Config DenseOp::parse_config(const LayerConfig &config) {
  Config c;
  c.input_features = config.get<size_t>("input_features");
  c.output_features = config.get<size_t>("output_features");
  c.use_bias = config.get<bool>("use_bias", true);
  return c;
}

}  // namespace tunx
