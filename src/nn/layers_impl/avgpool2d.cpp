/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/avgpool2d.hpp"

#include <stdexcept>

#include "nn/engines/iengine.hpp"
#include "tensor/ops.hpp"
#include "type/type.hpp"

namespace tunx {

Vec<Vec<size_t>> AvgPool2DOp::output_shapes(const Vec<Vec<size_t>> &input_shapes,
                                            const Config &config) {
  if (input_shapes.size() != 1 || input_shapes[0].size() != 4) {
    throw std::runtime_error("AvgPool2DOp: input shape must be 4D (NHWC format)");
  }
  const auto &shape = input_shapes[0];
  size_t batch_size = shape[0];
  size_t padded_h = shape[1] + 2 * config.pad_h;
  size_t padded_w = shape[2] + 2 * config.pad_w;
  size_t channels = shape[3];

  size_t output_h = (padded_h - config.pool_h) / config.stride_h + 1;
  size_t output_w = (padded_w - config.pool_w) / config.stride_w + 1;

  return {{batch_size, output_h, output_w, channels}};
}

Tensor AvgPool2DOp::forward(OpContext &ctx, const Tensor &input, const Config &config) {
  if (input.dims() != 4) {
    throw std::runtime_error("AvgPool2DOp: input must be 4D (NHWC format)");
  }

  const auto &shape = input.shape();
  size_t batch_size = shape[0];
  size_t input_h = shape[1];
  size_t input_w = shape[2];
  size_t channels = shape[3];

  size_t output_h = (input_h + 2 * config.pad_h - config.pool_h) / config.stride_h + 1;
  size_t output_w = (input_w + 2 * config.pad_w - config.pool_w) / config.stride_w + 1;

  AvgPool2DStats stats{.batch_size = batch_size,
                       .height = input_h,
                       .width = input_w,
                       .channels = channels,
                       .pool_h = config.pool_h,
                       .pool_w = config.pool_w,
                       .stride_h = config.stride_h,
                       .stride_w = config.stride_w,
                       .pad_h = config.pad_h,
                       .pad_w = config.pad_w};

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  WorkspaceReq ws_req = ctx.engine->query_avgpool_graph(ctx.handle, stats, type_desc);

  Tensor output = ctx.make_tensor({batch_size, output_h, output_w, channels}, input.dtype());
  size_t ws_size = ctx.is_training ? ws_req.fwd_workspace : ws_req.inf_workspace;
  Tensor ws = ctx.make_tensor({ws_size}, DType_t::BYTE);

  ctx.engine->avgpool_fwd(ctx.handle, stats, input.data_as<void>(), output.data_as<void>(),
                          ws.data_as<void>(), type_desc);

  return output;
}

Tensor AvgPool2DOp::backward(OpContext &ctx, const Tensor &grad_output, const Config &config) {
  if (grad_output.dims() != 4) {
    throw std::runtime_error("AvgPool2DOp: grad_output must be 4D (NHWC format)");
  }

  const auto &grad_shape = grad_output.shape();
  size_t batch_size = grad_shape[0];
  size_t output_h = grad_shape[1];
  size_t output_w = grad_shape[2];
  size_t channels = grad_shape[3];
  size_t input_h = (output_h - 1) * config.stride_h + config.pool_h - 2 * config.pad_h;
  size_t input_w = (output_w - 1) * config.stride_w + config.pool_w - 2 * config.pad_w;

  AvgPool2DStats stats{.batch_size = batch_size,
                       .height = input_h,
                       .width = input_w,
                       .channels = channels,
                       .pool_h = config.pool_h,
                       .pool_w = config.pool_w,
                       .stride_h = config.stride_h,
                       .stride_w = config.stride_w,
                       .pad_h = config.pad_h,
                       .pad_w = config.pad_w};

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  Tensor grad_input =
      ctx.make_tensor({batch_size, input_h, input_w, channels}, grad_output.dtype());
  fill(grad_input, 0.0f);

  WorkspaceReq ws_req = ctx.engine->query_avgpool_graph(ctx.handle, stats, type_desc);
  Tensor ws = ctx.make_tensor({ws_req.bwd_workspace}, DType_t::BYTE);

  ctx.engine->avgpool_bwd(ctx.handle, stats, grad_output.data_as<void>(),
                          grad_input.data_as<void>(), ws.data_as<void>(), type_desc);

  return grad_input;
}

LayerConfig AvgPool2DOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.name = name;
  lcfg.type = TYPE_NAME;
  lcfg.set("pool_h", config.pool_h);
  lcfg.set("pool_w", config.pool_w);
  lcfg.set("stride_h", config.stride_h);
  lcfg.set("stride_w", config.stride_w);
  lcfg.set("pad_h", config.pad_h);
  lcfg.set("pad_w", config.pad_w);
  return lcfg;
}

AvgPool2DOp::Config AvgPool2DOp::parse_config(const LayerConfig &config) {
  Config c;
  c.pool_h = config.get<size_t>("pool_h");
  c.pool_w = config.get<size_t>("pool_w");
  c.stride_h = config.get<size_t>("stride_h");
  c.stride_w = config.get<size_t>("stride_w");
  c.pad_h = config.get<size_t>("pad_h");
  c.pad_w = config.get<size_t>("pad_w");
  return c;
}

}  // namespace tunx
