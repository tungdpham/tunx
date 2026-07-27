/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/conv2d.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include "nn/engines/iengine.hpp"
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace tunx {

Tensor Conv2DOp::forward(OpContext &ctx, const Tensor &input, const Param &weight,
                         const Param &bias, const Config &config) {
  if (input.dims() != 4) {
    throw std::invalid_argument("Conv2D: Input tensor must be 4-dimensional (NHWC)");
  }

  size_t channels = input.dim(3);

  if (channels != config.in_channels) {
    std::cerr << "Input shape: " << channels << " channels, expected: " << config.in_channels
              << " channels" << std::endl;
    throw std::invalid_argument("Input channel size mismatch in Conv2DOp");
  }

  if (ctx.is_training) {
    ctx.residuals["input"] = input;
  }

  size_t batch_size = input.dim(0);
  size_t input_h = input.dim(1);
  size_t input_w = input.dim(2);

  size_t output_h = (input_h + 2 * config.pad_h - config.kernel_h) / config.stride_h + 1;
  size_t output_w = (input_w + 2 * config.pad_w - config.kernel_w) / config.stride_w + 1;

  Tensor output =
      ctx.make_tensor({batch_size, output_h, output_w, config.out_channels}, input.dtype());

  Conv2DStats stats{
      .batch_size = batch_size,
      .in_channels = config.in_channels,
      .out_channels = config.out_channels,
      .input_h = input_h,
      .input_w = input_w,
      .kernel_h = config.kernel_h,
      .kernel_w = config.kernel_w,
      .stride_h = config.stride_h,
      .stride_w = config.stride_w,
      .pad_h = config.pad_h,
      .pad_w = config.pad_w,
      .use_bias = config.use_bias,
  };

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  WorkspaceReq ws_req = ctx.engine->query_conv2d_graph(ctx.handle, stats, type_desc);
  Tensor ws = ctx.make_tensor({ws_req.fwd_workspace}, DType_t::BYTE);

  ctx.engine->conv2d_fwd(ctx.handle, stats, input.data_as<void>(), weight.data_as<void>(),
                         config.use_bias ? bias.data_as<void>() : nullptr, output.data_as<void>(),
                         ws.data_as<void>(), type_desc);

  return output;
}

Tensor Conv2DOp::backward(OpContext &ctx, const Tensor &grad_output, Param &weight, Param &bias,
                          const Config &config) {
  if (grad_output.dims() != 4) {
    throw std::invalid_argument("Conv2D: Input tensor must be 4-dimensional (NHWC)");
  }

  size_t channels = grad_output.dim(3);

  if (channels != config.out_channels) {
    std::cerr << "Gradient shape: " << channels << " channels, expected: " << config.out_channels
              << " channels" << std::endl;
    throw std::invalid_argument("Gradient channel size mismatch in Conv2DOp");
  }

  const Tensor &input = ctx.residuals["input"];
  const auto &input_shape = input.shape();
  Tensor grad_input = ctx.make_tensor(input_shape, ctx.io_dtype);

  size_t batch_size = input_shape[0];
  size_t input_h = input_shape[1];
  size_t input_w = input_shape[2];

  Conv2DStats stats{
      .batch_size = batch_size,
      .in_channels = config.in_channels,
      .out_channels = config.out_channels,
      .input_h = input_h,
      .input_w = input_w,
      .kernel_h = config.kernel_h,
      .kernel_w = config.kernel_w,
      .stride_h = config.stride_h,
      .stride_w = config.stride_w,
      .pad_h = config.pad_h,
      .pad_w = config.pad_w,
      .use_bias = config.use_bias,
  };

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  WorkspaceReq ws_req = ctx.engine->query_conv2d_graph(ctx.handle, stats, type_desc);
  Tensor ws = ctx.make_tensor({ws_req.bwd_workspace}, DType_t::BYTE);

  ctx.engine->conv2d_wgrad(ctx.handle, stats, grad_output.data_as<void>(), input.data_as<void>(),
                           weight.grad_as<void>(), ws.data_as<void>(), type_desc);

  if (config.use_bias) {
    ctx.engine->conv2d_bgrad(ctx.handle, stats, grad_output.data_as<void>(), bias.grad_as<void>(),
                             ws.data_as<void>(), type_desc);
  }

  ctx.engine->conv2d_dgrad(ctx.handle, stats, grad_output.data_as<void>(), weight.data_as<void>(),
                           grad_input.data_as<void>(), ws.data_as<void>(), type_desc);

  return grad_input;
}

LayerConfig Conv2DOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.name = name;
  lcfg.type = TYPE_NAME;
  lcfg.set("in_channels", config.in_channels);
  lcfg.set("out_channels", config.out_channels);
  lcfg.set("kernel_h", config.kernel_h);
  lcfg.set("kernel_w", config.kernel_w);
  lcfg.set("stride_h", config.stride_h);
  lcfg.set("stride_w", config.stride_w);
  lcfg.set("pad_h", config.pad_h);
  lcfg.set("pad_w", config.pad_w);
  lcfg.set("use_bias", config.use_bias);
  return lcfg;
}

Conv2DOp::Config Conv2DOp::parse_config(const LayerConfig &config) {
  Config c;
  c.in_channels = config.get<size_t>("in_channels");
  c.out_channels = config.get<size_t>("out_channels");
  c.kernel_h = config.get<size_t>("kernel_h");
  c.kernel_w = config.get<size_t>("kernel_w");
  c.stride_h = config.get<size_t>("stride_h", 1);
  c.stride_w = config.get<size_t>("stride_w", 1);
  c.pad_h = config.get<size_t>("pad_h", 0);
  c.pad_w = config.get<size_t>("pad_w", 0);
  c.use_bias = config.get<bool>("use_bias", true);
  return c;
}

Vec<Vec<size_t>> Conv2DOp::output_shapes(const Vec<Vec<size_t>> &input_shapes,
                                         const Config &config) {
  if (input_shapes.empty() || input_shapes[0].size() != 4) {
    throw std::invalid_argument("Conv2DOp expects 4D input including batch size");
  }

  size_t batch_size = input_shapes[0][0];
  size_t output_h = (input_shapes[0][1] + 2 * config.pad_h - config.kernel_h) / config.stride_h + 1;
  size_t output_w = (input_shapes[0][2] + 2 * config.pad_w - config.kernel_w) / config.stride_w + 1;

  return {{batch_size, output_h, output_w, config.out_channels}};
}

Conv2D::Conv2D(size_t in_channels, size_t out_channels, size_t kernel_h, size_t kernel_w,
               size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, bool use_bias,
               const std::string &name)
    : FunctionalLayer(Conv2DOp::Config{in_channels, out_channels, kernel_h, kernel_w, stride_h,
                                       stride_w, pad_h, pad_w, use_bias},
                      name) {
  impl_->register_param(
      "weight", {out_channels, kernel_h, kernel_w, in_channels},
      [in_channels, kernel_h, kernel_w](Param &p, OpContext &ctx) {
        float stddev = static_cast<float>(
            1.0 / std::sqrt(static_cast<double>(in_channels * kernel_h * kernel_w)));
        long long seed = ctx.use_seed ? ctx.srand_seed
                                      : std::chrono::system_clock::now().time_since_epoch().count();
        fill_normal(p.data(), 0, stddev, seed);
      });

  if (use_bias) {
    impl_->register_param(
        "bias", {out_channels}, [in_channels, kernel_h, kernel_w](Param &p, OpContext &ctx) {
          float stddev = static_cast<float>(
              1.0 / std::sqrt(static_cast<double>(in_channels * kernel_h * kernel_w)));
          long long seed = ctx.use_seed
                               ? ctx.srand_seed
                               : std::chrono::system_clock::now().time_since_epoch().count();
          fill_normal(p.data(), 0, stddev, seed);
        });
  } else {
    impl_->register_param("bias_dummy", {0}, nullptr);
  }
}

}  // namespace tunx
