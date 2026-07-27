/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/legacy_conv2d.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "nn/engines/iengine.hpp"
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace tunx {

Vec<Vec<size_t>> LegacyConv2DOp::output_shapes(const Vec<Vec<size_t>> &input_shapes,
                                               const Config &config) {
  if (input_shapes.size() != 1 || input_shapes[0].size() != 4) {
    throw std::invalid_argument("LegacyConv2DOp expects 4D input including batch size");
  }

  size_t batch_size = input_shapes[0][0];
  size_t output_h = (input_shapes[0][2] + 2 * config.pad_h - config.kernel_h) / config.stride_h + 1;
  size_t output_w = (input_shapes[0][3] + 2 * config.pad_w - config.kernel_w) / config.stride_w + 1;

  return {{batch_size, config.out_channels, output_h, output_w}};
}

Tensor LegacyConv2DOp::forward(OpContext &ctx, const Tensor &input, const Param &weights,
                               const Param &bias, const Config &config) {
  if (input.dims() != 4) {
    throw std::invalid_argument("Conv2D: Input tensor must be 4-dimensional (NCHW)");
  }

  size_t channels = input.dim(1);
  if (channels != config.in_channels) {
    std::cerr << "Input shape: " << channels << " channels, expected: " << config.in_channels
              << " channels" << std::endl;
    throw std::invalid_argument("Input channel size mismatch in LegacyConv2DOp");
  }

  size_t batch_size = input.dim(0);
  size_t input_h = input.dim(2);
  size_t input_w = input.dim(3);

  size_t output_h = (input_h + 2 * config.pad_h - config.kernel_h) / config.stride_h + 1;
  size_t output_w = (input_w + 2 * config.pad_w - config.kernel_w) / config.stride_w + 1;

  Tensor output =
      ctx.make_tensor({batch_size, config.out_channels, output_h, output_w}, input.dtype());

  size_t kernel_size = config.in_channels * config.kernel_h * config.kernel_w;
  size_t output_size = batch_size * output_h * output_w;
  size_t col_matrix_size = kernel_size * output_size;

  Tensor col_buffer = ctx.make_tensor({col_matrix_size}, ctx.io_dtype);
  if (ctx.is_training) {
    ctx.residuals["col_buffer"] = col_buffer;
  }

  size_t output_buffer_size = config.out_channels * output_size;
  Tensor temp_output_buffer = ctx.make_tensor({output_buffer_size}, ctx.io_dtype);

  im2col(input, col_buffer, config.kernel_h, config.kernel_w, config.stride_h, config.stride_w,
         config.pad_h, config.pad_w, ctx.handle.get_stream());

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  ctx.engine->legacy_conv2d_fwd(ctx.handle, col_buffer.data_as<void>(), weights.data_as<void>(),
                                temp_output_buffer.data_as<void>(), output_size, kernel_size,
                                config.out_channels, type_desc);

  cnhw_to_nchw(temp_output_buffer, output, batch_size, config.out_channels, output_h, output_w,
               ctx.handle.get_stream());

  if (config.use_bias) {
    ctx.engine->legacy_conv2d_add_bias(ctx.handle, output.data_as<void>(), bias.data_as<void>(),
                                       batch_size, output_h, output_w, config.out_channels,
                                       type_desc);
  }

  return output;
}

Tensor LegacyConv2DOp::backward(OpContext &ctx, const Tensor &grad_output, Param &weights,
                                Param &bias, const Config &config) {
  if (grad_output.dims() != 4) {
    throw std::invalid_argument("Conv2D: Input tensor must be 4-dimensional (NCHW)");
  }
  size_t channels = grad_output.dim(1);
  if (channels != config.out_channels) {
    std::cerr << "Input shape: " << channels << " channels, expected: " << config.out_channels
              << " channels" << std::endl;
    throw std::invalid_argument("Gradient channel size mismatch in LegacyConv2DOp");
  }

  const Vec<size_t> &output_shape = grad_output.shape();
  size_t batch_size = output_shape[0];
  size_t output_h = output_shape[2];
  size_t output_w = output_shape[3];
  size_t input_h = (output_h - 1) * config.stride_h + config.kernel_h - 2 * config.pad_h;
  size_t input_w = (output_w - 1) * config.stride_w + config.kernel_w - 2 * config.pad_w;

  Tensor grad_input =
      ctx.make_tensor({batch_size, config.in_channels, input_h, input_w}, ctx.io_dtype);
  fill(grad_input, 0.0f);  // col2im accumulates, so we need to zero first

  Tensor col_buffer = ctx.residuals["col_buffer"];

  size_t kernel_size = config.in_channels * config.kernel_h * config.kernel_w;
  size_t output_size = batch_size * output_h * output_w;
  size_t col_grad_matrix_size = kernel_size * output_size;

  size_t gradient_buffer_size = config.out_channels * output_size;
  Tensor temp_gradient_buffer = ctx.make_tensor({gradient_buffer_size}, ctx.io_dtype);
  Tensor temp_col_grad_matrix_buffer = ctx.make_tensor({col_grad_matrix_size}, ctx.io_dtype);

  nchw_to_cnhw(grad_output, temp_gradient_buffer, batch_size, config.out_channels, output_h,
               output_w, ctx.handle.get_stream());

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  ctx.engine->legacy_conv2d_wgrad(ctx.handle, col_buffer.data_as<void>(),
                                  temp_gradient_buffer.data_as<void>(), weights.grad_as<void>(),
                                  output_size, kernel_size, config.out_channels, type_desc);

  ctx.engine->legacy_conv2d_dgrad(ctx.handle, temp_gradient_buffer.data_as<void>(),
                                  weights.data_as<void>(),
                                  temp_col_grad_matrix_buffer.data_as<void>(), output_size,
                                  kernel_size, config.out_channels, type_desc);

  col2im(temp_col_grad_matrix_buffer, grad_input, batch_size, config.in_channels, input_h, input_w,
         config.kernel_h, config.kernel_w, config.stride_h, config.stride_w, config.pad_h,
         config.pad_w, ctx.handle.get_stream());

  if (config.use_bias) {
    ctx.engine->legacy_conv2d_bgrad(ctx.handle, grad_output.data_as<void>(), bias.grad_as<void>(),
                                    batch_size, output_h, output_w, config.out_channels, type_desc);
  }

  return grad_input;
}

LayerConfig LegacyConv2DOp::get_config(const Config &config, const std::string &name) {
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

LegacyConv2DOp::Config LegacyConv2DOp::parse_config(const LayerConfig &config) {
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

LegacyConv2D::LegacyConv2D(size_t in_channels, size_t out_channels, size_t kernel_h,
                           size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h,
                           size_t pad_w, bool use_bias, const std::string &name)
    : FunctionalLayer(LegacyConv2DOp::Config{in_channels, out_channels, kernel_h, kernel_w,
                                             stride_h, stride_w, pad_h, pad_w, use_bias},
                      name) {
  impl_->register_param(
      "weights", {in_channels, out_channels, kernel_h, kernel_w},
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
