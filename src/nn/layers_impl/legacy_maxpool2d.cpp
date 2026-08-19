/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/legacy_maxpool2d.hpp"

#include <cstddef>
#include <stdexcept>

#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

Vec<Vec<size_t>> LegacyMaxPool2DOp::output_shapes(const Vec<Vec<size_t>> &input_shapes,
                                                  const Config &config) {
  if (input_shapes.size() != 1 || input_shapes[0].size() != 4) {
    throw std::invalid_argument("LegacyMaxPool2DOp expects 4D input including batch size");
  }

  size_t batch_size = input_shapes[0][0];
  size_t channels = input_shapes[0][1];
  size_t output_h = (input_shapes[0][2] + 2 * config.pad_h - config.pool_h) / config.stride_h + 1;
  size_t output_w = (input_shapes[0][3] + 2 * config.pad_w - config.pool_w) / config.stride_w + 1;

  return {{batch_size, channels, output_h, output_w}};
}

Tensor LegacyMaxPool2DOp::forward(OpContext &ctx, const Tensor &input, const Config &config) {
  if (config.pool_h == 0 || config.pool_w == 0) {
    throw std::invalid_argument("Pool dimensions must be positive");
  }
  if (config.stride_h == 0 || config.stride_w == 0) {
    throw std::invalid_argument("Stride dimensions must be positive");
  }

  const auto &shape = input.shape();
  if (shape.size() != 4) {
    throw std::invalid_argument("MaxPool2D: Input tensor must be 4-dimensional (NCHW)");
  }
  size_t batch_size = shape[0];
  size_t channels = shape[1];
  size_t input_h = shape[2];
  size_t input_w = shape[3];

  size_t output_h = (input_h + 2 * config.pad_h - config.pool_h) / config.stride_h + 1;
  size_t output_w = (input_w + 2 * config.pad_w - config.pool_w) / config.stride_w + 1;

  Tensor output = ctx.make_tensor({batch_size, channels, output_h, output_w}, input.dtype());

  Tensor mask_indices =
      ctx.make_tensor({batch_size, channels, output_h, output_w}, DType_t::SIZE_T);
  if (ctx.is_training) {
    ctx.residuals["mask_indices"] = mask_indices;
  }

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  ctx.engine->legacy_maxpool2d_fwd(
      ctx.handle, input.data_as<void>(), output.data_as<void>(), batch_size, channels, input_h,
      input_w, output_h, output_w, config.pool_h, config.pool_w, config.stride_h, config.stride_w,
      config.pad_h, config.pad_w, mask_indices.data_as<void>(), type_desc);

  return output;
}

Tensor LegacyMaxPool2DOp::backward(OpContext &ctx, const Tensor &grad_output,
                                   const Config &config) {
  Tensor &mask_indices = ctx.residuals["mask_indices"];

  const auto &grad_shape = grad_output.shape();
  if (grad_shape.size() != 4) {
    throw std::invalid_argument("MaxPool2D: Gradient tensor must be 4-dimensional (NCHW)");
  }
  size_t batch_size = grad_shape[0];
  size_t channels = grad_shape[1];
  size_t output_h = grad_shape[2];
  size_t output_w = grad_shape[3];

  size_t input_h = (output_h - 1) * config.stride_h + config.pool_h - 2 * config.pad_h;
  size_t input_w = (output_w - 1) * config.stride_w + config.pool_w - 2 * config.pad_w;

  Tensor grad_input =
      ctx.make_tensor({batch_size, channels, input_h, input_w}, grad_output.dtype());

  fill(grad_input, 0.0f);

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  ctx.engine->legacy_maxpool2d_bwd(ctx.handle, grad_output.data_as<void>(),
                                   grad_input.data_as<void>(), batch_size, channels, output_h,
                                   output_w, mask_indices.data_as<void>(), type_desc);

  return grad_input;
}

LayerConfig LegacyMaxPool2DOp::get_config(const Config &config, const std::string &name) {
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

LegacyMaxPool2DOp::Config LegacyMaxPool2DOp::parse_config(const LayerConfig &config) {
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
