/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/legacy_dense.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "nn/engines/iengine.hpp"
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace tunx {

Vec<Vec<size_t>> LegacyDenseOp::output_shapes(const Vec<Vec<size_t>> &input_shapes,
                                              const Config &config) {
  if (input_shapes.empty() || input_shapes[0].empty()) {
    throw std::runtime_error("LegacyDenseOp::output_shapes: Input shape is empty.");
  }
  Vec<size_t> out_shape = input_shapes[0];
  out_shape.back() = config.output_features;
  return {out_shape};
}

Tensor LegacyDenseOp::forward(OpContext &ctx, const Tensor &input, const Param &weights,
                              const Param &bias, const Config &config) {
  const Vec<size_t> &in_shape = input.shape();
  size_t last_dim = in_shape.back();
  size_t batch_size = 1;
  for (size_t i = 0; i < in_shape.size() - 1; ++i) {
    batch_size *= in_shape[i];
  }

  if (last_dim != config.input_features) {
    std::cerr << "Input last dim: " << last_dim << " features, expected: " << config.input_features
              << " features" << std::endl;
    throw std::invalid_argument("Input feature size mismatch in LegacyDenseOp");
  }

  if (ctx.is_training) {
    ctx.residuals["input"] = input;
  }

  Vec<size_t> out_shape = in_shape;
  out_shape.back() = config.output_features;
  Tensor output = ctx.make_tensor(out_shape, ctx.io_dtype);

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  ctx.engine->legacy_dense_fwd(ctx.handle, input.data_as<void>(), weights.data_as<void>(),
                               output.data_as<void>(), batch_size, config.input_features,
                               config.output_features, type_desc);

  if (config.use_bias) {
    ctx.engine->legacy_dense_add_bias(ctx.handle, output.data_as<void>(), bias.data_as<void>(),
                                      batch_size, config.output_features, type_desc);
  }

  return output;
}

Tensor LegacyDenseOp::backward(OpContext &ctx, const Tensor &grad_output, Param &weights,
                               Param &bias, const Config &config) {
  if (grad_output.shape().back() != config.output_features) {
    throw std::invalid_argument("Gradient feature size mismatch in LegacyDenseOp");
  }
  const Tensor &input = ctx.residuals["input"];
  const Vec<size_t> &in_shape = input.shape();
  size_t batch_size = 1;
  for (size_t i = 0; i < in_shape.size() - 1; ++i) {
    batch_size *= in_shape[i];
  }

  Tensor grad_input = ctx.make_tensor(input.shape(), ctx.io_dtype);

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  ctx.engine->legacy_dense_wgrad(ctx.handle, input.data_as<void>(), grad_output.data_as<void>(),
                                 weights.grad_as<void>(), batch_size, config.input_features,
                                 config.output_features, type_desc);

  if (config.use_bias) {
    ctx.engine->legacy_dense_bgrad(ctx.handle, grad_output.data_as<void>(), bias.grad_as<void>(),
                                   batch_size, config.output_features, type_desc);
  }

  ctx.engine->legacy_dense_dgrad(ctx.handle, grad_output.data_as<void>(), weights.data_as<void>(),
                                 grad_input.data_as<void>(), batch_size, config.input_features,
                                 config.output_features, type_desc);

  return grad_input;
}

LayerConfig LegacyDenseOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.name = name;
  lcfg.type = TYPE_NAME;
  lcfg.set("input_features", config.input_features);
  lcfg.set("output_features", config.output_features);
  lcfg.set("use_bias", config.use_bias);
  return lcfg;
}

LegacyDenseOp::Config LegacyDenseOp::parse_config(const LayerConfig &config) {
  Config c;
  c.input_features = config.get<size_t>("input_features");
  c.output_features = config.get<size_t>("output_features");
  c.use_bias = config.get<bool>("use_bias");
  return c;
}

LegacyDense::LegacyDense(size_t input_features, size_t output_features, bool use_bias,
                         const std::string &name)
    : FunctionalLayer(LegacyDenseOp::Config{input_features, output_features, use_bias}, name) {
  impl_->register_param(
      "weights", {output_features, input_features}, [input_features](Param &p, OpContext &ctx) {
        float stddev = static_cast<float>(1.0 / std::sqrt(static_cast<double>(input_features)));
        long long seed = ctx.use_seed ? ctx.srand_seed
                                      : std::chrono::system_clock::now().time_since_epoch().count();
        fill_normal(p.data(), 0, stddev, seed);
      });

  if (use_bias) {
    impl_->register_param("bias", {output_features}, [input_features](Param &p, OpContext &ctx) {
      float stddev = static_cast<float>(1.0 / std::sqrt(static_cast<double>(input_features)));
      long long seed = ctx.use_seed ? ctx.srand_seed
                                    : std::chrono::system_clock::now().time_since_epoch().count();
      fill_normal(p.data(), 0, stddev, seed);
    });
  } else {
    impl_->register_param("bias_dummy", {0}, nullptr);
  }
}

}  // namespace tunx
