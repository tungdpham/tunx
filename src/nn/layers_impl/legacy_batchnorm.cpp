/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/legacy_batchnorm.hpp"

#include <cmath>
#include <stdexcept>

#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

Vec<Vec<size_t>> LegacyBatchNormOp::output_shapes(const Vec<Vec<size_t>> &input_shapes,
                                                  const Config &config) {
  return input_shapes;
}

Tensor LegacyBatchNormOp::forward(OpContext &ctx, const Tensor &input, Param &gamma, Param &beta,
                                  Param &running_mean, Param &running_var, const Config &config) {
  if (input.dims() < 3) {
    throw std::invalid_argument("BatchNorm: Input tensor must have at least 3 dimensions");
  }
  if (input.dim(1) != config.num_features) {
    throw std::invalid_argument("BatchNorm: Input channels must match num_features");
  }

  size_t batch_size, channels, spatial_size;
  batch_size = input.dim(0);
  channels = input.dim(1);
  spatial_size = input.stride(1);

  Tensor output = ctx.make_tensor(input.shape(), ctx.io_dtype);

  Tensor norm = ctx.make_tensor(input.shape(), ctx.io_dtype);
  Tensor batch_inv_std = ctx.make_tensor({config.num_features}, ctx.io_dtype);
  Tensor batch_mean = ctx.make_tensor({config.num_features}, ctx.io_dtype);

  ctx.residuals["norm"] = norm;
  ctx.residuals["inv_std"] = batch_inv_std;
  ctx.residuals["mean"] = batch_mean;

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  if (ctx.is_training) {
    ctx.engine->legacy_batchnorm_fwd(ctx.handle, input.data_as<void>(), batch_mean.data_as<void>(),
                                     batch_inv_std.data_as<void>(), running_mean.data_as<void>(),
                                     running_var.data_as<void>(), gamma.data_as<void>(),
                                     beta.data_as<void>(), output.data_as<void>(),
                                     norm.data_as<void>(), batch_size, channels, spatial_size,
                                     config.momentum, config.epsilon, config.affine, type_desc);
  } else {
    ctx.engine->legacy_batchnorm_infer(ctx.handle, input.data_as<void>(),
                                       running_mean.data_as<void>(), running_var.data_as<void>(),
                                       gamma.data_as<void>(), beta.data_as<void>(),
                                       output.data_as<void>(), batch_size, channels, spatial_size,
                                       config.epsilon, config.affine, type_desc);
  }

  return output;
}

Tensor LegacyBatchNormOp::backward(OpContext &ctx, const Tensor &grad_output, Param &gamma,
                                   Param &beta, const Config &config) {
  const Tensor &norm = ctx.residuals["norm"];
  const Tensor &inv_std = ctx.residuals["inv_std"];

  size_t batch_size = grad_output.dim(0);
  size_t channels = grad_output.dim(1);
  size_t spatial_size = grad_output.stride(1);

  Tensor grad_input = ctx.make_tensor(grad_output.shape(), ctx.io_dtype);
  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  ctx.engine->legacy_batchnorm_bwd(
      ctx.handle, grad_output.data_as<void>(), norm.data_as<void>(), inv_std.data_as<void>(),
      gamma.data_as<void>(), gamma.grad_as<void>(), beta.grad_as<void>(),
      grad_input.data_as<void>(), batch_size, channels, spatial_size, config.affine, type_desc);

  return grad_input;
}

LayerConfig LegacyBatchNormOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.name = name;
  lcfg.type = TYPE_NAME;
  lcfg.set("num_features", config.num_features);
  lcfg.set("epsilon", config.epsilon);
  lcfg.set("momentum", config.momentum);
  lcfg.set("affine", config.affine);
  return lcfg;
}

LegacyBatchNormOp::Config LegacyBatchNormOp::parse_config(const LayerConfig &config) {
  Config c;
  c.num_features = config.get<size_t>("num_features");
  c.epsilon = config.get<float>("epsilon");
  c.momentum = config.get<float>("momentum");
  c.affine = config.get<bool>("affine");
  return c;
}

LegacyBatchNorm::LegacyBatchNorm(size_t num_features, float epsilon, float momentum, bool affine,
                                 const std::string &name)
    : FunctionalLayer(LegacyBatchNormOp::Config{num_features, epsilon, momentum, affine}, name) {
  impl_->register_param("gamma", {num_features},
                        [](Param &p, OpContext &ctx) { fill(p.data(), 1.0f); });
  impl_->register_param("beta", {num_features},
                        [](Param &p, OpContext &ctx) { fill(p.data(), 0.0f); });
  impl_->register_param("running_mean", {num_features}, [](Param &p, OpContext &ctx) {
    p.set_requires_grad(false);
    fill(p.data(), 0.0f);
  });
  impl_->register_param("running_var", {num_features}, [](Param &p, OpContext &ctx) {
    p.set_requires_grad(false);
    fill(p.data(), 1.0f);
  });
}

}  // namespace tunx
