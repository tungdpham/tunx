/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/batchnorm.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "nn/param.hpp"
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace tunx {

void BatchNormOp::init(OpContext &ctx, const Config &config) {
  size_t num_features = config.num_features;

  Param gamma = ctx.make_param({num_features});
  fill(gamma.data(), 1.0f);

  Param beta = ctx.make_param({num_features});
  fill(beta.data(), 0.0f);

  Param prev_running_mean = ctx.make_param({num_features}, ctx.compute_dtype);
  prev_running_mean.set_requires_grad(false);
  fill(prev_running_mean.data(), 0.0f);

  Param prev_running_var = ctx.make_param({num_features}, ctx.compute_dtype);
  prev_running_var.set_requires_grad(false);
  fill(prev_running_var.data(), 1.0f);

  Param running_mean = ctx.make_param({num_features}, ctx.compute_dtype);
  running_mean.set_requires_grad(false);
  fill(running_mean.data(), 0.0f);

  Param running_var = ctx.make_param({num_features}, ctx.compute_dtype);
  running_var.set_requires_grad(false);
  fill(running_var.data(), 1.0f);
}

Vec<Vec<size_t>> BatchNormOp::output_shapes(const Vec<Vec<size_t>> &input_shapes,
                                            const Config &config) {
  if (input_shapes.size() != 1) {
    throw std::runtime_error("BatchNormOp: expected exactly 1 input");
  }
  return {input_shapes[0]};
}

Tensor BatchNormOp::forward(OpContext &ctx, const Tensor &input, const Param &gamma,
                            const Param &beta, Param &prev_running_mean, Param &prev_running_var,
                            Param &running_mean, Param &running_var, const Config &config) {
  if (input.dims() < 4) {
    throw std::invalid_argument("BatchNormOp: Input tensor must have at least 4 dimensions got " +
                                std::to_string(input.dims()) + " dims");
  }
  if (input.dim(3) != config.num_features) {
    throw std::invalid_argument("BatchNormOp: Input channels must match num_features " +
                                std::to_string(config.num_features) + ", but got " +
                                std::to_string(input.dim(3)));
  }

  size_t N = input.dim(0);
  size_t H = input.dim(1);
  size_t W = input.dim(2);
  size_t C = input.dim(3);

  BatchNormStats stats{
      .batch_size = N,
      .height = H,
      .width = W,
      .channels = C,
      .epsilon = config.epsilon,
      .momentum = config.momentum,
      .use_relu = config.use_relu,
  };

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  WorkspaceReq ws_req = ctx.engine->query_batchnorm_graph(ctx.handle, stats, type_desc);

  if (ctx.is_training) {
    ctx.residuals["input"] = input;

    Tensor batch_mean = ctx.make_tensor({C}, DType_t::FP32);
    Tensor batch_invar = ctx.make_tensor({C}, DType_t::FP32);
    ctx.residuals["batch_mean"] = batch_mean;
    ctx.residuals["batch_invar"] = batch_invar;

    Tensor relu_mask;
    if (config.use_relu) {
      relu_mask = ctx.make_tensor(input.shape(), DType_t::BOOL);
      ctx.residuals["relu_mask"] = relu_mask;
    }

    Tensor output = ctx.make_tensor(input.shape(), ctx.io_dtype);

    Tensor ws = ctx.make_tensor({ws_req.fwd_workspace}, DType_t::BYTE);

    ctx.engine->batchnorm_fwd(
        ctx.handle, stats, input.data_as<void>(), gamma.data_as<void>(), beta.data_as<void>(),
        output.data_as<void>(), prev_running_mean.data_as<void>(), prev_running_var.data_as<void>(),
        running_mean.data_as<void>(), running_var.data_as<void>(), batch_mean.data_as<void>(),
        batch_invar.data_as<void>(), config.use_relu ? relu_mask.data_as<void>() : nullptr,
        ws.data_as<void>(), type_desc);

    copy(running_mean.data(), prev_running_mean.data(), ctx.handle.get_stream());
    copy(running_var.data(), prev_running_var.data(), ctx.handle.get_stream());

    return output;
  } else {
    Tensor output = ctx.make_tensor(input.shape(), ctx.io_dtype);

    Tensor ws = ctx.make_tensor({ws_req.inf_workspace}, DType_t::BYTE);

    ctx.engine->batchnorm_infer(ctx.handle, stats, input.data_as<void>(), gamma.data_as<void>(),
                                beta.data_as<void>(), running_mean.data_as<void>(),
                                running_var.data_as<void>(), output.data_as<void>(),
                                ws.data_as<void>(), type_desc);

    return output;
  }
}

Tensor BatchNormOp::backward(OpContext &ctx, const Tensor &grad_output, Param &gamma, Param &beta,
                             const Config &config) {
  const Tensor &input = ctx.residuals["input"];
  Tensor &batch_mean = ctx.residuals["batch_mean"];
  Tensor &batch_invar = ctx.residuals["batch_invar"];
  Tensor relu_mask = config.use_relu ? ctx.residuals["relu_mask"] : Tensor();

  size_t N = grad_output.dim(0);
  size_t H = grad_output.dim(1);
  size_t W = grad_output.dim(2);
  size_t C = grad_output.dim(3);

  BatchNormStats stats{
      .batch_size = N,
      .height = H,
      .width = W,
      .channels = C,
      .epsilon = config.epsilon,
      .momentum = config.momentum,
      .use_relu = config.use_relu,
  };

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  Tensor grad_input = ctx.make_tensor(grad_output.shape(), ctx.io_dtype);

  WorkspaceReq ws_req = ctx.engine->query_batchnorm_graph(ctx.handle, stats, type_desc);
  Tensor workspace = ctx.make_tensor({ws_req.bwd_workspace}, DType_t::BYTE);

  ctx.engine->batchnorm_bwd(ctx.handle, stats, grad_output.data_as<void>(), input.data_as<void>(),
                            config.use_relu ? relu_mask.data_as<void>() : nullptr,
                            gamma.data_as<void>(), grad_input.data_as<void>(),
                            gamma.grad_as<void>(), beta.grad_as<void>(), batch_mean.data_as<void>(),
                            batch_invar.data_as<void>(), workspace.data_as<void>(), type_desc);

  return grad_input;
}

LayerConfig BatchNormOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.name = name;
  lcfg.type = TYPE_NAME;
  lcfg.set("num_features", config.num_features);
  lcfg.set("epsilon", config.epsilon);
  lcfg.set("momentum", config.momentum);
  lcfg.set("affine", config.affine);
  lcfg.set("use_relu", config.use_relu);
  return lcfg;
}

BatchNormOp::Config BatchNormOp::parse_config(const LayerConfig &config) {
  Config c;
  c.num_features = config.get<size_t>("num_features");
  c.epsilon = config.get<float>("epsilon", 1e-5f);
  c.momentum = config.get<float>("momentum", 0.1f);
  c.affine = config.get<bool>("affine", true);
  c.use_relu = config.get<bool>("use_relu", false);
  return c;
}

}  // namespace tunx
