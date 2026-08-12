#ifdef TUNX_USE_CUDNN
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn_frontend.h>
#include <cudnn_frontend/graph_interface.h>
#include <cudnn_frontend/graph_properties.h>
#include <cudnn_frontend_utils.h>
#include <cudnn_graph.h>
#include <fmt/core.h>

#include "internal.cuh"
#include "math/cuda/axpy.hpp"
#include "nn/engines/cudnn_engine.hpp"
#include "nn/engines/engine_handle.hpp"
#include "nn/stats/stats.hpp"
#include "type/type.hpp"

namespace tunx {

struct batchnorm_fwd_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> scale;
  std::shared_ptr<fe::graph::Tensor_attributes> bias;
  std::shared_ptr<fe::graph::Tensor_attributes> y;
  std::shared_ptr<fe::graph::Tensor_attributes> mean;
  std::shared_ptr<fe::graph::Tensor_attributes> invar;
  std::shared_ptr<fe::graph::Tensor_attributes> prev_mean;
  std::shared_ptr<fe::graph::Tensor_attributes> prev_var;
  std::shared_ptr<fe::graph::Tensor_attributes> next_mean;
  std::shared_ptr<fe::graph::Tensor_attributes> next_var;
  std::shared_ptr<fe::graph::Tensor_attributes> relu_mask;
  size_t workspace_size;

  batchnorm_fwd_graph(cudnnHandle_t handle, const BatchNormStats& stats, DTypeDesc& type_desc) {
    const int64 n = static_cast<int64>(stats.batch_size);
    const int64 c = static_cast<int64>(stats.channels);
    const int64 h = static_cast<int64>(stats.height);
    const int64 w = static_cast<int64>(stats.width);
    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto param_type = to_fe_data_type(type_desc.param_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    x = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("X")
                          .set_dim({n, c, h, w})
                          .set_stride({h * w * c, 1, w * c, c})
                          .set_data_type(io_type));

    scale = graph->tensor(fe::graph::Tensor_attributes()
                              .set_name("scale")
                              .set_dim({1, c, 1, 1})
                              .set_stride({c, 1, c, c})
                              .set_data_type(param_type));

    bias = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("bias")
                             .set_dim({1, c, 1, 1})
                             .set_stride({c, 1, c, c})
                             .set_data_type(param_type));

    prev_mean = graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("prev_running_mean")
                                  .set_dim({1, c, 1, 1})
                                  .set_stride({c, 1, c, c})
                                  .set_data_type(fe::DataType_t::FLOAT));

    prev_var = graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name("prev_running_var")
                                 .set_dim({1, c, 1, 1})
                                 .set_stride({c, 1, c, c})
                                 .set_data_type(fe::DataType_t::FLOAT));

    auto epsilon = graph->tensor(stats.epsilon);
    auto momentum = graph->tensor(stats.momentum);

    auto bn_options = fe::graph::Batchnorm_attributes().set_epsilon(epsilon);
    bn_options.set_previous_running_stats(prev_mean, prev_var, momentum);

    auto outputs = graph->batchnorm(x, scale, bias, bn_options);
    y = outputs[0];
    mean = outputs[1];
    invar = outputs[2];
    next_mean = outputs[3];
    next_var = outputs[4];

    relu_mask = y;
    if (stats.use_relu) {
      y = graph->pointwise(
          y, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD));

      float lower_clip = 0.0f;
      auto s_lower_clip = graph->tensor(lower_clip);
      auto relu_lower_clip_mask_attr = fe::graph::Pointwise_attributes()
                                           .set_mode(fe::PointwiseMode_t::CMP_GT)
                                           .set_compute_data_type(compute_type);
      relu_mask = graph->pointwise(y, s_lower_clip, relu_lower_clip_mask_attr);
      relu_mask->set_output(true).set_data_type(fe::DataType_t::BOOLEAN);
    }

    y->set_output(true).set_data_type(io_type);
    mean->set_output(true).set_data_type(compute_type);
    invar->set_output(true).set_data_type(compute_type);
    next_mean->set_output(true).set_data_type(compute_type);
    next_var->set_output(true).set_data_type(compute_type);

    ensure_ok(graph->validate(), "batchnorm forward validate");
    ensure_ok(graph->build_operation_graph(handle), "batchnorm forward build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "batchnorm forward create plans");
    ensure_ok(graph->check_support(), "batchnorm forward check support");
    ensure_ok(graph->build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE, false),
              "batchnorm forward build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "batchnorm forward workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

struct batchnorm_inf_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> scale;
  std::shared_ptr<fe::graph::Tensor_attributes> bias;
  std::shared_ptr<fe::graph::Tensor_attributes> saved_mean;
  std::shared_ptr<fe::graph::Tensor_attributes> saved_var;
  std::shared_ptr<fe::graph::Tensor_attributes> y;
  size_t workspace_size;

  batchnorm_inf_graph(cudnnHandle_t handle, const BatchNormStats& stats, DTypeDesc& type_desc) {
    const int64 n = static_cast<int64>(stats.batch_size);
    const int64 c = static_cast<int64>(stats.channels);
    const int64 h = static_cast<int64>(stats.height);
    const int64 w = static_cast<int64>(stats.width);
    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto param_type = to_fe_data_type(type_desc.param_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    x = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("X")
                          .set_dim({n, c, h, w})
                          .set_stride({h * w * c, 1, w * c, c})
                          .set_data_type(io_type));

    scale = graph->tensor(fe::graph::Tensor_attributes()
                              .set_name("scale")
                              .set_dim({1, c, 1, 1})
                              .set_stride({c, 1, c, c})
                              .set_data_type(param_type));

    bias = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("bias")
                             .set_dim({1, c, 1, 1})
                             .set_stride({c, 1, c, c})
                             .set_data_type(param_type));

    saved_mean = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("saved_mean")
                                   .set_dim({1, c, 1, 1})
                                   .set_stride({c, 1, c, c})
                                   .set_data_type(compute_type));

    saved_var = graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("saved_var")
                                  .set_dim({1, c, 1, 1})
                                  .set_stride({c, 1, c, c})
                                  .set_data_type(compute_type));

    auto epsilon_tensor = graph->tensor(stats.epsilon);
    auto var_plus_eps =
        graph->pointwise(saved_var, epsilon_tensor,
                         fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD));
    auto inv_std = graph->pointwise(
        var_plus_eps, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RSQRT));
    auto x_minus_mean = graph->pointwise(
        x, saved_mean, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::SUB));
    auto normalized =
        graph->pointwise(x_minus_mean, inv_std,
                         fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL));
    auto scaled = graph->pointwise(
        normalized, scale, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL));
    y = graph->pointwise(scaled, bias,
                         fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD));

    if (stats.use_relu) {
      y = graph->pointwise(
          y, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD));
    }

    y->set_output(true).set_data_type(io_type);

    ensure_ok(graph->validate(), "batchnorm inference validate");
    ensure_ok(graph->build_operation_graph(handle), "batchnorm inference build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "batchnorm inference create plans");
    ensure_ok(graph->check_support(), "batchnorm inference check support");
    ensure_ok(graph->build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE, false),
              "batchnorm inference build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "batchnorm inference workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

struct batchnorm_bwd_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> dy;
  std::shared_ptr<fe::graph::Tensor_attributes> mask;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> scale;
  std::shared_ptr<fe::graph::Tensor_attributes> mean;
  std::shared_ptr<fe::graph::Tensor_attributes> invar;
  std::shared_ptr<fe::graph::Tensor_attributes> dx;
  std::shared_ptr<fe::graph::Tensor_attributes> dscale;
  std::shared_ptr<fe::graph::Tensor_attributes> dbias;
  size_t workspace_size;

  batchnorm_bwd_graph(cudnnHandle_t handle, const BatchNormStats& stats, DTypeDesc& type_desc) {
    const int64 n = static_cast<int64>(stats.batch_size);
    const int64 c = static_cast<int64>(stats.channels);
    const int64 h = static_cast<int64>(stats.height);
    const int64 w = static_cast<int64>(stats.width);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto param_type = to_fe_data_type(type_desc.param_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    dy = graph->tensor(fe::graph::Tensor_attributes()
                           .set_name("DY")
                           .set_dim({n, c, h, w})
                           .set_stride({h * w * c, 1, w * c, c}));
    auto DX_drelu = dy;
    if (stats.use_relu) {
      mask = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Mask")
                               .set_dim({n, c, h, w})
                               .set_stride({h * w * c, 1, w * c, c})
                               .set_data_type(fe::DataType_t::BOOLEAN));

      auto mul_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
      DX_drelu = graph->pointwise(dy, mask, mul_options);
      DX_drelu->set_output(false).set_data_type(io_type);
    }

    x = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("X")
                          .set_dim({n, c, h, w})
                          .set_stride({h * w * c, 1, w * c, c}));

    scale = graph->tensor(fe::graph::Tensor_attributes()
                              .set_name("scale")
                              .set_dim({1, c, 1, 1})
                              .set_stride({c, 1, c, c})
                              .set_data_type(param_type));

    mean = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("mean")
                             .set_dim({1, c, 1, 1})
                             .set_stride({c, 1, c, c})
                             .set_data_type(compute_type));

    invar = graph->tensor(fe::graph::Tensor_attributes()
                              .set_name("inv_variance")
                              .set_dim({1, c, 1, 1})
                              .set_stride({c, 1, c, c})
                              .set_data_type(compute_type));

    auto DBN_options =
        fe::graph::Batchnorm_backward_attributes().set_saved_mean_and_inv_variance(mean, invar);
    auto outputs = graph->batchnorm_backward(DX_drelu, x, scale, DBN_options);
    dx = outputs[0];
    dscale = outputs[1];
    dbias = outputs[2];

    dx->set_output(true);
    dscale->set_output(true).set_data_type(param_type);
    dbias->set_output(true).set_data_type(param_type);

    ensure_ok(graph->validate(), "batchnorm backward validate");
    ensure_ok(graph->build_operation_graph(handle), "batchnorm backward build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "batchnorm backward create plans");
    ensure_ok(graph->check_support(), "batchnorm backward check support");
    ensure_ok(graph->build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE, false),
              "batchnorm backward build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "batchnorm backward workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

struct layernorm_fwd_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> scale;
  std::shared_ptr<fe::graph::Tensor_attributes> bias;
  std::shared_ptr<fe::graph::Tensor_attributes> y;
  std::shared_ptr<fe::graph::Tensor_attributes> mean;
  std::shared_ptr<fe::graph::Tensor_attributes> inv_variance;
  size_t workspace_size;

  layernorm_fwd_graph(cudnnHandle_t handle, const LayerNormStats& stats, DTypeDesc& type_desc) {
    const int64 b = static_cast<int64>(stats.batch_size);
    const int64 s = static_cast<int64>(stats.seq_len);
    const int64 d = static_cast<int64>(stats.channels);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto param_type = to_fe_data_type(type_desc.param_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    x = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("input")
                          .set_dim({b, s, d})
                          .set_stride({s * d, d, 1})
                          .set_data_type(io_type));

    auto epsilon = graph->tensor(stats.epsilon);

    auto ln_options = fe::graph::Layernorm_attributes()
                          .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
                          .set_epsilon(epsilon);

    scale = graph->tensor(fe::graph::Tensor_attributes()
                              .set_name("scale")
                              .set_dim({1, 1, d})
                              .set_stride({d, d, 1})
                              .set_data_type(param_type));

    bias = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("bias")
                             .set_dim({1, 1, d})
                             .set_stride({d, d, 1})
                             .set_data_type(param_type));

    auto outputs = graph->layernorm(x, scale, bias, ln_options);
    y = outputs[0];
    mean = outputs[1];
    inv_variance = outputs[2];

    y->set_output(true).set_data_type(io_type);
    mean->set_output(true).set_data_type(compute_type);
    inv_variance->set_output(true).set_data_type(compute_type);

    ensure_ok(graph->validate(), "layernorm_fwd validate");
    ensure_ok(graph->build_operation_graph(handle), "layernorm_fwd build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "layernorm_fwd create plans");
    ensure_ok(graph->check_support(), "layernorm_fwd check support");
    ensure_ok(graph->build_plans(), "layernorm_fwd build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "layernorm_fwd workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

struct layernorm_inf_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> scale;
  std::shared_ptr<fe::graph::Tensor_attributes> bias;
  std::shared_ptr<fe::graph::Tensor_attributes> y;
  size_t workspace_size;

  layernorm_inf_graph(cudnnHandle_t handle, const LayerNormStats& stats, DTypeDesc& type_desc) {
    const int64 b = static_cast<int64>(stats.batch_size);
    const int64 s = static_cast<int64>(stats.seq_len);
    const int64 d = static_cast<int64>(stats.channels);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto param_type = to_fe_data_type(type_desc.param_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    x = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("input")
                          .set_dim({b, s, d})
                          .set_stride({s * d, d, 1})
                          .set_data_type(io_type));

    auto epsilon = graph->tensor(stats.epsilon);

    auto ln_options = fe::graph::Layernorm_attributes()
                          .set_forward_phase(fe::NormFwdPhase_t::INFERENCE)
                          .set_epsilon(epsilon);

    scale = graph->tensor(fe::graph::Tensor_attributes()
                              .set_name("scale")
                              .set_dim({1, 1, d})
                              .set_stride({d, d, 1})
                              .set_data_type(param_type));

    bias = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("bias")
                             .set_dim({1, 1, d})
                             .set_stride({d, d, 1})
                             .set_data_type(param_type));

    auto outputs = graph->layernorm(x, scale, bias, ln_options);
    y = outputs[0];

    y->set_output(true).set_data_type(io_type);

    ensure_ok(graph->validate(), "layernorm_inf validate");
    ensure_ok(graph->build_operation_graph(handle), "layernorm_inf build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "layernorm_inf create plans");
    ensure_ok(graph->check_support(), "layernorm_inf check support");
    ensure_ok(graph->build_plans(), "layernorm_inf build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "layernorm_inf workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

struct layernorm_bwd_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> dy;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> scale;
  std::shared_ptr<fe::graph::Tensor_attributes> mean;
  std::shared_ptr<fe::graph::Tensor_attributes> inv_variance;
  std::shared_ptr<fe::graph::Tensor_attributes> dx;
  std::shared_ptr<fe::graph::Tensor_attributes> dscale;
  std::shared_ptr<fe::graph::Tensor_attributes> dbias;
  size_t workspace_size;

  layernorm_bwd_graph(cudnnHandle_t handle, const LayerNormStats& stats, DTypeDesc& type_desc) {
    const int64 b = static_cast<int64>(stats.batch_size);
    const int64 s = static_cast<int64>(stats.seq_len);
    const int64 d = static_cast<int64>(stats.channels);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto param_type = to_fe_data_type(type_desc.param_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    dy = graph->tensor(fe::graph::Tensor_attributes()
                           .set_name("grad_output")
                           .set_dim({b, s, d})
                           .set_stride({s * d, d, 1})
                           .set_data_type(io_type));

    x = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("input")
                          .set_dim({b, s, d})
                          .set_stride({s * d, d, 1})
                          .set_data_type(io_type));

    mean = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("mean")
                             .set_dim({b, s, 1})
                             .set_stride({s, 1, 1})
                             .set_data_type(compute_type));

    inv_variance = graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name("inv_variance")
                                     .set_dim({b, s, 1})
                                     .set_stride({s, 1, 1})
                                     .set_data_type(compute_type));

    auto ln_bwd_options =
        fe::graph::Layernorm_backward_attributes().set_saved_mean_and_inv_variance(mean,
                                                                                   inv_variance);

    scale = graph->tensor(fe::graph::Tensor_attributes()
                              .set_name("scale")
                              .set_dim({1, 1, d})
                              .set_stride({d, d, 1})
                              .set_data_type(param_type));

    auto outputs = graph->layernorm_backward(dy, x, scale, ln_bwd_options);
    dx = outputs[0];
    dscale = outputs[1];
    dbias = outputs[2];

    dx->set_output(true).set_data_type(io_type);
    dscale->set_output(true).set_data_type(param_type);
    dbias->set_output(true).set_data_type(param_type);

    ensure_ok(graph->validate(), "layernorm_bwd validate");
    ensure_ok(graph->build_operation_graph(handle), "layernorm_bwd build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "layernorm_bwd create plans");
    ensure_ok(graph->check_support(), "layernorm_bwd check support");
    ensure_ok(graph->build_plans(), "layernorm_bwd build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "layernorm_bwd workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

WorkspaceReq CuDNNEngine::query_batchnorm_graph(engine_handle backend_handle,
                                                const BatchNormStats& stats, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();

  GraphCacheKey fwd_key{
      .op_type = OpType::BATCHNORM_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels, stats.height, stats.width},
      .attributes = {{"epsilon", stats.epsilon},
                     {"momentum", stats.momentum},
                     {"use_relu", stats.use_relu}},
  };
  auto it_fwd = graph_cache_.find(fwd_key);
  if (it_fwd == graph_cache_.end()) {
    it_fwd = graph_cache_.emplace(fwd_key, batchnorm_fwd_graph(handle, stats, type_desc)).first;
  }

  GraphCacheKey inf_key{
      .op_type = OpType::BATCHNORM_INFER,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels, stats.height, stats.width},
      .attributes = {{"epsilon", stats.epsilon},
                     {"momentum", stats.momentum},
                     {"use_relu", stats.use_relu}},
  };
  auto it_inf = graph_cache_.find(inf_key);
  if (it_inf == graph_cache_.end()) {
    it_inf = graph_cache_.emplace(inf_key, batchnorm_inf_graph(handle, stats, type_desc)).first;
  }

  GraphCacheKey bwd_key{
      .op_type = OpType::BATCHNORM_BWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels, stats.height, stats.width},
      .attributes = {{"epsilon", stats.epsilon},
                     {"momentum", stats.momentum},
                     {"use_relu", stats.use_relu}},
  };
  auto it_bwd = graph_cache_.find(bwd_key);
  if (it_bwd == graph_cache_.end()) {
    it_bwd = graph_cache_.emplace(bwd_key, batchnorm_bwd_graph(handle, stats, type_desc)).first;
  }

  auto& fwd_graph = std::any_cast<batchnorm_fwd_graph&>(it_fwd->second);
  auto& inf_graph = std::any_cast<batchnorm_inf_graph&>(it_inf->second);
  auto& bwd_graph = std::any_cast<batchnorm_bwd_graph&>(it_bwd->second);

  size_t temp = 2 * stats.channels * get_dtype_size(type_desc.param_dtype);
  return {fwd_graph.workspace_size, bwd_graph.workspace_size + temp, inf_graph.workspace_size};
}

WorkspaceReq CuDNNEngine::query_layernorm_graph(engine_handle backend_handle,
                                                const LayerNormStats& stats, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();

  GraphCacheKey fwd_key{
      .op_type = OpType::LAYERNORM_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.seq_len, stats.channels},
      .attributes = {{"epsilon", stats.epsilon}},
  };
  auto it_fwd = graph_cache_.find(fwd_key);
  if (it_fwd == graph_cache_.end()) {
    it_fwd = graph_cache_.emplace(fwd_key, layernorm_fwd_graph(handle, stats, type_desc)).first;
  }

  GraphCacheKey inf_key{
      .op_type = OpType::LAYERNORM_INFER,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.seq_len, stats.channels},
      .attributes = {{"epsilon", stats.epsilon}},
  };
  auto it_inf = graph_cache_.find(inf_key);
  if (it_inf == graph_cache_.end()) {
    it_inf = graph_cache_.emplace(inf_key, layernorm_inf_graph(handle, stats, type_desc)).first;
  }

  GraphCacheKey bwd_key{
      .op_type = OpType::LAYERNORM_BWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.seq_len, stats.channels},
      .attributes = {{"epsilon", stats.epsilon}},
  };
  auto it_bwd = graph_cache_.find(bwd_key);
  if (it_bwd == graph_cache_.end()) {
    it_bwd = graph_cache_.emplace(bwd_key, layernorm_bwd_graph(handle, stats, type_desc)).first;
  }

  auto& fwd_graph = std::any_cast<layernorm_fwd_graph&>(it_fwd->second);
  auto& inf_graph = std::any_cast<layernorm_inf_graph&>(it_inf->second);
  auto& bwd_graph = std::any_cast<layernorm_bwd_graph&>(it_bwd->second);

  size_t temp = 2 * stats.channels * get_dtype_size(type_desc.param_dtype);
  return {fwd_graph.workspace_size, bwd_graph.workspace_size + temp, inf_graph.workspace_size};
}

void CuDNNEngine::batchnorm_fwd(engine_handle backend_handle, const BatchNormStats& stats,
                                const void* input, const void* gamma, const void* beta,
                                void* output, void* prev_running_mean, void* prev_running_var,
                                void* next_running_mean, void* next_running_var, void* batch_mean,
                                void* batch_invar, void* relu_mask, void* workspace,
                                DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  GraphCacheKey key{
      .op_type = OpType::BATCHNORM_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels, stats.height, stats.width},
      .attributes = {{"epsilon", stats.epsilon},
                     {"momentum", stats.momentum},
                     {"use_relu", stats.use_relu}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error("cuDNN Graph not found for batchnorm fwd.");
  }
  auto& graph_struct = std::any_cast<batchnorm_fwd_graph&>(it->second);
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.x, const_cast<void*>(input)},
      {graph_struct.scale, const_cast<void*>(gamma)},
      {graph_struct.bias, const_cast<void*>(beta)},
      {graph_struct.y, output},
      {graph_struct.mean, batch_mean},
      {graph_struct.invar, batch_invar},
      {graph_struct.prev_mean, prev_running_mean},
      {graph_struct.prev_var, prev_running_var},
      {graph_struct.next_mean, next_running_mean},
      {graph_struct.next_var, next_running_var}};

  if (stats.use_relu) {
    variant_pack[graph_struct.relu_mask] = relu_mask;
  }

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "batchnorm fwd execute");
}

void CuDNNEngine::batchnorm_infer(engine_handle backend_handle, const BatchNormStats& stats,
                                  const void* input, const void* gamma, const void* beta,
                                  const void* saved_mean, const void* saved_var, void* output,
                                  void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  GraphCacheKey key{
      .op_type = OpType::BATCHNORM_INFER,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels, stats.height, stats.width},
      .attributes = {{"epsilon", stats.epsilon},
                     {"momentum", stats.momentum},
                     {"use_relu", stats.use_relu}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error("cuDNN Graph not found for batchnorm infer.");
  }
  auto& graph_struct = std::any_cast<batchnorm_inf_graph&>(it->second);
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.x, const_cast<void*>(input)},
      {graph_struct.scale, const_cast<void*>(gamma)},
      {graph_struct.bias, const_cast<void*>(beta)},
      {graph_struct.y, output},
      {graph_struct.saved_mean, const_cast<void*>(saved_mean)},
      {graph_struct.saved_var, const_cast<void*>(saved_var)}};

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "batchnorm infer execute");
}

void CuDNNEngine::batchnorm_bwd(engine_handle backend_handle, const BatchNormStats& stats,
                                const void* grad_output, const void* input, const void* relu_mask,
                                const void* gamma, void* grad_input, void* grad_gamma,
                                void* grad_beta, const void* batch_mean, const void* batch_invar,
                                void* workspace, DTypeDesc type_desc) {
  size_t grad_gamma_temp_size = stats.channels * get_dtype_size(type_desc.param_dtype);
  size_t grad_beta_temp_size = stats.channels * get_dtype_size(type_desc.param_dtype);
  void* grad_gamma_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_gamma_temp_size;
  void* grad_beta_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_beta_temp_size;

  assert(grad_gamma != grad_gamma_temp && "grad_gamma should be different from grad_gamma_temp");
  assert(grad_beta != grad_beta_temp && "grad_beta should be different from grad_beta_temp");
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  GraphCacheKey key{
      .op_type = OpType::BATCHNORM_BWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels, stats.height, stats.width},
      .attributes = {{"epsilon", stats.epsilon},
                     {"momentum", stats.momentum},
                     {"use_relu", stats.use_relu}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error("cuDNN Graph not found for batchnorm bwd.");
  }
  auto& graph_struct = std::any_cast<batchnorm_bwd_graph&>(it->second);
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.dy, const_cast<void*>(grad_output)},
      {graph_struct.x, const_cast<void*>(input)},
      {graph_struct.scale, const_cast<void*>(gamma)},
      {graph_struct.mean, const_cast<void*>(batch_mean)},
      {graph_struct.invar, const_cast<void*>(batch_invar)},
      {graph_struct.dx, grad_input},
      {graph_struct.dscale, grad_gamma_temp},
      {graph_struct.dbias, grad_beta_temp}};

  if (stats.use_relu) {
    variant_pack[graph_struct.mask] = const_cast<void*>(relu_mask);
  }

  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "batchnorm backward execute");

  size_t num_elements = stats.channels;
  cuda::axpy(grad_gamma_temp, grad_gamma, num_elements, type_desc.param_dtype, stream);
  cuda::axpy(grad_beta_temp, grad_beta, num_elements, type_desc.param_dtype, stream);
}

void CuDNNEngine::layernorm_fwd(engine_handle backend_handle, const LayerNormStats& stats,
                                const void* input, const void* gamma, const void* beta,
                                void* output, void* mean, void* inv_variance, void* workspace,
                                DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  GraphCacheKey key{
      .op_type = OpType::LAYERNORM_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.seq_len, stats.channels},
      .attributes = {{"epsilon", stats.epsilon}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error("cuDNN Graph not found for layernorm fwd.");
  }
  auto& graph_struct = std::any_cast<layernorm_fwd_graph&>(it->second);

  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack;
  variant_pack[graph_struct.x] = const_cast<void*>(input);
  variant_pack[graph_struct.scale] = const_cast<void*>(gamma);
  variant_pack[graph_struct.bias] = const_cast<void*>(beta);
  variant_pack[graph_struct.y] = output;
  variant_pack[graph_struct.mean] = mean;
  variant_pack[graph_struct.inv_variance] = inv_variance;

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "layernorm fwd execute");
}

void CuDNNEngine::layernorm_infer(engine_handle backend_handle, const LayerNormStats& stats,
                                  const void* input, const void* gamma, const void* beta,
                                  void* output, void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  GraphCacheKey key{
      .op_type = OpType::LAYERNORM_INFER,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.seq_len, stats.channels},
      .attributes = {{"epsilon", stats.epsilon}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error(
        "cuDNN Graph not found for layernorm infer. Please call query_layernorm_graph first.");
  }
  auto& graph_struct = std::any_cast<layernorm_inf_graph&>(it->second);

  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack;
  variant_pack[graph_struct.x] = const_cast<void*>(input);
  variant_pack[graph_struct.scale] = const_cast<void*>(gamma);
  variant_pack[graph_struct.bias] = const_cast<void*>(beta);
  variant_pack[graph_struct.y] = output;

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "layernorm infer execute");
}

void CuDNNEngine::layernorm_bwd(engine_handle backend_handle, const LayerNormStats& stats,
                                const void* grad_output, const void* input, const void* gamma,
                                const void* mean, const void* inv_variance, void* grad_input,
                                void* grad_gamma, void* grad_beta, void* workspace,
                                DTypeDesc type_desc) {
  size_t grad_gamma_temp_size = stats.channels * get_dtype_size(type_desc.param_dtype);
  size_t grad_beta_temp_size = stats.channels * get_dtype_size(type_desc.param_dtype);
  void* grad_gamma_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_gamma_temp_size;
  void* grad_beta_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_beta_temp_size;

  assert(grad_gamma != grad_gamma_temp && "grad_gamma should be different from grad_gamma_temp");
  assert(grad_beta != grad_beta_temp && "grad_beta should be different from grad_beta_temp");
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  GraphCacheKey key{
      .op_type = OpType::LAYERNORM_BWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.seq_len, stats.channels},
      .attributes = {{"epsilon", stats.epsilon}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error("cuDNN Graph not found for layernorm bwd.");
  }
  auto& graph_struct = std::any_cast<layernorm_bwd_graph&>(it->second);

  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack;
  variant_pack[graph_struct.dy] = const_cast<void*>(grad_output);
  variant_pack[graph_struct.x] = const_cast<void*>(input);
  variant_pack[graph_struct.mean] = const_cast<void*>(mean);
  variant_pack[graph_struct.inv_variance] = const_cast<void*>(inv_variance);
  variant_pack[graph_struct.scale] = const_cast<void*>(gamma);
  variant_pack[graph_struct.dx] = grad_input;
  variant_pack[graph_struct.dscale] = grad_gamma_temp;
  variant_pack[graph_struct.dbias] = grad_beta_temp;

  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "layernorm bwd execute");

  size_t num_elements = stats.channels;
  cuda::axpy(grad_gamma_temp, grad_gamma, num_elements, type_desc.param_dtype, stream);
  cuda::axpy(grad_beta_temp, grad_beta, num_elements, type_desc.param_dtype, stream);
}

}  // namespace tunx

#endif
