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

#include <memory>
#include <stdexcept>
#include <unordered_map>

#include "internal.cuh"
#include "math/cuda/axpy.hpp"
#include "nn/engines/cudnn_engine.hpp"
#include "nn/engines/engine_handle.hpp"
#include "nn/stats/stats.hpp"
#include "type/type.hpp"

namespace tunx {

struct conv2d_fwd_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> w;
  std::shared_ptr<fe::graph::Tensor_attributes> b;
  std::shared_ptr<fe::graph::Tensor_attributes> y;

  size_t workspace_size;

  conv2d_fwd_graph(cudnnHandle_t handle, const Conv2DStats& stats, DTypeDesc& type_desc) {
    const int64 n = static_cast<int64>(stats.batch_size);
    const int64 c = static_cast<int64>(stats.in_channels);
    const int64 h = static_cast<int64>(stats.input_h);
    const int64 w_dim = static_cast<int64>(stats.input_w);
    const int64 k = static_cast<int64>(stats.out_channels);
    const int64 r = static_cast<int64>(stats.kernel_h);
    const int64 s = static_cast<int64>(stats.kernel_w);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    x = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("X")
                          .set_dim({n, c, h, w_dim})
                          .set_stride({h * w_dim * c, 1, w_dim * c, c}));

    w = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("W")
                          .set_dim({k, c, r, s})
                          .set_stride({r * s * c, 1, s * c, c}));

    auto conv_options =
        fe::graph::Conv_fprop_attributes()
            .set_pre_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_post_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_stride({static_cast<int64>(stats.stride_h), static_cast<int64>(stats.stride_w)})
            .set_dilation({1, 1});

    auto conv_output = graph->conv_fprop(x, w, conv_options);

    if (stats.use_bias) {
      b = graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("B")
                            .set_dim({1, k, 1, 1})
                            .set_stride({k, 1, k, k}));

      auto bias_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
      y = graph->pointwise(conv_output, b, bias_options);
      y->set_output(true).set_data_type(io_type);
    } else {
      y = conv_output;
      y->set_output(true).set_data_type(io_type);
    }

    ensure_ok(graph->validate(), "conv_fprop validate");
    ensure_ok(graph->build_operation_graph(handle), "conv_fprop build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "conv_fprop create plans");
    ensure_ok(graph->check_support(), "conv_fprop check support");
    ensure_ok(graph->build_plans(), "conv_fprop build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "conv_fprop workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

struct conv2d_dgrad_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> dy;
  std::shared_ptr<fe::graph::Tensor_attributes> w;
  std::shared_ptr<fe::graph::Tensor_attributes> dx;

  size_t workspace_size;

  conv2d_dgrad_graph(cudnnHandle_t handle, const Conv2DStats& stats, DTypeDesc& type_desc) {
    const int64 n = static_cast<int64>(stats.batch_size);
    const int64 c = static_cast<int64>(stats.in_channels);
    const int64 h = static_cast<int64>(stats.input_h);
    const int64 w_dim = static_cast<int64>(stats.input_w);
    const int64 k = static_cast<int64>(stats.out_channels);
    const int64 r = static_cast<int64>(stats.kernel_h);
    const int64 s = static_cast<int64>(stats.kernel_w);
    const int64 output_h = (stats.input_h + stats.pad_h * 2 - stats.kernel_h) / stats.stride_h + 1;
    const int64 output_w = (stats.input_w + stats.pad_w * 2 - stats.kernel_w) / stats.stride_w + 1;
    const int64 p = static_cast<int64>(output_h);
    const int64 q = static_cast<int64>(output_w);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    dy = graph->tensor(fe::graph::Tensor_attributes()
                           .set_name("DY")
                           .set_dim({n, k, p, q})
                           .set_stride({p * q * k, 1, q * k, k}));

    w = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("W")
                          .set_dim({k, c, r, s})
                          .set_stride({r * s * c, 1, s * c, c}));

    auto dgrad_options =
        fe::graph::Conv_dgrad_attributes()
            .set_pre_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_post_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_stride({static_cast<int64>(stats.stride_h), static_cast<int64>(stats.stride_w)})
            .set_dilation({1, 1});

    dx = graph->conv_dgrad(dy, w, dgrad_options);
    dx->set_dim({n, c, h, w_dim})
        .set_stride({h * w_dim * c, 1, w_dim * c, c})
        .set_data_type(io_type)
        .set_output(true);

    ensure_ok(graph->validate(), "conv_dgrad validate");
    ensure_ok(graph->build_operation_graph(handle), "conv_dgrad build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "conv_dgrad create plans");
    ensure_ok(graph->check_support(), "conv_dgrad check support");
    ensure_ok(graph->build_plans(), "conv_dgrad build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "conv_dgrad workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

struct conv2d_wgrad_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> dy;
  std::shared_ptr<fe::graph::Tensor_attributes> dw;

  size_t workspace_size;

  conv2d_wgrad_graph(cudnnHandle_t handle, const Conv2DStats& stats, DTypeDesc& type_desc) {
    const int64 n = static_cast<int64>(stats.batch_size);
    const int64 c = static_cast<int64>(stats.in_channels);
    const int64 h = static_cast<int64>(stats.input_h);
    const int64 w_dim = static_cast<int64>(stats.input_w);
    const int64 k = static_cast<int64>(stats.out_channels);
    const int64 r = static_cast<int64>(stats.kernel_h);
    const int64 s = static_cast<int64>(stats.kernel_w);
    const int64 output_h = (stats.input_h + stats.pad_h * 2 - stats.kernel_h) / stats.stride_h + 1;
    const int64 output_w = (stats.input_w + stats.pad_w * 2 - stats.kernel_w) / stats.stride_w + 1;
    const int64 p = static_cast<int64>(output_h);
    const int64 q = static_cast<int64>(output_w);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto param_type = to_fe_data_type(type_desc.param_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    x = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("X")
                          .set_dim({n, c, h, w_dim})
                          .set_stride({h * w_dim * c, 1, w_dim * c, c}));

    dy = graph->tensor(fe::graph::Tensor_attributes()
                           .set_name("DY")
                           .set_dim({n, k, p, q})
                           .set_stride({p * q * k, 1, q * k, k}));

    auto wgrad_options =
        fe::graph::Conv_wgrad_attributes()
            .set_pre_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_post_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_stride({static_cast<int64>(stats.stride_h), static_cast<int64>(stats.stride_w)})
            .set_dilation({1, 1});

    dw = graph->conv_wgrad(dy, x, wgrad_options);
    dw->set_output(true)
        .set_dim({k, c, r, s})
        .set_stride({r * s * c, 1, s * c, c})
        .set_data_type(param_type);

    ensure_ok(graph->validate(), "conv_wgrad validate");
    ensure_ok(graph->build_operation_graph(handle), "conv_wgrad build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "conv_wgrad create plans");
    ensure_ok(graph->check_support(), "conv_wgrad check support");
    ensure_ok(graph->build_plans(), "conv_wgrad build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "conv_wgrad workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

struct conv2d_bgrad_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> dy;
  std::shared_ptr<fe::graph::Tensor_attributes> db;

  size_t workspace_size;

  conv2d_bgrad_graph(cudnnHandle_t handle, const Conv2DStats& stats, DTypeDesc& type_desc) {
    const int64 n = static_cast<int64>(stats.batch_size);
    const int64 k = static_cast<int64>(stats.out_channels);
    const int64 output_h = (stats.input_h + stats.pad_h * 2 - stats.kernel_h) / stats.stride_h + 1;
    const int64 output_w = (stats.input_w + stats.pad_w * 2 - stats.kernel_w) / stats.stride_w + 1;
    const int64 p = static_cast<int64>(output_h);
    const int64 q = static_cast<int64>(output_w);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto param_type = to_fe_data_type(type_desc.param_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    dy = graph->tensor(fe::graph::Tensor_attributes()
                           .set_name("DY")
                           .set_dim({n, k, p, q})
                           .set_stride({p * q * k, 1, q * k, k}));

    auto reduction_options = fe::graph::Reduction_attributes().set_mode(fe::ReductionMode_t::ADD);

    db = graph->reduction(dy, reduction_options);
    db->set_output(true)
        .set_dim({1, k, 1, 1})
        .set_stride({k, 1, k, k})
        .set_data_type(param_type)
        .set_name("DB");

    ensure_ok(graph->validate(), "grad_bias validate");
    ensure_ok(graph->build_operation_graph(handle), "grad_bias build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "grad_bias create plans");
    ensure_ok(graph->check_support(), "grad_bias check support");
    ensure_ok(graph->build_plans(), "grad_bias build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "grad_bias workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

WorkspaceReq CuDNNEngine::query_conv2d_graph(engine_handle backend_handle, const Conv2DStats& stats,
                                             DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();

  GraphCacheKey fwd_key{
      .op_type = OpType::CONV2D_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_channels, stats.out_channels, stats.input_h,
               stats.input_w},
      .attributes = {{"kernel_h", stats.kernel_h},
                     {"kernel_w", stats.kernel_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w},
                     {"use_bias", stats.use_bias}},
  };
  auto it_fwd = graph_cache_.find(fwd_key);
  if (it_fwd == graph_cache_.end()) {
    it_fwd = graph_cache_.emplace(fwd_key, conv2d_fwd_graph(handle, stats, type_desc)).first;
  }

  GraphCacheKey dgrad_key{
      .op_type = OpType::CONV2D_DGRAD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_channels, stats.out_channels, stats.input_h,
               stats.input_w},
      .attributes = {{"kernel_h", stats.kernel_h},
                     {"kernel_w", stats.kernel_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w},
                     {"use_bias", stats.use_bias}},
  };
  auto it_dgrad = graph_cache_.find(dgrad_key);
  if (it_dgrad == graph_cache_.end()) {
    it_dgrad = graph_cache_.emplace(dgrad_key, conv2d_dgrad_graph(handle, stats, type_desc)).first;
  }

  GraphCacheKey wgrad_key{
      .op_type = OpType::CONV2D_WGRAD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_channels, stats.out_channels, stats.input_h,
               stats.input_w},
      .attributes = {{"kernel_h", stats.kernel_h},
                     {"kernel_w", stats.kernel_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w},
                     {"use_bias", stats.use_bias}},
  };
  auto it_wgrad = graph_cache_.find(wgrad_key);
  if (it_wgrad == graph_cache_.end()) {
    it_wgrad = graph_cache_.emplace(wgrad_key, conv2d_wgrad_graph(handle, stats, type_desc)).first;
  }

  auto& fwd_graph = std::any_cast<conv2d_fwd_graph&>(it_fwd->second);
  auto& dgrad_graph = std::any_cast<conv2d_dgrad_graph&>(it_dgrad->second);
  auto& wgrad_graph = std::any_cast<conv2d_wgrad_graph&>(it_wgrad->second);
  size_t wgrad_temp_size = stats.out_channels * stats.in_channels * stats.kernel_h *
                           stats.kernel_w * get_dtype_size(type_desc.param_dtype);

  size_t fwd_ws = fwd_graph.workspace_size;
  size_t bwd_ws =
      std::max(dgrad_graph.workspace_size, wgrad_graph.workspace_size + wgrad_temp_size);

  if (stats.use_bias) {
    GraphCacheKey bgrad_key{
        .op_type = OpType::CONV2D_BGRAD,
        .dtype_desc = type_desc,
        .dims = {stats.batch_size, stats.in_channels, stats.out_channels, stats.input_h,
                 stats.input_w},
        .attributes = {{"kernel_h", stats.kernel_h},
                       {"kernel_w", stats.kernel_w},
                       {"stride_h", stats.stride_h},
                       {"stride_w", stats.stride_w},
                       {"pad_h", stats.pad_h},
                       {"pad_w", stats.pad_w},
                       {"use_bias", stats.use_bias}},
    };
    auto it_bgrad = graph_cache_.find(bgrad_key);
    if (it_bgrad == graph_cache_.end()) {
      it_bgrad =
          graph_cache_.emplace(bgrad_key, conv2d_bgrad_graph(handle, stats, type_desc)).first;
    }
    auto& bgrad_graph = std::any_cast<conv2d_bgrad_graph&>(it_bgrad->second);
    size_t bgrad_temp_size = stats.out_channels * get_dtype_size(type_desc.param_dtype);
    bwd_ws = std::max(bwd_ws, bgrad_graph.workspace_size + bgrad_temp_size);
  }

  return {fwd_ws, bwd_ws, fwd_ws};
}

void CuDNNEngine::conv2d_fwd(engine_handle backend_handle, const Conv2DStats& stats,
                             const void* input, const void* weight, const void* bias, void* output,
                             void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  GraphCacheKey key{
      .op_type = OpType::CONV2D_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_channels, stats.out_channels, stats.input_h,
               stats.input_w},
      .attributes = {{"kernel_h", stats.kernel_h},
                     {"kernel_w", stats.kernel_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w},
                     {"use_bias", stats.use_bias}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error("cuDNN Graph not found for conv2d fwd.");
  }
  auto& graph_struct = std::any_cast<conv2d_fwd_graph&>(it->second);

  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.x, const_cast<void*>(input)},
      {graph_struct.w, const_cast<void*>(weight)},
      {graph_struct.y, output}};

  if (stats.use_bias && bias && graph_struct.b) {
    variant_pack[graph_struct.b] = const_cast<void*>(bias);
  }

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "conv2d fwd execute");
}

void CuDNNEngine::conv2d_dgrad(engine_handle backend_handle, const Conv2DStats& stats,
                               const void* grad_output, const void* weight, void* grad_input,
                               void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  GraphCacheKey key{
      .op_type = OpType::CONV2D_DGRAD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_channels, stats.out_channels, stats.input_h,
               stats.input_w},
      .attributes = {{"kernel_h", stats.kernel_h},
                     {"kernel_w", stats.kernel_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w},
                     {"use_bias", stats.use_bias}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error("cuDNN Graph not found for conv2d dgrad.");
  }
  auto& graph_struct = std::any_cast<conv2d_dgrad_graph&>(it->second);
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.dy, const_cast<void*>(grad_output)},
      {graph_struct.w, const_cast<void*>(weight)},
      {graph_struct.dx, grad_input}};

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "conv2d dgrad execute");
}

void CuDNNEngine::conv2d_wgrad(engine_handle backend_handle, const Conv2DStats& stats,
                               const void* grad_output, const void* input, void* grad_weight,
                               void* workspace, DTypeDesc type_desc) {
  size_t grad_weight_temp_size = stats.out_channels * stats.kernel_h * stats.kernel_w *
                                 stats.in_channels * get_dtype_size(type_desc.param_dtype);
  void* grad_weight_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_weight_temp_size;

  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  GraphCacheKey key{
      .op_type = OpType::CONV2D_WGRAD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_channels, stats.out_channels, stats.input_h,
               stats.input_w},
      .attributes = {{"kernel_h", stats.kernel_h},
                     {"kernel_w", stats.kernel_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w},
                     {"use_bias", stats.use_bias}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error("cuDNN Graph not found for conv2d wgrad.");
  }
  auto& graph_struct = std::any_cast<conv2d_wgrad_graph&>(it->second);

  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.x, const_cast<void*>(input)},
      {graph_struct.dy, const_cast<void*>(grad_output)},
      {graph_struct.dw, grad_weight_temp}};
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "conv2d wgrad execute");

  size_t num_elements =
      static_cast<size_t>(stats.out_channels) * stats.in_channels * stats.kernel_h * stats.kernel_w;
  cuda::axpy(grad_weight_temp, grad_weight, num_elements, type_desc.param_dtype, stream);
}

void CuDNNEngine::conv2d_bgrad(engine_handle backend_handle, const Conv2DStats& stats,
                               const void* grad_output, void* grad_bias, void* workspace,
                               DTypeDesc type_desc) {
  size_t grad_bias_temp_size = stats.out_channels * get_dtype_size(type_desc.param_dtype);
  void* grad_bias_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_bias_temp_size;

  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  GraphCacheKey key{
      .op_type = OpType::CONV2D_BGRAD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_channels, stats.out_channels, stats.input_h,
               stats.input_w},
      .attributes = {{"kernel_h", stats.kernel_h},
                     {"kernel_w", stats.kernel_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w},
                     {"use_bias", stats.use_bias}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error("cuDNN Graph not found for conv2d bgrad.");
  }
  auto& graph_struct = std::any_cast<conv2d_bgrad_graph&>(it->second);
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.dy, const_cast<void*>(grad_output)},
      {graph_struct.db, grad_bias_temp},
  };

  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "conv2d bgrad execute");

  size_t num_elements = stats.out_channels;

  cuda::axpy(grad_bias_temp, grad_bias, num_elements, type_desc.param_dtype, stream);
}

}  // namespace tunx

#endif
