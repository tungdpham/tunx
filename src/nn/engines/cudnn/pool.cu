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
#include "nn/engines/cudnn_engine.hpp"
#include "nn/engines/engine_handle.hpp"
#include "nn/stats/stats.hpp"
#include "type/type.hpp"

namespace tunx {

struct avgpool2d_fwd_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> y;

  size_t workspace_size;

  avgpool2d_fwd_graph(cudnnHandle_t handle, const AvgPool2DStats& stats, DTypeDesc& type_desc) {
    const int64 n = static_cast<int64>(stats.batch_size);
    const int64 c = static_cast<int64>(stats.channels);
    const int64 h = static_cast<int64>(stats.height);
    const int64 w = static_cast<int64>(stats.width);
    const int64 output_h = (h + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
    const int64 output_w = (w + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;

    auto io_type = to_fe_data_type(type_desc.io_dtype);
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

    auto resample_options =
        fe::graph::Resample_attributes()
            .set_name("AvgPool")
            .set_resampling_mode(fe::ResampleMode_t::AVGPOOL_EXCLUDE_PADDING)
            .set_padding_mode(fe::PaddingMode_t::ZERO_PAD)
            .set_window({static_cast<int64>(stats.pool_h), static_cast<int64>(stats.pool_w)})
            .set_pre_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_post_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_stride({static_cast<int64>(stats.stride_h), static_cast<int64>(stats.stride_w)})
            .set_compute_data_type(compute_type)
            .set_generate_index(false);

    auto outputs = graph->resample(x, resample_options);
    y = outputs[0];
    y->set_output(true)
        .set_dim({n, c, output_h, output_w})
        .set_stride({output_h * output_w * c, 1, output_w * c, c})
        .set_data_type(io_type);

    ensure_ok(graph->validate(), "avgpool2d_fwd validate");
    ensure_ok(graph->build_operation_graph(handle), "avgpool2d_fwd build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "avgpool2d_fwd create plans");
    ensure_ok(graph->check_support(), "avgpool2d_fwd check support");
    ensure_ok(graph->build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE, false),
              "avgpool2d_fwd build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "avgpool2d_fwd workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

// avgpool_bwd delegated to cuda_engine_

struct maxpool2d_inf_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> y;
  std::shared_ptr<fe::graph::Tensor_attributes> mask;

  size_t workspace_size;
  size_t mask_size;

  maxpool2d_inf_graph(cudnnHandle_t handle, const MaxPool2DStats& stats, DTypeDesc& type_desc) {
    const int64 n = static_cast<int64>(stats.batch_size);
    const int64 c = static_cast<int64>(stats.channels);
    const int64 h = static_cast<int64>(stats.height);
    const int64 w = static_cast<int64>(stats.width);
    const int64 output_h = (h + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
    const int64 output_w = (w + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;

    auto io_type = to_fe_data_type(type_desc.io_dtype);
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

    auto resample_options =
        fe::graph::Resample_attributes()
            .set_name("MaxPoolInfer")
            .set_resampling_mode(fe::ResampleMode_t::MAXPOOL)
            .set_padding_mode(fe::PaddingMode_t::NEG_INF_PAD)
            .set_window({static_cast<int64>(stats.pool_h), static_cast<int64>(stats.pool_w)})
            .set_pre_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_post_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_stride({static_cast<int64>(stats.stride_h), static_cast<int64>(stats.stride_w)})
            .set_compute_data_type(compute_type)
            .set_generate_index(true);

    auto outputs = graph->resample(x, resample_options);
    y = outputs[0];
    mask = outputs[1];
    y->set_output(true)
        .set_dim({n, c, output_h, output_w})
        .set_stride({output_h * output_w * c, 1, output_w * c, c})
        .set_data_type(io_type);
    mask->set_output(true)
        .set_dim({n, c, output_h, output_w})
        .set_stride({output_h * output_w * c, 1, output_w * c, c})
        .set_data_type(fe::DataType_t::INT32);

    ensure_ok(graph->validate(), "maxpool2d_inf validate");
    ensure_ok(graph->build_operation_graph(handle), "maxpool2d_inf build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "maxpool2d_inf create plans");
    ensure_ok(graph->check_support(), "maxpool2d_inf check support");
    ensure_ok(graph->build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE, false),
              "maxpool2d_inf build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "maxpool2d_inf workspace");
    size_t base_ws = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
    mask_size = n * c * output_h * output_w * sizeof(int32_t);
    mask_size = (mask_size + 255) & ~static_cast<size_t>(255);
    workspace_size = base_ws + mask_size;
  }
};

struct maxpool2d_fwd_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> y;
  std::shared_ptr<fe::graph::Tensor_attributes> mask;

  size_t workspace_size;

  maxpool2d_fwd_graph(cudnnHandle_t handle, const MaxPool2DStats& stats, DTypeDesc& type_desc) {
    const int64 n = static_cast<int64>(stats.batch_size);
    const int64 c = static_cast<int64>(stats.channels);
    const int64 h = static_cast<int64>(stats.height);
    const int64 w = static_cast<int64>(stats.width);
    const int64 output_h = (h + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
    const int64 output_w = (w + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;

    auto io_type = to_fe_data_type(type_desc.io_dtype);
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

    auto resample_options =
        fe::graph::Resample_attributes()
            .set_name("MaxPool")
            .set_resampling_mode(fe::ResampleMode_t::MAXPOOL)
            .set_padding_mode(fe::PaddingMode_t::NEG_INF_PAD)
            .set_window({static_cast<int64>(stats.pool_h), static_cast<int64>(stats.pool_w)})
            .set_pre_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_post_padding({static_cast<int64>(stats.pad_h), static_cast<int64>(stats.pad_w)})
            .set_stride({static_cast<int64>(stats.stride_h), static_cast<int64>(stats.stride_w)})
            .set_compute_data_type(compute_type)
            .set_generate_index(true);

    auto outputs = graph->resample(x, resample_options);
    y = outputs[0];
    mask = outputs[1];
    y->set_output(true)
        .set_dim({n, c, output_h, output_w})
        .set_stride({output_h * output_w * c, 1, output_w * c, c})
        .set_data_type(io_type);
    mask->set_output(true)
        .set_dim({n, c, output_h, output_w})
        .set_stride({output_h * output_w * c, 1, output_w * c, c})
        .set_data_type(fe::DataType_t::INT32);

    ensure_ok(graph->validate(), "maxpool2d_fwd validate");
    ensure_ok(graph->build_operation_graph(handle), "maxpool2d_fwd build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "maxpool2d_fwd create plans");
    ensure_ok(graph->check_support(), "maxpool2d_fwd check support");
    ensure_ok(graph->build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE, false),
              "maxpool2d_fwd build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "maxpool2d_fwd workspace");
    workspace_size = (static_cast<size_t>(ws) + 255) & ~static_cast<size_t>(255);
  }
};

// maxpool2d_bwd delegated to cuda_engine_

WorkspaceReq CuDNNEngine::query_avgpool_graph(engine_handle backend_handle,
                                              const AvgPool2DStats& stats, DTypeDesc type_desc) {
  return cuda_engine_.query_avgpool_graph(backend_handle, stats, type_desc);
}

WorkspaceReq CuDNNEngine::query_maxpool2d_graph(engine_handle backend_handle,
                                                const MaxPool2DStats& stats, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();

  GraphCacheKey fwd_key{
      .op_type = OpType::MAXPOOL2D_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels, stats.height, stats.width},
      .attributes = {{"pool_h", stats.pool_h},
                     {"pool_w", stats.pool_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w}},
  };
  auto it_fwd = graph_cache_.find(fwd_key);
  if (it_fwd == graph_cache_.end()) {
    it_fwd = graph_cache_.emplace(fwd_key, maxpool2d_fwd_graph(handle, stats, type_desc)).first;
  }

  GraphCacheKey inf_key{
      .op_type = OpType::MAXPOOL2D_INFER,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels, stats.height, stats.width},
      .attributes = {{"pool_h", stats.pool_h},
                     {"pool_w", stats.pool_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w}},
  };
  auto it_inf = graph_cache_.find(inf_key);
  if (it_inf == graph_cache_.end()) {
    it_inf = graph_cache_.emplace(inf_key, maxpool2d_inf_graph(handle, stats, type_desc)).first;
  }

  auto& fwd_graph = std::any_cast<maxpool2d_fwd_graph&>(it_fwd->second);
  auto& inf_graph = std::any_cast<maxpool2d_inf_graph&>(it_inf->second);

  return {fwd_graph.workspace_size, 0, inf_graph.workspace_size};
}

void CuDNNEngine::avgpool_fwd(engine_handle backend_handle, const AvgPool2DStats& stats,
                              const void* input, void* output, void* workspace,
                              DTypeDesc type_desc) {
  cuda_engine_.avgpool_fwd(backend_handle, stats, input, output, workspace, type_desc);
}

void CuDNNEngine::avgpool_bwd(engine_handle backend_handle, const AvgPool2DStats& stats,
                              const void* grad_output, void* grad_input, void* workspace,
                              DTypeDesc type_desc) {
  cuda_engine_.avgpool_bwd(backend_handle, stats, grad_output, grad_input, workspace, type_desc);
}

void CuDNNEngine::maxpool2d_fwd(engine_handle backend_handle, const MaxPool2DStats& stats,
                                const void* input, void* output, void* mask, void* workspace,
                                DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  GraphCacheKey key{
      .op_type = OpType::MAXPOOL2D_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels, stats.height, stats.width},
      .attributes = {{"pool_h", stats.pool_h},
                     {"pool_w", stats.pool_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error("cuDNN Graph not found for maxpool2d fwd.");
  }
  auto& graph_struct = std::any_cast<maxpool2d_fwd_graph&>(it->second);
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.x, const_cast<void*>(input)},
      {graph_struct.y, output},
      {graph_struct.mask, mask},
  };
  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "avgpool fwd execute");
}

void CuDNNEngine::maxpool2d_infer(engine_handle backend_handle, const MaxPool2DStats& stats,
                                  const void* input, void* output, void* workspace,
                                  DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  GraphCacheKey key{
      .op_type = OpType::MAXPOOL2D_INFER,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.channels, stats.height, stats.width},
      .attributes = {{"pool_h", stats.pool_h},
                     {"pool_w", stats.pool_w},
                     {"stride_h", stats.stride_h},
                     {"stride_w", stats.stride_w},
                     {"pad_h", stats.pad_h},
                     {"pad_w", stats.pad_w}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error(
        "cuDNN Graph not found for maxpool2d infer. Please call query_maxpool2d_graph first.");
  }
  auto& graph_struct = std::any_cast<maxpool2d_inf_graph&>(it->second);
  
  void* mask_ptr = nullptr;
  if (graph_struct.mask_size > 0 && workspace != nullptr) {
    size_t base_ws = graph_struct.workspace_size - graph_struct.mask_size;
    mask_ptr = static_cast<char*>(workspace) + base_ws;
  }

  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.x, const_cast<void*>(input)},
      {graph_struct.y, output},
      {graph_struct.mask, mask_ptr},
  };
  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "maxpool2d infer execute");
}

void CuDNNEngine::maxpool2d_bwd(engine_handle backend_handle, const MaxPool2DStats& stats,
                                const void* grad_output, void* grad_input, const void* mask,
                                void* workspace, DTypeDesc type_desc) {
  cuda_engine_.maxpool2d_bwd(backend_handle, stats, grad_output, grad_input, mask, workspace, type_desc);
}

}  // namespace tunx

#endif
