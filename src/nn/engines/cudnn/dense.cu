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

struct dense_fwd_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> w;
  std::shared_ptr<fe::graph::Tensor_attributes> b;
  std::shared_ptr<fe::graph::Tensor_attributes> y;

  size_t workspace_size;

  dense_fwd_graph(cudnnHandle_t handle, const DenseStats& stats, DTypeDesc& type_desc) {
    const int64 batch = static_cast<int64>(1);
    const int64 m = static_cast<int64>(stats.batch_size);
    const int64 n = static_cast<int64>(stats.out_features);
    const int64 k = static_cast<int64>(stats.in_features);

    fe::DataType_t io_type = to_fe_data_type(type_desc.io_dtype);
    fe::DataType_t param_type = to_fe_data_type(type_desc.param_dtype);
    fe::DataType_t compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    x = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("Input")
                          .set_dim({batch, m, k})
                          .set_stride({m * k, k, 1})
                          .set_data_type(io_type));

    w = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("Weight")
                          .set_dim({batch, k, n})
                          .set_stride({0, 1, k})
                          .set_data_type(param_type));

    b = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("Bias")
                          .set_dim({batch, 1, n})
                          .set_stride({0, 0, 1})
                          .set_data_type(param_type));

    auto weight_cast = w;

    {
      auto identity_attributes = fe::graph::Pointwise_attributes()
                                     .set_name("Cast_Weight")
                                     .set_mode(fe::PointwiseMode_t::IDENTITY)
                                     .set_compute_data_type(compute_type);
      weight_cast = graph->pointwise(w, identity_attributes);
      weight_cast->set_data_type(io_type);
    }

    auto matmul_attributes =
        fe::graph::Matmul_attributes().set_name("FWD_GEMM").set_compute_data_type(compute_type);
    y = graph->matmul(x, weight_cast, matmul_attributes);
    if (stats.use_bias) {
      y->set_is_virtual(true);

      auto add_bias_attributes = fe::graph::Pointwise_attributes()
                                     .set_name("Add_Bias")
                                     .set_mode(fe::PointwiseMode_t::ADD)
                                     .set_compute_data_type(compute_type);

      y = graph->pointwise(y, b, add_bias_attributes);
    }

    y->set_output(true).set_data_type(io_type);

    ensure_ok(graph->validate(), "fwd_gemm validate");
    ensure_ok(graph->build_operation_graph(handle), "fwd_gemm build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "fwd_gemm create plans");
    ensure_ok(graph->check_support(), "fwd_gemm check support");
    ensure_ok(graph->build_plans(), "fwd_gemm build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "fwd_gemm workspace");
    assert(ws >= 0);
    workspace_size = ws;
  }
};

struct dense_dgrad_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> dy;
  std::shared_ptr<fe::graph::Tensor_attributes> w;
  std::shared_ptr<fe::graph::Tensor_attributes> dx;

  size_t workspace_size;

  dense_dgrad_graph(cudnnHandle_t handle, const DenseStats& stats, DTypeDesc& type_desc) {
    const int64 batch = static_cast<int64>(1);
    const int64 m = static_cast<int64>(stats.batch_size);
    const int64 n = static_cast<int64>(stats.out_features);
    const int64 k = static_cast<int64>(stats.in_features);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto param_type = to_fe_data_type(type_desc.param_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    dy = graph->tensor(fe::graph::Tensor_attributes()
                           .set_name("Gradient")
                           .set_dim({batch, m, n})
                           .set_stride({m * n, n, 1})
                           .set_data_type(io_type));

    w = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("Weight")
                          .set_dim({batch, n, k})
                          .set_stride({0, k, 1})
                          .set_data_type(param_type));

    auto weight_cast = w;

    {
      auto identity_attributes = fe::graph::Pointwise_attributes()
                                     .set_name("Cast_Weight")
                                     .set_mode(fe::PointwiseMode_t::IDENTITY)
                                     .set_compute_data_type(compute_type);
      weight_cast = graph->pointwise(w, identity_attributes);
      weight_cast->set_data_type(io_type);
    }

    auto matmul_attributes =
        fe::graph::Matmul_attributes().set_name("DGRAD_GEMM").set_compute_data_type(compute_type);
    dx = graph->matmul(dy, weight_cast, matmul_attributes);
    dx->set_output(true).set_data_type(io_type);

    ensure_ok(graph->validate(), "dgrad_gemm validate");
    ensure_ok(graph->build_operation_graph(handle), "dgrad_gemm build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "dgrad_gemm create plans");
    ensure_ok(graph->check_support(), "dgrad_gemm check support");
    ensure_ok(graph->build_plans(), "dgrad_gemm build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "dgrad_gemm workspace");
    assert(ws >= 0);
    workspace_size = ws;
  }
};

struct dense_wgrad_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> dy;
  std::shared_ptr<fe::graph::Tensor_attributes> dw;
  std::shared_ptr<fe::graph::Tensor_attributes> dw_temp;

  size_t workspace_size;

  dense_wgrad_graph(cudnnHandle_t handle, const DenseStats& stats, DTypeDesc& type_desc) {
    const int64 batch = static_cast<int64>(1);
    const int64 m = static_cast<int64>(stats.batch_size);
    const int64 n = static_cast<int64>(stats.out_features);
    const int64 k = static_cast<int64>(stats.in_features);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto param_type = to_fe_data_type(type_desc.param_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    dy = graph->tensor(fe::graph::Tensor_attributes()
                           .set_name("Gradient")
                           .set_dim({batch, n, m})
                           .set_stride({m * n, 1, n})
                           .set_data_type(io_type));

    x = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("Input")
                          .set_dim({batch, m, k})
                          .set_stride({m * k, k, 1})
                          .set_data_type(io_type));

    dw = graph->tensor(fe::graph::Tensor_attributes()
                           .set_name("dw")
                           .set_dim({batch, n, k})
                           .set_stride({n * k, k, 1})
                           .set_data_type(param_type));

    auto matmul_attributes =
        fe::graph::Matmul_attributes().set_name("WGRAD_GEMM").set_compute_data_type(compute_type);
    auto matmul_output = graph->matmul(dy, x, matmul_attributes);
    matmul_output->set_data_type(param_type);

    auto add_attributes = fe::graph::Pointwise_attributes()
                              .set_name("DW_accumulate")
                              .set_mode(fe::PointwiseMode_t::ADD);

    dw_temp = graph->pointwise(dw, matmul_output, add_attributes);
    dw_temp->set_output(true).set_data_type(param_type);

    ensure_ok(graph->validate(), "wgrad_gemm validate");
    ensure_ok(graph->build_operation_graph(handle), "wgrad_gemm build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "wgrad_gemm create plans");
    ensure_ok(graph->check_support(), "wgrad_gemm check support");
    ensure_ok(graph->build_plans(), "wgrad_gemm build plans");

    int64 ws = 0;
    ensure_ok(graph->get_workspace_size(ws), "wgrad_gemm workspace");

    workspace_size = static_cast<size_t>(ws);
  }
};

// bgrad_reduce_accumulate_kernel delegated to cuda_engine_

WorkspaceReq CuDNNEngine::query_dense_graph(engine_handle backend_handle, const DenseStats& stats,
                                            DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();

  GraphCacheKey fwd_key{
      .op_type = OpType::DENSE_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_features, stats.out_features},
      .attributes = {{"use_bias", stats.use_bias}},
  };
  auto it_fwd = graph_cache_.find(fwd_key);
  if (it_fwd == graph_cache_.end()) {
    it_fwd = graph_cache_.emplace(fwd_key, dense_fwd_graph(handle, stats, type_desc)).first;
  }
  GraphCacheKey wgrad_key{
      .op_type = OpType::DENSE_WGRAD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_features, stats.out_features},
      .attributes = {{"use_bias", stats.use_bias}},
  };
  auto it_wgrad = graph_cache_.find(wgrad_key);
  if (it_wgrad == graph_cache_.end()) {
    it_wgrad = graph_cache_.emplace(wgrad_key, dense_wgrad_graph(handle, stats, type_desc)).first;
  }
  GraphCacheKey dgrad_key{
      .op_type = OpType::DENSE_DGRAD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_features, stats.out_features},
      .attributes = {{"use_bias", stats.use_bias}},
  };
  auto it_dgrad = graph_cache_.find(dgrad_key);
  if (it_dgrad == graph_cache_.end()) {
    it_dgrad = graph_cache_.emplace(dgrad_key, dense_dgrad_graph(handle, stats, type_desc)).first;
  }

  auto& fwd_graph = std::any_cast<dense_fwd_graph&>(it_fwd->second);
  auto& dgrad_graph = std::any_cast<dense_dgrad_graph&>(it_dgrad->second);
  auto& wgrad_graph = std::any_cast<dense_wgrad_graph&>(it_wgrad->second);
  size_t fwd_workspace = fwd_graph.workspace_size;
  size_t bwd_workspace = std::max({dgrad_graph.workspace_size, wgrad_graph.workspace_size});
  // TODO: add inf graph
  return {fwd_workspace, bwd_workspace, fwd_workspace};
}

void CuDNNEngine::dense_fwd(engine_handle backend_handle, const DenseStats& stats,
                            const void* input, const void* weight, const void* bias, void* output,
                            void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  GraphCacheKey key{
      .op_type = OpType::DENSE_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_features, stats.out_features},
      .attributes = {{"use_bias", stats.use_bias}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error(
        "cuDNN Graph not found for dense fwd. Please call query_dense_graph first.");
  }
  auto& graph_struct = std::any_cast<dense_fwd_graph&>(it->second);
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.x, const_cast<void*>(input)},
      {graph_struct.w, const_cast<void*>(weight)},
      {graph_struct.b, const_cast<void*>(bias)},
      {graph_struct.y, output},
  };
  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "dense_fwd execute");
}

void CuDNNEngine::dense_wgrad(engine_handle backend_handle, const DenseStats& stats,
                              const void* grad_output, const void* input, void* grad_weight,
                              void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  GraphCacheKey key{
      .op_type = OpType::DENSE_WGRAD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_features, stats.out_features},
      .attributes = {{"use_bias", stats.use_bias}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error(
        "cuDNN Graph not found for dense wgrad. Please call query_dense_graph first.");
  }
  auto& graph_struct = std::any_cast<dense_wgrad_graph&>(it->second);
  // for cudnn gemm, in-place accumulation is possible.
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.dy, const_cast<void*>(grad_output)},
      {graph_struct.x, const_cast<void*>(input)},
      {graph_struct.dw, const_cast<void*>(grad_weight)},
      {graph_struct.dw_temp, grad_weight},
  };
  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "dense_wgrad execute");
}

void CuDNNEngine::dense_dgrad(engine_handle backend_handle, const DenseStats& stats,
                              const void* grad_output, const void* weight, void* grad_input,
                              void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  GraphCacheKey key{
      .op_type = OpType::DENSE_DGRAD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.in_features, stats.out_features},
      .attributes = {{"use_bias", stats.use_bias}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error(
        "cuDNN Graph not found for dense dgrad. Please call query_dense_graph first.");
  }
  auto& graph_struct = std::any_cast<dense_dgrad_graph&>(it->second);
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
      {graph_struct.dy, const_cast<void*>(grad_output)},
      {graph_struct.w, const_cast<void*>(weight)},
      {graph_struct.dx, grad_input}};
  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "dense_dgrad execute");
}

void CuDNNEngine::dense_bgrad(engine_handle backend_handle, const DenseStats& stats,
                              const void* grad_output, void* grad_bias, void* workspace,
                              DTypeDesc type_desc) {
  cuda_engine_.dense_bgrad(backend_handle, stats, grad_output, grad_bias, workspace, type_desc);
}

}  // namespace tunx

#endif
