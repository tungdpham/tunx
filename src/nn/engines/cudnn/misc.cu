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
#include "nn/engines/cudnn_engine.hpp"
#include "nn/engines/engine_handle.hpp"
#include "nn/stats/stats.hpp"
#include "type/type.hpp"

namespace tunx {

struct sdpa_fwd_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> Q, K, V, O, Stats;

  sdpa_fwd_graph(cudnnHandle_t handle, const AttentionStats& stats, DTypeDesc type_desc) {
    const int64_t b = static_cast<int64_t>(stats.batch_size);
    const int64_t h = static_cast<int64_t>(stats.num_heads);
    const int64_t s = static_cast<int64_t>(stats.seq_len);
    const int64_t d = static_cast<int64_t>(stats.head_dim);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    Q = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("Q")
                          .set_uid(1)
                          .set_dim({b, h, s, d})
                          .set_stride({h * s * d, s * d, d, 1}));

    K = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("K")
                          .set_uid(2)
                          .set_dim({b, h, s, d})
                          .set_stride({h * s * d, s * d, d, 1}));

    V = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("V")
                          .set_uid(3)
                          .set_dim({b, h, s, d})
                          .set_stride({h * s * d, s * d, d, 1}));

    auto sdpa_options = fe::graph::SDPA_attributes()
                            .set_name("flash_attention")
                            .set_attn_scale(stats.attn_scale)
                            .set_generate_stats(true);

    if (stats.is_causal) {
      sdpa_options.set_diagonal_alignment(cudnn_frontend::DiagonalAlignment_t::TOP_LEFT)
          .set_diagonal_band_right_bound(0);
    }

    auto [out_O, out_Stats] = graph->sdpa(Q, K, V, sdpa_options);
    O = out_O;
    Stats = out_Stats;

    O->set_output(true).set_dim({b, h, s, d}).set_stride({h * s * d, s * d, d, 1}).set_uid(4);
    Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_uid(5);

    ensure_ok(graph->validate(), "sdpa validate");
    ensure_ok(graph->build_operation_graph(handle), "sdpa build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "sdpa create plans");
    ensure_ok(graph->check_support(), "sdpa check support");
    ensure_ok(graph->build_plans(), "sdpa build plans");
  }
};

struct sdpa_bwd_graph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> Q, K, V, O, dO, Stats, dQ, dK, dV;

  sdpa_bwd_graph(cudnnHandle_t handle, const AttentionStats& stats, DTypeDesc type_desc) {
    const int64_t b = static_cast<int64_t>(stats.batch_size);
    const int64_t h = static_cast<int64_t>(stats.num_heads);
    const int64_t s = static_cast<int64_t>(stats.seq_len);
    const int64_t d = static_cast<int64_t>(stats.head_dim);

    auto io_type = to_fe_data_type(type_desc.io_dtype);
    auto compute_type = to_fe_compute_type(type_desc.compute_dtype);

    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(io_type)
        .set_intermediate_data_type(compute_type)
        .set_compute_data_type(compute_type);

    Q = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("Q")
                          .set_uid(1)
                          .set_dim({b, h, s, d})
                          .set_stride({h * s * d, s * d, d, 1}));

    K = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("K")
                          .set_uid(2)
                          .set_dim({b, h, s, d})
                          .set_stride({h * s * d, s * d, d, 1}));

    V = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("V")
                          .set_uid(3)
                          .set_dim({b, h, s, d})
                          .set_stride({h * s * d, s * d, d, 1}));

    O = graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("O")
                          .set_uid(4)
                          .set_dim({b, h, s, d})
                          .set_stride({h * s * d, s * d, d, 1}));

    dO = graph->tensor(fe::graph::Tensor_attributes()
                           .set_name("dO")
                           .set_uid(5)
                           .set_dim({b, h, s, d})
                           .set_stride({h * s * d, s * d, d, 1}));

    Stats = graph->tensor(fe::graph::Tensor_attributes()
                              .set_name("Stats")
                              .set_uid(6)
                              .set_dim({b, h, s, 1})
                              .set_stride({h * s, s, 1, 1})
                              .set_data_type(fe::DataType_t::FLOAT));

    auto sdpa_options = fe::graph::SDPA_backward_attributes()
                            .set_name("flash_attention_backward")
                            .set_attn_scale(stats.attn_scale);

    if (stats.is_causal) {
      sdpa_options.set_diagonal_alignment(cudnn_frontend::DiagonalAlignment_t::TOP_LEFT)
          .set_diagonal_band_right_bound(0);
    }

    auto [out_dQ, out_dK, out_dV] = graph->sdpa_backward(Q, K, V, O, dO, Stats, sdpa_options);
    dQ = out_dQ;
    dK = out_dK;
    dV = out_dV;

    dQ->set_output(true).set_uid(7).set_dim({b, h, s, d}).set_stride({h * s * d, s * d, d, 1});
    dK->set_output(true).set_uid(8).set_dim({b, h, s, d}).set_stride({h * s * d, s * d, d, 1});
    dV->set_output(true).set_uid(9).set_dim({b, h, s, d}).set_stride({h * s * d, s * d, d, 1});

    ensure_ok(graph->validate(), "sdpa_backward validate");
    ensure_ok(graph->build_operation_graph(handle), "sdpa_backward build op graph");
    ensure_ok(graph->create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::B}),
              "sdpa_backward create plans");
    ensure_ok(graph->check_support(), "sdpa_backward check support");
    ensure_ok(graph->build_plans(), "sdpa_backward build plans");
  }
};

template <typename T>
__global__ void dropout_fwd_kernel(const T* __restrict__ input, T* __restrict__ output,
                                   bool* __restrict__ mask, float dropout_rate, float scale,
                                   unsigned long long seed, size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  unsigned int state = (unsigned int)(seed ^ idx ^ 0x12345678);
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  float rand_val = (state % 100000) / 100000.0f;

  bool keep = rand_val > dropout_rate;
  mask[idx] = keep;

  if (keep) {
    output[idx] = (T)((float)input[idx] * scale);
  } else {
    output[idx] = (T)0.0f;
  }
}

template <typename T>
__global__ void dropout_bwd_kernel(const T* __restrict__ grad_output, T* __restrict__ grad_input,
                                   const bool* __restrict__ mask, float scale,
                                   size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  if (mask[idx]) {
    grad_input[idx] = (T)((float)grad_output[idx] * scale);
  } else {
    grad_input[idx] = (T)0.0f;
  }
}

WorkspaceReq CuDNNEngine::query_dropout_graph(engine_handle backend_handle,
                                              const DropoutStats& stats, DTypeDesc type_desc) {
  return {0, 0, 0};
}

WorkspaceReq CuDNNEngine::query_sdpa_graph(engine_handle backend_handle,
                                           const AttentionStats& stats, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();

  GraphCacheKey fwd_key{
      .op_type = OpType::SDPA_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.num_heads, stats.seq_len, stats.head_dim},
      .attributes = {{"is_causal", stats.is_causal}, {"attn_scale", stats.attn_scale}},
  };
  auto it_fwd = graph_cache_.find(fwd_key);
  if (it_fwd == graph_cache_.end()) {
    it_fwd = graph_cache_.emplace(fwd_key, sdpa_fwd_graph(handle, stats, type_desc)).first;
  }

  GraphCacheKey bwd_key{
      .op_type = OpType::SDPA_BWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.num_heads, stats.seq_len, stats.head_dim},
      .attributes = {{"is_causal", stats.is_causal}, {"attn_scale", stats.attn_scale}},
  };
  auto it_bwd = graph_cache_.find(bwd_key);
  if (it_bwd == graph_cache_.end()) {
    it_bwd = graph_cache_.emplace(bwd_key, sdpa_bwd_graph(handle, stats, type_desc)).first;
  }

  auto& fwd_graph = std::any_cast<sdpa_fwd_graph&>(it_fwd->second);
  auto& bwd_graph = std::any_cast<sdpa_bwd_graph&>(it_bwd->second);

  int64_t fwd_workspace_size = 0;
  int64_t bwd_workspace_size = 0;
  ensure_ok(fwd_graph.graph->get_workspace_size(fwd_workspace_size), "sdpa fwd workspace");
  ensure_ok(bwd_graph.graph->get_workspace_size(bwd_workspace_size), "sdpa bwd workspace");

  return WorkspaceReq{static_cast<size_t>(fwd_workspace_size),
                      static_cast<size_t>(bwd_workspace_size),
                      static_cast<size_t>(fwd_workspace_size)};
}

void CuDNNEngine::sdpa_fwd(engine_handle backend_handle, const AttentionStats& stats,
                           const void* q_data, const void* k_data, const void* v_data, void* o_data,
                           void* stats_data, void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();

  GraphCacheKey key{
      .op_type = OpType::SDPA_FWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.num_heads, stats.seq_len, stats.head_dim},
      .attributes = {{"is_causal", stats.is_causal}, {"attn_scale", stats.attn_scale}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error(
        "cuDNN Graph not found for sdpa fwd. Please call query_sdpa_graph first.");
  }
  auto& graph_struct = std::any_cast<sdpa_fwd_graph&>(it->second);

  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
      {1, const_cast<void*>(q_data)},
      {2, const_cast<void*>(k_data)},
      {3, const_cast<void*>(v_data)},
      {4, o_data},
      {5, stats_data}};

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "sdpa_fwd execute");
}

void CuDNNEngine::sdpa_bwd(engine_handle backend_handle, const AttentionStats& stats,
                           const void* q_data, const void* k_data, const void* v_data,
                           const void* o_data, const void* dO_data, const void* stats_data,
                           void* dQ_data, void* dK_data, void* dV_data, void* workspace,
                           DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();

  GraphCacheKey key{
      .op_type = OpType::SDPA_BWD,
      .dtype_desc = type_desc,
      .dims = {stats.batch_size, stats.num_heads, stats.seq_len, stats.head_dim},
      .attributes = {{"is_causal", stats.is_causal}, {"attn_scale", stats.attn_scale}},
  };
  auto it = graph_cache_.find(key);
  if (it == graph_cache_.end()) {
    throw std::runtime_error(
        "cuDNN Graph not found for sdpa bwd. Please call query_sdpa_graph first.");
  }
  auto& graph_struct = std::any_cast<sdpa_bwd_graph&>(it->second);

  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
      {1, const_cast<void*>(q_data)},
      {2, const_cast<void*>(k_data)},
      {3, const_cast<void*>(v_data)},
      {4, const_cast<void*>(o_data)},
      {5, const_cast<void*>(dO_data)},
      {6, const_cast<void*>(stats_data)},
      {7, dQ_data},
      {8, dK_data},
      {9, dV_data}};

  auto status = graph_struct.graph->execute(handle, variant_pack, workspace);
  ensure_ok(status, "sdpa_bwd execute");
}

void CuDNNEngine::dropout_fwd(engine_handle backend_handle, const DropoutStats& stats,
                              const void* input, void* output, bool* mask, void* workspace,
                              DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  size_t total_elements = stats.batch_size * stats.channels * stats.spatial_size;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  float scale = 1.0f / (1.0f - static_cast<float>(stats.dropout_rate));

  DISPATCH_DTYPE(type_desc.io_dtype, T, {
    dropout_fwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<T*>(output), mask,
        static_cast<float>(stats.dropout_rate), scale, 0x12345678ULL, total_elements);
  });
}

void CuDNNEngine::dropout_bwd(engine_handle backend_handle, const DropoutStats& stats,
                              const void* grad_output, void* grad_input, const bool* mask,
                              double scale, void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  size_t total_elements = stats.batch_size * stats.channels * stats.spatial_size;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  DISPATCH_DTYPE(type_desc.io_dtype, T, {
    dropout_bwd_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(grad_output), static_cast<T*>(grad_input), mask,
        static_cast<float>(scale), total_elements);
  });
}

WorkspaceReq CuDNNEngine::query_transpose_graph(engine_handle backend_handle,
                                                const TransposeStats& stats, DTypeDesc type_desc) {
  return WorkspaceReq{0, 0, 0};
}

namespace {

struct CuDNNTransposeParams {
  size_t ndim;
  size_t dim0;
  size_t dim1;
  size_t shape[8];
  size_t strides[8];
  size_t out_strides[8];
};

template <typename T>
__global__ void cudnn_transpose_kernel(const T* input, T* output, CuDNNTransposeParams p,
                                       size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  size_t in_idx = idx;
  size_t out_idx = 0;
  size_t coords[8];

  for (size_t i = 0; i < p.ndim; ++i) {
    coords[i] = in_idx / p.strides[i];
    in_idx %= p.strides[i];
  }

  size_t temp = coords[p.dim0];
  coords[p.dim0] = coords[p.dim1];
  coords[p.dim1] = temp;

  for (size_t i = 0; i < p.ndim; ++i) {
    out_idx += coords[i] * p.out_strides[i];
  }

  output[out_idx] = input[idx];
}
}  // namespace

void CuDNNEngine::transpose(engine_handle backend_handle, const TransposeStats& stats,
                            const void* input, void* output, void* workspace, DTypeDesc type_desc) {
  cudnnHandle_t handle = backend_handle.as<CuDNNEngineHandle>()->handle();
  cudaStream_t stream;
  cudnnGetStream(handle, &stream);

  CuDNNTransposeParams p;
  p.ndim = stats.ndim;
  p.dim0 = stats.dim0;
  p.dim1 = stats.dim1;

  size_t total_elements = 1;
  for (int i = static_cast<int>(p.ndim) - 1; i >= 0; --i) {
    p.shape[i] = stats.shape[i];
    p.strides[i] = total_elements;
    total_elements *= p.shape[i];
  }

  size_t out_shape[8] = {0};
  for (size_t i = 0; i < p.ndim; ++i) out_shape[i] = p.shape[i];
  std::swap(out_shape[p.dim0], out_shape[p.dim1]);

  size_t out_total = 1;
  for (int i = static_cast<int>(p.ndim) - 1; i >= 0; --i) {
    p.out_strides[i] = out_total;
    out_total *= out_shape[i];
  }

  if (total_elements == 0) return;

  size_t threads = 256;
  size_t blocks = (total_elements + threads - 1) / threads;

  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    cudnn_transpose_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<T*>(output), p, total_elements);
  });
}

}  // namespace tunx

#endif
