#include "internal.hpp"
#include "nn/engines/cpu_engine.hpp"
#include "threading/thread_handler.hpp"

namespace tunx {
namespace {

// Helpers for transpose
template <typename T>
void transpose_impl(const T* input, T* output, const size_t* shape, size_t ndim, size_t dim0,
                    size_t dim1) {
  size_t total_elements = 1;
  size_t strides[8];
  for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
    strides[i] = total_elements;
    total_elements *= shape[i];
  }

  size_t out_shape[8];
  for (size_t i = 0; i < ndim; ++i) out_shape[i] = shape[i];
  std::swap(out_shape[dim0], out_shape[dim1]);

  size_t out_strides[8];
  size_t out_total = 1;
  for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
    out_strides[i] = out_total;
    out_total *= out_shape[i];
  }

  parallel_for<size_t>(0, total_elements, [&](size_t idx) {
    size_t in_idx = idx;
    size_t out_idx = 0;
    size_t coords[8];
    for (size_t i = 0; i < ndim; ++i) {
      coords[i] = in_idx / strides[i];
      in_idx %= strides[i];
    }
    std::swap(coords[dim0], coords[dim1]);
    for (size_t i = 0; i < ndim; ++i) {
      out_idx += coords[i] * out_strides[i];
    }
    output[out_idx] = input[idx];
  });
}

}  // namespace

void CPUEngine::sdpa_fwd(engine_handle backend_handle, const AttentionStats& stats,
                         const void* q_data, const void* k_data, const void* v_data, void* o_data,
                         void* stats_data, void* workspace, DTypeDesc type_desc) {
  throw std::runtime_error("SDPA forward is not yet implemented for CPUEngine");
}

void CPUEngine::sdpa_bwd(engine_handle backend_handle, const AttentionStats& stats,
                         const void* q_data, const void* k_data, const void* v_data,
                         const void* o_data, const void* dO_data, const void* stats_data,
                         void* dQ_data, void* dK_data, void* dV_data, void* workspace,
                         DTypeDesc type_desc) {
  throw std::runtime_error("SDPA backward is not yet implemented for CPUEngine");
}

void CPUEngine::transpose(engine_handle backend_handle, const TransposeStats& stats,
                          const void* input, void* output, void* workspace, DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    transpose_impl<T>(static_cast<const T*>(input), static_cast<T*>(output), stats.shape,
                      stats.ndim, stats.dim0, stats.dim1);
  });
}

WorkspaceReq CPUEngine::query_sdpa_graph(engine_handle backend_handle, const AttentionStats& stats,
                                         DTypeDesc type_desc) {
  return WorkspaceReq{0, 0, 0};
}

WorkspaceReq CPUEngine::query_transpose_graph(engine_handle backend_handle,
                                              const TransposeStats& stats, DTypeDesc type_desc) {
  return WorkspaceReq{0, 0, 0};
}

}  // namespace tunx
