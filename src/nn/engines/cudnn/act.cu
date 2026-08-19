#ifdef TUNX_USE_CUDNN
#include <cudnn_frontend.h>
#include <cudnn_graph.h>

#include "nn/engines/cudnn_engine.hpp"
#include "nn/engines/engine_handle.hpp"
#include "nn/stats/stats.hpp"

namespace tunx {

WorkspaceReq CuDNNEngine::query_relu_graph(engine_handle backend_handle, const ReLUStats& stats,
                                           DTypeDesc type_desc) {
  return cuda_engine_.query_relu_graph(backend_handle, stats, type_desc);
}

void CuDNNEngine::relu_fwd(engine_handle backend_handle, const ReLUStats& stats, const void* input,
                           void* output, void* workspace, DTypeDesc type_desc) {
  cuda_engine_.relu_fwd(backend_handle, stats, input, output, workspace, type_desc);
}

void CuDNNEngine::relu_inf(engine_handle backend_handle, const ReLUStats& stats, const void* input,
                           void* output, void* workspace, DTypeDesc type_desc) {
  cuda_engine_.relu_inf(backend_handle, stats, input, output, workspace, type_desc);
}

void CuDNNEngine::relu_bwd(engine_handle backend_handle, const ReLUStats& stats,
                           const void* grad_output, void* grad_input, const void* output,
                           void* workspace, DTypeDesc type_desc) {
  cuda_engine_.relu_bwd(backend_handle, stats, grad_output, grad_input, output, workspace,
                        type_desc);
}

}  // namespace tunx

#endif
