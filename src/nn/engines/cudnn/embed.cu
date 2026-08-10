#ifdef TUNX_USE_CUDNN
#include <cudnn_frontend.h>
#include <cudnn_graph.h>

#include "nn/engines/cudnn_engine.hpp"
#include "nn/engines/engine_handle.hpp"
#include "nn/stats/stats.hpp"
#include "type/type.hpp"

namespace tunx {

WorkspaceReq CuDNNEngine::query_class_token_graph(engine_handle backend_handle,
                                                  const ClassTokenStats& stats,
                                                  DTypeDesc type_desc) {
  return cuda_engine_.query_class_token_graph(backend_handle, stats, type_desc);
}

WorkspaceReq CuDNNEngine::query_embedding_graph(engine_handle backend_handle,
                                                const EmbeddingStats& stats, DTypeDesc type_desc) {
  return cuda_engine_.query_embedding_graph(backend_handle, stats, type_desc);
}

WorkspaceReq CuDNNEngine::query_positional_embedding_graph(engine_handle backend_handle,
                                                           const PositionalEmbeddingStats& stats,
                                                           DTypeDesc type_desc) {
  return cuda_engine_.query_positional_embedding_graph(backend_handle, stats, type_desc);
}

void CuDNNEngine::class_token_fwd(engine_handle backend_handle, const ClassTokenStats& stats,
                                  const void* input, const void* token, void* output,
                                  void* workspace, DTypeDesc type_desc) {
  cuda_engine_.class_token_fwd(backend_handle, stats, input, token, output, workspace, type_desc);
}

void CuDNNEngine::class_token_bwd(engine_handle backend_handle, const ClassTokenStats& stats,
                                  const void* grad_output, void* grad_input, void* grad_token,
                                  void* workspace, DTypeDesc type_desc) {
  cuda_engine_.class_token_bwd(backend_handle, stats, grad_output, grad_input, grad_token, workspace,
                               type_desc);
}

void CuDNNEngine::embedding_fwd(engine_handle backend_handle, const EmbeddingStats& stats,
                                const void* input, const void* weight, void* output,
                                void* workspace, DTypeDesc type_desc) {
  cuda_engine_.embedding_fwd(backend_handle, stats, input, weight, output, workspace, type_desc);
}

void CuDNNEngine::embedding_bwd(engine_handle backend_handle, const EmbeddingStats& stats,
                                const void* grad_output, const void* input, void* grad_weight,
                                void* workspace, DTypeDesc type_desc) {
  cuda_engine_.embedding_bwd(backend_handle, stats, grad_output, input, grad_weight, workspace,
                             type_desc);
}

void CuDNNEngine::positional_embedding_fwd(engine_handle backend_handle,
                                           const PositionalEmbeddingStats& stats, const void* input,
                                           const void* pos_embedding, void* output, void* workspace,
                                           DTypeDesc type_desc) {
  cuda_engine_.positional_embedding_fwd(backend_handle, stats, input, pos_embedding, output,
                                        workspace, type_desc);
}

void CuDNNEngine::positional_embedding_bwd(engine_handle backend_handle,
                                           const PositionalEmbeddingStats& stats,
                                           const void* grad_output, void* grad_pos_embedding,
                                           void* workspace, DTypeDesc type_desc) {
  cuda_engine_.positional_embedding_bwd(backend_handle, stats, grad_output, grad_pos_embedding,
                                        workspace, type_desc);
}

}  // namespace tunx

#endif
