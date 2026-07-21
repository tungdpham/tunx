#pragma once

#include "device/stream.hpp"
#include "nn/engines/engine_handle.hpp"
#ifdef TUNX_USE_CUDA
#include "nn/engines/iengine.hpp"

namespace tunx {

class CUDAEngineHandle : public IEngineHandle {
public:
  CUDAEngineHandle(stream s)
      : stream_(s) {}

  ~CUDAEngineHandle() override = default;

  stream get_stream() override { return stream_; }

private:
  stream stream_;
};

class CUDAEngine : public IEngine {
public:
  CUDAEngine() = default;
  ~CUDAEngine() override = default;

  engine_handle create_handle(stream s) override;

  WorkspaceReq query_dense_graph(engine_handle backend_handle, const DenseStats& stats,
                                 DTypeDesc type_desc) override;
  WorkspaceReq query_avgpool_graph(engine_handle backend_handle, const AvgPool2DStats& stats,
                                   DTypeDesc type_desc) override;
  WorkspaceReq query_maxpool2d_graph(engine_handle backend_handle, const MaxPool2DStats& stats,
                                     DTypeDesc type_desc) override;
  WorkspaceReq query_class_token_graph(engine_handle backend_handle, const ClassTokenStats& stats,
                                       DTypeDesc type_desc) override;
  WorkspaceReq query_dropout_graph(engine_handle backend_handle, const DropoutStats& stats,
                                   DTypeDesc type_desc) override;
  WorkspaceReq query_embedding_graph(engine_handle backend_handle, const EmbeddingStats& stats,
                                     DTypeDesc type_desc) override;
  WorkspaceReq query_positional_embedding_graph(engine_handle backend_handle,
                                                const PositionalEmbeddingStats& stats,
                                                DTypeDesc type_desc) override;
  WorkspaceReq query_relu_graph(engine_handle backend_handle, const ReLUStats& stats,
                                DTypeDesc type_desc) override;
  WorkspaceReq query_batchnorm_graph(engine_handle backend_handle, const BatchNormStats& stats,
                                     DTypeDesc type_desc) override;
  WorkspaceReq query_conv2d_graph(engine_handle backend_handle, const Conv2DStats& stats,
                                  DTypeDesc type_desc) override;
  WorkspaceReq query_layernorm_graph(engine_handle backend_handle, const LayerNormStats& stats,
                                     DTypeDesc type_desc) override;
  WorkspaceReq query_sdpa_graph(engine_handle backend_handle, const AttentionStats& stats,
                                DTypeDesc type_desc) override;
  WorkspaceReq query_transpose_graph(engine_handle backend_handle, const TransposeStats& stats,
                                     DTypeDesc type_desc) override;

  void dense_fwd(engine_handle backend_handle, const DenseStats& stats, const void* input,
                 const void* weight, const void* bias, void* output, void* workspace,
                 DTypeDesc type_desc) override;

  void dense_wgrad(engine_handle backend_handle, const DenseStats& stats, const void* grad_output,
                   const void* input, void* grad_weight, void* workspace,
                   DTypeDesc type_desc) override;

  void dense_dgrad(engine_handle backend_handle, const DenseStats& stats, const void* grad_output,
                   const void* weight, void* grad_input, void* workspace,
                   DTypeDesc type_desc) override;

  void dense_bgrad(engine_handle backend_handle, const DenseStats& stats, const void* grad_output,
                   void* grad_bias, void* workspace, DTypeDesc type_desc) override;

  void avgpool_fwd(engine_handle backend_handle, const AvgPool2DStats& stats, const void* input,
                   void* output, void* workspace, DTypeDesc type_desc) override;

  void avgpool_bwd(engine_handle backend_handle, const AvgPool2DStats& stats,
                   const void* grad_output, void* grad_input, void* workspace,
                   DTypeDesc type_desc) override;

  void maxpool2d_fwd(engine_handle backend_handle, const MaxPool2DStats& stats, const void* input,
                     void* output, void* mask, void* workspace, DTypeDesc type_desc) override;

  void maxpool2d_infer(engine_handle backend_handle, const MaxPool2DStats& stats, const void* input,
                       void* output, void* workspace, DTypeDesc type_desc) override;

  void maxpool2d_bwd(engine_handle backend_handle, const MaxPool2DStats& stats,
                     const void* grad_output, void* grad_input, const void* mask, void* workspace,
                     DTypeDesc type_desc) override;

  void class_token_fwd(engine_handle backend_handle, const ClassTokenStats& stats,
                       const void* input, const void* token, void* output, void* workspace,
                       DTypeDesc type_desc) override;

  void class_token_bwd(engine_handle backend_handle, const ClassTokenStats& stats,
                       const void* grad_output, void* grad_input, void* grad_token, void* workspace,
                       DTypeDesc type_desc) override;

  void dropout_fwd(engine_handle backend_handle, const DropoutStats& stats, const void* input,
                   void* output, bool* mask, void* workspace, DTypeDesc type_desc) override;

  void dropout_bwd(engine_handle backend_handle, const DropoutStats& stats, const void* grad_output,
                   void* grad_input, const bool* mask, double scale, void* workspace,
                   DTypeDesc type_desc) override;

  void relu_fwd(engine_handle backend_handle, const ReLUStats& stats, const void* input,
                void* output, bool* mask, void* workspace, DTypeDesc type_desc) override;

  void relu_inf(engine_handle backend_handle, const ReLUStats& stats, const void* input,
                void* output, void* workspace, DTypeDesc type_desc) override;

  void relu_bwd(engine_handle backend_handle, const ReLUStats& stats, const void* grad_output,
                void* grad_input, const bool* mask, void* workspace, DTypeDesc type_desc) override;

  void embedding_fwd(engine_handle backend_handle, const EmbeddingStats& stats, const void* input,
                     const void* weight, void* output, void* workspace,
                     DTypeDesc type_desc) override;

  void embedding_bwd(engine_handle backend_handle, const EmbeddingStats& stats,
                     const void* grad_output, const void* input, void* grad_weight, void* workspace,
                     DTypeDesc type_desc) override;

  void positional_embedding_fwd(engine_handle backend_handle, const PositionalEmbeddingStats& stats,
                                const void* input, const void* pos_embedding, void* output,
                                void* workspace, DTypeDesc type_desc) override;

  void positional_embedding_bwd(engine_handle backend_handle, const PositionalEmbeddingStats& stats,
                                const void* grad_output, void* grad_pos_embedding, void* workspace,
                                DTypeDesc type_desc) override;

  void batchnorm_fwd(engine_handle backend_handle, const BatchNormStats& stats, const void* input,
                     const void* gamma, const void* beta, void* output, void* prev_running_mean,
                     void* prev_running_var, void* next_running_mean, void* next_running_var,
                     void* batch_mean, void* batch_invar, void* relu_mask, void* workspace,
                     DTypeDesc type_desc) override;

  void batchnorm_infer(engine_handle backend_handle, const BatchNormStats& stats, const void* input,
                       const void* gamma, const void* beta, const void* saved_mean,
                       const void* saved_var, void* output, void* workspace,
                       DTypeDesc type_desc) override;

  void batchnorm_bwd(engine_handle backend_handle, const BatchNormStats& stats,
                     const void* grad_output, const void* input, const void* relu_mask,
                     const void* gamma, void* grad_input, void* grad_gamma, void* grad_beta,
                     const void* batch_mean, const void* batch_invar, void* workspace,
                     DTypeDesc type_desc) override;

  void conv2d_fwd(engine_handle backend_handle, const Conv2DStats& stats, const void* input,
                  const void* weight, const void* bias, void* output, void* workspace,
                  DTypeDesc type_desc) override;

  void conv2d_dgrad(engine_handle backend_handle, const Conv2DStats& stats, const void* grad_output,
                    const void* weight, void* grad_input, void* workspace,
                    DTypeDesc type_desc) override;

  void conv2d_wgrad(engine_handle backend_handle, const Conv2DStats& stats, const void* grad_output,
                    const void* input, void* grad_weight, void* workspace,
                    DTypeDesc type_desc) override;

  void conv2d_bgrad(engine_handle backend_handle, const Conv2DStats& stats, const void* grad_output,
                    void* grad_bias, void* workspace, DTypeDesc type_desc) override;

  void layernorm_fwd(engine_handle backend_handle, const LayerNormStats& stats, const void* input,
                     const void* gamma, const void* beta, void* output, void* mean,
                     void* inv_variance, void* workspace, DTypeDesc type_desc) override;

  void layernorm_infer(engine_handle backend_handle, const LayerNormStats& stats, const void* input,
                       const void* gamma, const void* beta, void* output, void* workspace,
                       DTypeDesc type_desc) override;

  void layernorm_bwd(engine_handle backend_handle, const LayerNormStats& stats,
                     const void* grad_output, const void* input, const void* gamma,
                     const void* mean, const void* inv_variance, void* grad_input, void* grad_gamma,
                     void* grad_beta, void* workspace, DTypeDesc type_desc) override;

  void sdpa_fwd(engine_handle backend_handle, const AttentionStats& stats, const void* q_data,
                const void* k_data, const void* v_data, void* o_data, void* stats_data,
                void* workspace, DTypeDesc type_desc) override;

  void sdpa_bwd(engine_handle backend_handle, const AttentionStats& stats, const void* q_data,
                const void* k_data, const void* v_data, const void* o_data, const void* dO_data,
                const void* stats_data, void* dQ_data, void* dK_data, void* dV_data,
                void* workspace, DTypeDesc type_desc) override;

  void transpose(engine_handle backend_handle, const TransposeStats& stats, const void* input,
                 void* output, void* workspace, DTypeDesc type_desc) override;
};

}  // namespace tunx

#endif