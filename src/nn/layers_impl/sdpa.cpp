/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/sdpa.hpp"

#include <cmath>
#include <stdexcept>

namespace tunx {

LayerConfig SDPAOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lc;
  lc.type = TYPE_NAME;
  lc.name = name;
  lc.set("attn_scale", config.attn_scale);
  lc.set("is_causal", config.is_causal);
  return lc;
}

SDPAOp::Config SDPAOp::parse_config(const LayerConfig &lc) {
  Config config;
  config.attn_scale = lc.get<float>("attn_scale", 1.0f);
  config.is_causal = lc.get<bool>("is_causal", false);
  return config;
}

Vec<Vec<size_t>> SDPAOp::output_shapes(const Vec<Vec<size_t>> &input_shapes, const Config &config) {
  if (input_shapes.size() != 3) {
    throw std::runtime_error("SDPAOp: expected exactly 3 inputs (Q, K, V)");
  }

  // All inputs should have same shape: (B, H, S, D)
  const auto &q_shape = input_shapes[0];
  const auto &k_shape = input_shapes[1];
  const auto &v_shape = input_shapes[2];

  if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4) {
    throw std::runtime_error("SDPAOp: inputs must be 4D (B, H, S, D)");
  }

  if (q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0]) {
    throw std::runtime_error("SDPAOp: batch size mismatch");
  }
  if (q_shape[1] != k_shape[1] || q_shape[1] != v_shape[1]) {
    throw std::runtime_error("SDPAOp: number of heads mismatch");
  }
  if (q_shape[2] != k_shape[2] || q_shape[2] != v_shape[2]) {
    throw std::runtime_error("SDPAOp: sequence length mismatch");
  }
  if (q_shape[3] != k_shape[3] || q_shape[3] != v_shape[3]) {
    throw std::runtime_error("SDPAOp: head dim mismatch");
  }

  // Output shape same as Q: (B, H, S, D)
  return {q_shape};
}

Tensor SDPAOp::forward(OpContext &ctx, const Tensor &q, const Tensor &k, const Tensor &v,
                       const Config &config) {
  if (q.dims() != 4) {
    throw std::runtime_error("SDPAOp: Q must be 4D (B, H, S, D)");
  }

  const auto &q_shape = q.shape();
  size_t batch_size = q_shape[0];
  size_t num_heads = q_shape[1];
  size_t seq_len = q_shape[2];
  size_t head_dim = q_shape[3];

  // Validate K and V shapes
  {
    const auto &k_shape = k.shape();
    const auto &v_shape = v.shape();
    if (k_shape != q_shape || v_shape != q_shape) {
      throw std::runtime_error("SDPAOp: Q, K, V must have same shape");
    }
  }

  Tensor output = ctx.make_tensor(q_shape, ctx.io_dtype);

  if (ctx.is_training) {
    ctx.residuals["q"] = q;
    ctx.residuals["k"] = k;
    ctx.residuals["v"] = v;
    ctx.residuals["output"] = output;
  }

  AttentionStats stats{
      .batch_size = batch_size,
      .num_heads = num_heads,
      .seq_len = seq_len,
      .head_dim = head_dim,
      .attn_scale = config.attn_scale,
      .is_causal = config.is_causal,
  };

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  WorkspaceReq req = ctx.engine->query_sdpa_graph(ctx.handle, stats, type_desc);

  Tensor workspace = ctx.make_tensor({req.fwd_workspace}, DType_t::BYTE);

  size_t stats_elements =
      std::max(batch_size * num_heads * seq_len * seq_len, batch_size * num_heads * seq_len * 1);
  Tensor stats_tensor = ctx.make_tensor({stats_elements}, ctx.io_dtype);

  if (ctx.is_training) {
    ctx.residuals["stats"] = stats_tensor;
  }

  ctx.engine->sdpa_fwd(ctx.handle, stats, q.data_as<void>(), k.data_as<void>(), v.data_as<void>(),
                       output.data_as<void>(), stats_tensor.data_as<void>(),
                       workspace.data_as<void>(), type_desc);

  return output;
}

Vec<Tensor> SDPAOp::backward(OpContext &ctx, const Tensor &grad_output, const Config &config) {
  const Tensor &q = ctx.residuals["q"];
  const Tensor &k = ctx.residuals["k"];
  const Tensor &v = ctx.residuals["v"];
  const Tensor &output = ctx.residuals["output"];
  const Tensor &stats_tensor = ctx.residuals["stats"];

  const Vec<size_t> &q_shape = q.shape();
  size_t batch_size = q_shape[0];
  size_t num_heads = q_shape[1];
  size_t seq_len = q_shape[2];
  size_t head_dim = q_shape[3];

  Tensor grad_q = ctx.make_tensor(q_shape, ctx.io_dtype);
  Tensor grad_k = ctx.make_tensor(q_shape, ctx.io_dtype);
  Tensor grad_v = ctx.make_tensor(q_shape, ctx.io_dtype);

  AttentionStats stats{
      .batch_size = batch_size,
      .num_heads = num_heads,
      .seq_len = seq_len,
      .head_dim = head_dim,
      .attn_scale = config.attn_scale,
      .is_causal = config.is_causal,
  };

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  WorkspaceReq req = ctx.engine->query_sdpa_graph(ctx.handle, stats, type_desc);

  Tensor workspace = ctx.make_tensor({req.bwd_workspace}, DType_t::BYTE);

  ctx.engine->sdpa_bwd(ctx.handle, stats, q.data_as<void>(), k.data_as<void>(), v.data_as<void>(),
                       output.data_as<void>(), grad_output.data_as<void>(),
                       stats_tensor.data_as<void>(), grad_q.data_as<void>(), grad_k.data_as<void>(),
                       grad_v.data_as<void>(), workspace.data_as<void>(), type_desc);

  return {grad_q, grad_k, grad_v};
}

SDPA::SDPA(float attn_scale, bool is_causal, const std::string &name)
    : FunctionalLayer(SDPAOp::Config{attn_scale, is_causal}, name) {}

}  // namespace tunx
