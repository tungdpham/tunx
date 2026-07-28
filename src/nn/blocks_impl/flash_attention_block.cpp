#include "nn/blocks_impl/flash_attention_block.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "nn/engines/engine_handle.hpp"
#include "nn/layers_impl/dense.hpp"
#include "nn/stats/stats.hpp"
#include "tensor/ops.hpp"
#include "type/type.hpp"

namespace tunx {

LayerConfig FlashAttentionBlockOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lc;
  lc.name = name;
  lc.type = TYPE_NAME;
  lc.set("embed_dim", config.embed_dim);
  lc.set("num_heads", config.num_heads);
  lc.set("is_causal", config.is_causal);
  return lc;
}

FlashAttentionBlockOp::Config FlashAttentionBlockOp::parse_config(const LayerConfig &lc) {
  Config config;
  config.embed_dim = lc.get<size_t>("embed_dim");
  config.num_heads = lc.get<size_t>("num_heads");
  config.is_causal = lc.get<bool>("is_causal", true);
  return config;
}

Vec<Vec<size_t>> FlashAttentionBlockOp::output_shapes(const Vec<Vec<size_t>> &input_shapes,
                                                      const Config &config) {
  return input_shapes;
}

Tensor FlashAttentionBlockOp::forward(OpContext &ctx, const Tensor &input, Layer q_proj,
                                      Layer k_proj, Layer v_proj, Layer out_proj,
                                      const Config &config) {
  if (input.dims() != 3) {
    throw std::invalid_argument("FlashAttentionBlock: Input must be 3D (B, S, E) but got " +
                                std::to_string(input.dims()) + "D");
  }

  const auto &input_shape = input.shape();
  size_t batch_size = input_shape[0];
  size_t seq_len = input_shape[1];
  size_t embed_dim = input.dim(2);

  if (embed_dim != config.embed_dim) {
    throw std::invalid_argument("FlashAttentionBlock: Input embed_dim mismatch");
  }

  size_t head_dim = config.embed_dim / config.num_heads;

  AttentionStats stats{
      .batch_size = batch_size,
      .num_heads = config.num_heads,
      .seq_len = seq_len,
      .head_dim = head_dim,
      .attn_scale = 1.0f,
      .is_causal = config.is_causal,
  };

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  if (ctx.is_training) {
    ctx.residuals["input"] = input;
  }

  Tensor attn_out = ctx.make_tensor({batch_size, seq_len, config.embed_dim}, ctx.io_dtype);
  ctx.residuals["attn_out"] = attn_out;
  Tensor stats_tensor = ctx.make_tensor({batch_size, config.num_heads, seq_len, 1}, DType_t::FP32);
  ctx.residuals["stats_tensor"] = stats_tensor;

  ctx.ws_allocator->flip();

  Tensor attn_heads = ctx.make_tensor({batch_size, config.num_heads, seq_len, head_dim},
                                      ctx.io_dtype == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);
  Tensor q_heads = ctx.make_tensor({batch_size, config.num_heads, seq_len, head_dim},
                                   ctx.io_dtype == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);
  Tensor k_heads = ctx.make_tensor({batch_size, config.num_heads, seq_len, head_dim},
                                   ctx.io_dtype == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);
  Tensor v_heads = ctx.make_tensor({batch_size, config.num_heads, seq_len, head_dim},
                                   ctx.io_dtype == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);

  Tensor q = q_proj.forward({input}, ctx.residuals["q_proj"])[0];
  Tensor k = k_proj.forward({input}, ctx.residuals["k_proj"])[0];
  Tensor v = v_proj.forward({input}, ctx.residuals["v_proj"])[0];

  TransposeStats t_stats;
  t_stats.ndim = 4;
  t_stats.dim0 = 1;
  t_stats.dim1 = 2;
  t_stats.shape[0] = batch_size;
  t_stats.shape[1] = seq_len;
  t_stats.shape[2] = config.num_heads;
  t_stats.shape[3] = head_dim;

  DTypeDesc t_desc = type_desc;
  t_desc.compute_dtype = ctx.io_dtype == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16;

  ctx.engine->transpose(ctx.handle, t_stats, q.data_as<void>(), q_heads.data_as<void>(), nullptr,
                        t_desc);
  ctx.engine->transpose(ctx.handle, t_stats, k.data_as<void>(), k_heads.data_as<void>(), nullptr,
                        t_desc);
  ctx.engine->transpose(ctx.handle, t_stats, v.data_as<void>(), v_heads.data_as<void>(), nullptr,
                        t_desc);

  {
    WorkspaceReq ws_req = ctx.engine->query_sdpa_graph(ctx.handle, stats, type_desc);
    size_t workspace_size = ws_req.fwd_workspace;
    Tensor workspace = ctx.make_tensor({workspace_size}, DType_t::BYTE);

    ctx.engine->sdpa_fwd(ctx.handle, stats, q_heads.data_as<void>(), k_heads.data_as<void>(),
                         v_heads.data_as<void>(), attn_heads.data_as<void>(),
                         stats_tensor.data_as<void>(), workspace.data_as<void>(), type_desc);
  }

  TransposeStats t_stats_rev;
  t_stats_rev.ndim = 4;
  t_stats_rev.dim0 = 1;
  t_stats_rev.dim1 = 2;
  t_stats_rev.shape[0] = batch_size;
  t_stats_rev.shape[1] = config.num_heads;
  t_stats_rev.shape[2] = seq_len;
  t_stats_rev.shape[3] = head_dim;

  ctx.engine->transpose(ctx.handle, t_stats_rev, attn_heads.data_as<void>(),
                        attn_out.data_as<void>(), nullptr, t_desc);

  ctx.ws_allocator->flip();

  Tensor output = out_proj.forward({attn_out}, ctx.residuals["out_proj"])[0];
  return output;
}

Vec<Tensor> FlashAttentionBlockOp::backward(OpContext &ctx, const Tensor &grad_output,
                                            Layer q_proj, Layer k_proj, Layer v_proj,
                                            Layer out_proj, const Config &config) {
  const auto &grad_shape = grad_output.shape();
  size_t batch_size = grad_shape[0];
  size_t seq_len = grad_shape[1];

  size_t head_dim = config.embed_dim / config.num_heads;

  AttentionStats stats{
      .batch_size = batch_size,
      .num_heads = config.num_heads,
      .seq_len = seq_len,
      .head_dim = head_dim,
      .attn_scale = 1.0f,
      .is_causal = config.is_causal,
  };

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  const Tensor &input = ctx.residuals["input"];
  Tensor &attn_out = ctx.residuals["attn_out"];
  Tensor &stats_tensor = ctx.residuals["stats_tensor"];

  Tensor grad_input = ctx.make_tensor({batch_size, seq_len, config.embed_dim}, ctx.io_dtype);

  ctx.ws_allocator->flip();

  Tensor grad_q = ctx.make_tensor({batch_size, seq_len, config.embed_dim}, ctx.io_dtype);
  Tensor grad_k = ctx.make_tensor({batch_size, seq_len, config.embed_dim}, ctx.io_dtype);
  Tensor grad_v = ctx.make_tensor({batch_size, seq_len, config.embed_dim}, ctx.io_dtype);

  Tensor grad_attn_heads = ctx.make_tensor(
      {batch_size, config.num_heads, seq_len, head_dim},
      ctx.io_dtype == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);

  TransposeStats t_stats;
  t_stats.ndim = 4;
  t_stats.dim0 = 1;
  t_stats.dim1 = 2;
  t_stats.shape[0] = batch_size;
  t_stats.shape[1] = seq_len;
  t_stats.shape[2] = config.num_heads;
  t_stats.shape[3] = head_dim;

  DTypeDesc t_desc = type_desc;
  t_desc.compute_dtype = ctx.io_dtype == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16;

  {
    Tensor grad_attn_out = out_proj.backward({grad_output}, ctx.residuals["out_proj"])[0];
    ctx.engine->transpose(ctx.handle, t_stats, grad_attn_out.data_as<void>(),
                          grad_attn_heads.data_as<void>(), nullptr, t_desc);
  }

  Tensor grad_q_heads = ctx.make_tensor(
      {batch_size, config.num_heads, seq_len, head_dim},
      ctx.io_dtype == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);
  Tensor grad_k_heads = ctx.make_tensor(
      {batch_size, config.num_heads, seq_len, head_dim},
      ctx.io_dtype == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);
  Tensor grad_v_heads = ctx.make_tensor(
      {batch_size, config.num_heads, seq_len, head_dim},
      ctx.io_dtype == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);

  Tensor q_heads = ctx.make_tensor({batch_size, config.num_heads, seq_len, head_dim},
                                   ctx.io_dtype == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);
  Tensor k_heads = ctx.make_tensor({batch_size, config.num_heads, seq_len, head_dim},
                                   ctx.io_dtype == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);
  Tensor v_heads = ctx.make_tensor({batch_size, config.num_heads, seq_len, head_dim},
                                   ctx.io_dtype == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);

  Tensor attn_heads = ctx.make_tensor(
      {batch_size, config.num_heads, seq_len, head_dim},
      ctx.io_dtype == DType_t::FP32 ? DType_t::FP32 : DType_t::BF16);

  {
    Tensor q = q_proj.forward({input}, ctx.residuals["q_proj"])[0];
    Tensor k = k_proj.forward({input}, ctx.residuals["k_proj"])[0];
    Tensor v = v_proj.forward({input}, ctx.residuals["v_proj"])[0];

    ctx.engine->transpose(ctx.handle, t_stats, q.data_as<void>(), q_heads.data_as<void>(), nullptr,
                          t_desc);
    ctx.engine->transpose(ctx.handle, t_stats, k.data_as<void>(), k_heads.data_as<void>(), nullptr,
                          t_desc);
    ctx.engine->transpose(ctx.handle, t_stats, v.data_as<void>(), v_heads.data_as<void>(), nullptr,
                          t_desc);
    ctx.engine->transpose(ctx.handle, t_stats, attn_out.data_as<void>(),
                          attn_heads.data_as<void>(), nullptr, t_desc);
  }

  {
    WorkspaceReq ws_req = ctx.engine->query_sdpa_graph(ctx.handle, stats, type_desc);
    size_t workspace_size = ws_req.bwd_workspace;
    Tensor workspace = ctx.make_tensor({workspace_size}, DType_t::BYTE);

    ctx.engine->sdpa_bwd(ctx.handle, stats, q_heads.data_as<void>(), k_heads.data_as<void>(),
                         v_heads.data_as<void>(), attn_heads.data_as<void>(),
                         grad_attn_heads.data_as<void>(), stats_tensor.data_as<void>(),
                         grad_q_heads.data_as<void>(), grad_k_heads.data_as<void>(),
                         grad_v_heads.data_as<void>(), workspace.data_as<void>(), type_desc);
  }

  TransposeStats t_stats_rev;
  t_stats_rev.ndim = 4;
  t_stats_rev.dim0 = 1;
  t_stats_rev.dim1 = 2;
  t_stats_rev.shape[0] = batch_size;
  t_stats_rev.shape[1] = config.num_heads;
  t_stats_rev.shape[2] = seq_len;
  t_stats_rev.shape[3] = head_dim;

  ctx.engine->transpose(ctx.handle, t_stats_rev, grad_q_heads.data_as<void>(),
                        grad_q.data_as<void>(), nullptr, t_desc);
  ctx.engine->transpose(ctx.handle, t_stats_rev, grad_k_heads.data_as<void>(),
                        grad_k.data_as<void>(), nullptr, t_desc);
  ctx.engine->transpose(ctx.handle, t_stats_rev, grad_v_heads.data_as<void>(),
                        grad_v.data_as<void>(), nullptr, t_desc);

  ctx.ws_allocator->flip();

  Tensor grad_q_in = q_proj.backward({grad_q}, ctx.residuals["q_proj"])[0];
  Tensor grad_k_in = k_proj.backward({grad_k}, ctx.residuals["k_proj"])[0];
  Tensor grad_v_in = v_proj.backward({grad_v}, ctx.residuals["v_proj"])[0];

  add(grad_q_in, grad_k_in, grad_input, ctx.handle.get_stream());
  add(grad_input, grad_v_in, grad_input, ctx.handle.get_stream());

  return {grad_input};
}

FlashAttentionBlock::FlashAttentionBlock(size_t embed_dim, size_t num_heads, bool is_causal,
                                         const std::string &name)
    : FunctionalLayer<FlashAttentionBlockOp>(FlashAttentionBlockOp::Config{embed_dim, num_heads, is_causal}, name) {
  if (embed_dim % num_heads != 0) {
    throw std::invalid_argument("embed_dim must be divisible by num_heads");
  }

  impl_->register_layer(static_cast<std::shared_ptr<tunx::LayerImpl>>(tunx::Layer(tunx::Dense(embed_dim, embed_dim, true, name + "_q"))));
  impl_->register_layer(static_cast<std::shared_ptr<tunx::LayerImpl>>(tunx::Layer(tunx::Dense(embed_dim, embed_dim, true, name + "_k"))));
  impl_->register_layer(static_cast<std::shared_ptr<tunx::LayerImpl>>(tunx::Layer(tunx::Dense(embed_dim, embed_dim, true, name + "_v"))));
  impl_->register_layer(static_cast<std::shared_ptr<tunx::LayerImpl>>(tunx::Layer(tunx::Dense(embed_dim, embed_dim, true, name + "_out"))));
}

Layer FlashAttentionBlock::create_from_config(const LayerConfig &config) {
  auto cfg = FlashAttentionBlockOp::parse_config(config);
  return FlashAttentionBlock(cfg.embed_dim, cfg.num_heads, cfg.is_causal, config.name.empty() ? TYPE_NAME : config.name);
}

}  // namespace tunx
