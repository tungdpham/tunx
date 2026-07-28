/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/positional_embedding.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>

#include "nn/engines/iengine.hpp"
#include "nn/stats/stats.hpp"
#include "tensor/ops.hpp"
#include "type/type.hpp"

namespace tunx {

void PositionalEmbeddingOp::init(OpContext &ctx, const Config &config) {
  size_t embed_dim = config.embed_dim;
  size_t seq_len = config.seq_len;
  float bound = static_cast<float>(1.0 / std::sqrt(static_cast<double>(embed_dim)));
  long long seed =
      ctx.use_seed ? ctx.srand_seed : std::chrono::system_clock::now().time_since_epoch().count();
  Param pos_embedding = ctx.make_param({seq_len, embed_dim});
  fill_normal(pos_embedding.data(), 0, bound, seed);
}

Vec<Vec<size_t>> PositionalEmbeddingOp::output_shapes(const Vec<Vec<size_t>> &input_shapes,
                                                      const Config &config) {
  return input_shapes;
}

Tensor PositionalEmbeddingOp::forward(OpContext &ctx, const Tensor &input,
                                      const Param &pos_embedding, const Config &config) {
  const auto &shape = input.shape();
  if (shape.size() < 2) {
    throw std::runtime_error("PositionalEmbeddingOp: Input tensor must be at least 2D");
  }

  size_t last_dim = shape.back();
  size_t second_last_dim = shape[shape.size() - 2];

  if (last_dim != config.embed_dim) {
    throw std::runtime_error("PositionalEmbeddingOp: Input last dim must match embed_dim");
  }
  if (second_last_dim != config.seq_len) {
    throw std::runtime_error("PositionalEmbeddingOp: Input sequence length must match seq_len");
  }

  Tensor output = ctx.make_tensor(shape, ctx.io_dtype);

  size_t batch_size = 1;
  for (size_t i = 0; i + 2 < shape.size(); ++i) {
    batch_size *= shape[i];
  }

  PositionalEmbeddingStats stats{
      .batch_size = batch_size,
      .seq_len = config.seq_len,
      .embed_dim = config.embed_dim,
  };

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  WorkspaceReq ws_req = ctx.engine->query_positional_embedding_graph(ctx.handle, stats, type_desc);
  Tensor ws = ctx.make_tensor({ws_req.fwd_workspace}, DType_t::BYTE);

  ctx.engine->positional_embedding_fwd(ctx.handle, stats, input.data_as<void>(),
                                       pos_embedding.data_as<void>(), output.data_as<void>(),
                                       ws.data_as<void>(), type_desc);

  return output;
}

Tensor PositionalEmbeddingOp::backward(OpContext &ctx, const Tensor &grad_output,
                                       Param &pos_embedding, const Config &config) {
  const auto &shape = grad_output.shape();
  if (shape.size() < 2) {
    throw std::runtime_error("PositionalEmbeddingOp: Gradient tensor must be at least 2D");
  }

  size_t last_dim = shape.back();
  size_t second_last_dim = shape[shape.size() - 2];

  if (last_dim != config.embed_dim) {
    throw std::runtime_error("PositionalEmbeddingOp: Gradient last dim must match embed_dim");
  }
  if (second_last_dim != config.seq_len) {
    throw std::runtime_error("PositionalEmbeddingOp: Gradient sequence length must match seq_len");
  }

  Tensor grad_input = ctx.make_tensor(shape, ctx.io_dtype);
  copy(grad_output, grad_input, ctx.handle.get_stream());

  size_t batch_size = 1;
  for (size_t i = 0; i + 2 < shape.size(); ++i) {
    batch_size *= shape[i];
  }

  PositionalEmbeddingStats stats{
      .batch_size = batch_size,
      .seq_len = config.seq_len,
      .embed_dim = config.embed_dim,
  };

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  WorkspaceReq ws_req = ctx.engine->query_positional_embedding_graph(ctx.handle, stats, type_desc);
  Tensor ws = ctx.make_tensor({ws_req.bwd_workspace}, DType_t::BYTE);

  ctx.engine->positional_embedding_bwd(ctx.handle, stats, grad_output.data_as<void>(),
                                       pos_embedding.grad_as<void>(), ws.data_as<void>(),
                                       type_desc);

  return grad_input;
}

LayerConfig PositionalEmbeddingOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.name = name;
  lcfg.type = TYPE_NAME;
  lcfg.set("embed_dim", config.embed_dim);
  lcfg.set("seq_len", config.seq_len);
  return lcfg;
}

PositionalEmbeddingOp::Config PositionalEmbeddingOp::parse_config(const LayerConfig &config) {
  Config c;
  c.embed_dim = config.get<size_t>("embed_dim");
  c.seq_len = config.get<size_t>("seq_len");
  return c;
}

}  // namespace tunx
