/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/embedding.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>

#include "tensor/ops.hpp"
#include "type/type.hpp"

namespace tunx {

void EmbeddingOp::init(OpContext &ctx, const Config &config) {
  size_t vocab_size = config.vocab_size;
  size_t embed_dim = config.embed_dim;
  size_t padding_idx = config.padding_idx;
  float stddev = static_cast<float>(1.0 / std::sqrt(static_cast<double>(embed_dim)));
  long long seed =
      ctx.use_seed ? ctx.srand_seed : std::chrono::system_clock::now().time_since_epoch().count();
  Param weight = ctx.make_param({vocab_size, embed_dim});
  fill_normal(weight.data(), 0, stddev, seed);
  if (padding_idx < vocab_size) {
    for (size_t i = 0; i < embed_dim; ++i) {
      DISPATCH_DTYPE(weight.dtype(), T, weight.data().at<T>({padding_idx, i}) = 0.0f);
    }
  }
}

Vec<Vec<size_t>> EmbeddingOp::output_shapes(const Vec<Vec<size_t>> &input_shapes,
                                            const Config &config) {
  if (input_shapes.empty()) throw std::invalid_argument("EmbeddingOp expects input shapes");
  Vec<size_t> out = input_shapes[0];
  out.push_back(config.embed_dim);
  return {out};
}

Tensor EmbeddingOp::forward(OpContext &ctx, const Tensor &input, const Param &weight,
                            const Config &config) {
  if (ctx.is_training) {
    ctx.residuals["input"] = input;
  }

  size_t num_tokens = input.size();
  if (num_tokens == 0) return Tensor();

  Vec<size_t> out_shape = input.shape();
  out_shape.push_back(config.embed_dim);
  Tensor output = ctx.make_tensor(out_shape, ctx.io_dtype);

  EmbeddingStats stats{
      .num_indices = num_tokens,
      .vocab_size = config.vocab_size,
      .embed_dim = config.embed_dim,
      .padding_idx = config.padding_idx,
  };

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  if (input.dtype() != DType_t::INT32) {
    throw std::runtime_error("EmbeddingOp input dtype must be INT32");
  }

  WorkspaceReq ws_req = ctx.engine->query_embedding_graph(ctx.handle, stats, type_desc);
  Tensor ws = ctx.make_tensor({ws_req.fwd_workspace}, DType_t::BYTE);

  ctx.engine->embedding_fwd(ctx.handle, stats, input.data_as<void>(), weight.data_as<void>(),
                            output.data_as<void>(), ws.data_as<void>(), type_desc);

  return output;
}

Tensor EmbeddingOp::backward(OpContext &ctx, const Tensor &grad_output, Param &weight,
                             const Config &config) {
  const Tensor &input = ctx.residuals["input"];

  Tensor grad_input = ctx.make_tensor(input.shape(), input.dtype());
  fill(grad_input, 0.0f, ctx.handle.get_stream());

  size_t num_tokens = input.size();

  EmbeddingStats stats{
      .num_indices = num_tokens,
      .vocab_size = config.vocab_size,
      .embed_dim = config.embed_dim,
      .padding_idx = config.padding_idx,
  };

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  WorkspaceReq ws_req = ctx.engine->query_embedding_graph(ctx.handle, stats, type_desc);
  Tensor ws = ctx.make_tensor({ws_req.bwd_workspace}, DType_t::BYTE);

  ctx.engine->embedding_bwd(ctx.handle, stats, grad_output.data_as<void>(), input.data_as<void>(),
                            weight.grad_as<void>(), ws.data_as<void>(), type_desc);

  return grad_input;
}

LayerConfig EmbeddingOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.name = name;
  lcfg.type = TYPE_NAME;
  lcfg.set("vocab_size", config.vocab_size);
  lcfg.set("embed_dim", config.embed_dim);
  lcfg.set("padding_idx", config.padding_idx);
  return lcfg;
}

EmbeddingOp::Config EmbeddingOp::parse_config(const LayerConfig &config) {
  Config c;
  c.vocab_size = config.get<size_t>("vocab_size");
  c.embed_dim = config.get<size_t>("embed_dim");
  c.padding_idx = config.get<size_t>("padding_idx", static_cast<size_t>(-1));
  return c;
}

}  // namespace tunx
