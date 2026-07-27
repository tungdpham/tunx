/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/class_token.hpp"

#include <cmath>
#include <stdexcept>

#include "nn/engines/iengine.hpp"
#include "type/type.hpp"

namespace tunx {

Vec<Vec<size_t>> ClassTokenOp::output_shapes(const Vec<Vec<size_t>> &input_shapes,
                                             const Config &config) {
  if (input_shapes.empty() || input_shapes[0].size() < 3) {
    throw std::runtime_error("ClassTokenOp: Input shape must have at least 3 dimensions");
  }
  const auto &input_shape = input_shapes[0];
  size_t batch_size = input_shape[0];
  size_t seq_len = input_shape[1];
  size_t embed_dim = input_shape[2];
  return {{batch_size, seq_len + 1, embed_dim}};
}

Tensor ClassTokenOp::forward(OpContext &ctx, const Tensor &input, const Param &class_token,
                             const Config &config) {
  if (input.dims() != 3) {
    throw std::runtime_error(
        "ClassTokenOp: Input tensor must have 3 dimensions (Batch, Seq, Embed)");
  }
  size_t batch_size = input.dim(0);
  size_t seq_len = input.dim(1);
  size_t embed_dim = input.dim(2);

  if (embed_dim != config.embed_dim) {
    throw std::runtime_error("ClassTokenOp: Input embed_dim must match layer embed_dim");
  }

  ClassTokenStats stats{
      .batch_size = batch_size,
      .seq_len = seq_len,
      .embed_dim = embed_dim,
  };

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  WorkspaceReq ws_req = ctx.engine->query_class_token_graph(ctx.handle, stats, type_desc);

  Tensor output = ctx.make_tensor({batch_size, seq_len + 1, embed_dim}, ctx.io_dtype);
  Tensor ws = ctx.make_tensor({ws_req.fwd_workspace}, DType_t::BYTE);

  ctx.engine->class_token_fwd(ctx.handle, stats, input.data_as<void>(), class_token.data_as<void>(),
                              output.data_as<void>(), ws.data_as<void>(), type_desc);

  return output;
}

Tensor ClassTokenOp::backward(OpContext &ctx, const Tensor &grad_output, Param &class_token,
                              const Config &config) {
  if (grad_output.dims() != 3) {
    throw std::runtime_error(
        "ClassTokenOp: Gradient tensor must have 3 dimensions (Batch, Seq, Embed)");
  }
  size_t batch_size = grad_output.dim(0);
  size_t seq_len_plus_1 = grad_output.dim(1);
  size_t embed_dim = grad_output.dim(2);
  size_t seq_len = seq_len_plus_1 - 1;

  ClassTokenStats stats{
      .batch_size = batch_size,
      .seq_len = seq_len,
      .embed_dim = embed_dim,
  };

  DTypeDesc type_desc{
      .io_dtype = ctx.io_dtype,
      .param_dtype = ctx.param_dtype,
      .compute_dtype = ctx.compute_dtype,
  };

  Tensor grad_input = ctx.make_tensor({batch_size, seq_len, embed_dim}, ctx.io_dtype);
  WorkspaceReq ws_req = ctx.engine->query_class_token_graph(ctx.handle, stats, type_desc);
  Tensor ws = ctx.make_tensor({ws_req.bwd_workspace}, DType_t::BYTE);

  ctx.engine->class_token_bwd(ctx.handle, stats, grad_output.data_as<void>(),
                              grad_input.data_as<void>(), class_token.grad_as<void>(),
                              ws.data_as<void>(), type_desc);

  return grad_input;
}

LayerConfig ClassTokenOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.name = name;
  lcfg.type = TYPE_NAME;
  lcfg.set("embed_dim", config.embed_dim);
  return lcfg;
}

ClassTokenOp::Config ClassTokenOp::parse_config(const LayerConfig &config) {
  Config c;
  c.embed_dim = config.get<size_t>("embed_dim");
  return c;
}

}  // namespace tunx
