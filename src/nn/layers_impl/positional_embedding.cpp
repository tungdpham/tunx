/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/layers_impl/positional_embedding.hpp"

#include <cmath>
#include <stdexcept>

#include "nn/engines/iengine.hpp"
#include "nn/stats/stats.hpp"
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace tunx {
namespace internal {

PositionalEmbeddingImpl::PositionalEmbeddingImpl(size_t embed_dim, size_t seq_len,
                                                 const std::string &name)
    : SISOLayerImpl(name),
      embed_dim_(embed_dim),
      seq_len_(seq_len) {}

void PositionalEmbeddingImpl::init_impl() {
  float bound = static_cast<float>(1.0 / std::sqrt(static_cast<double>(embed_dim_)));
  long long seed = this->use_seed_ ? this->srand_seed_
                                   : std::chrono::system_clock::now().time_since_epoch().count();

  fill_normal(pos_embedding_, 0, bound, seed);

  fill(pos_embedding_gradients_, 0.0f);
}

Tensor PositionalEmbeddingImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  const auto &shape = input.shape();
  if (shape.size() < 2) {
    throw std::runtime_error("PositionalEmbeddingImpl: Input tensor must be at least 2D");
  }

  size_t last_dim = shape.back();
  size_t second_last_dim = shape[shape.size() - 2];

  if (last_dim != embed_dim_) {
    throw std::runtime_error("PositionalEmbeddingImpl: Input last dim (" +
                             std::to_string(last_dim) + ") must match embed_dim (" +
                             std::to_string(embed_dim_) + ")");
  }
  if (second_last_dim != seq_len_) {
    throw std::runtime_error("PositionalEmbeddingImpl: Input sequence length (" +
                             std::to_string(second_last_dim) + ") must match seq_len (" +
                             std::to_string(seq_len_) + ")");
  }

  Tensor output = make_tensor(shape, io_dtype_);

  size_t batch_size = 1;
  for (size_t i = 0; i + 2 < shape.size(); ++i) {
    batch_size *= shape[i];
  }

  PositionalEmbeddingStats stats{
      .batch_size = batch_size,
      .seq_len = seq_len_,
      .embed_dim = embed_dim_,
  };

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  WorkspaceReq ws_req = engine_->query_positional_embedding_graph(engine_handle_, stats, type_desc);
  Tensor ws = make_tensor({ws_req.fwd_workspace}, DType_t::BYTE);

  engine_->positional_embedding_fwd(engine_handle_, stats, input.data_as<void>(),
                                    pos_embedding_.data_as<void>(), output.data_as<void>(),
                                    ws.data_as<void>(), type_desc);

  return output;
}

Tensor PositionalEmbeddingImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  const auto &shape = grad_output.shape();
  if (shape.size() < 2) {
    throw std::runtime_error("PositionalEmbeddingImpl: Gradient tensor must be at least 2D");
  }

  size_t last_dim = shape.back();
  size_t second_last_dim = shape[shape.size() - 2];

  if (last_dim != embed_dim_) {
    throw std::runtime_error("PositionalEmbeddingImpl: Gradient last dim (" +
                             std::to_string(last_dim) + ") must match embed_dim (" +
                             std::to_string(embed_dim_) + ")");
  }
  if (second_last_dim != seq_len_) {
    throw std::runtime_error("PositionalEmbeddingImpl: Gradient sequence length (" +
                             std::to_string(second_last_dim) + ") must match seq_len (" +
                             std::to_string(seq_len_) + ")");
  }

  Tensor grad_input = make_tensor(shape, io_dtype_);

  copy(grad_output, grad_input, engine_handle_.get_stream());

  size_t batch_size = 1;
  for (size_t i = 0; i + 2 < shape.size(); ++i) {
    batch_size *= shape[i];
  }

  PositionalEmbeddingStats stats{
      .batch_size = batch_size,
      .seq_len = seq_len_,
      .embed_dim = embed_dim_,
  };

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  WorkspaceReq ws_req = engine_->query_positional_embedding_graph(engine_handle_, stats, type_desc);
  Tensor ws = make_tensor({ws_req.bwd_workspace}, DType_t::BYTE);

  engine_->positional_embedding_bwd(engine_handle_, stats, grad_output.data_as<void>(),
                                    pos_embedding_gradients_.data_as<void>(), ws.data_as<void>(),
                                    type_desc);

  return grad_input;
}

LayerConfig PositionalEmbeddingImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("embed_dim", embed_dim_);
  config.set("seq_len", seq_len_);
  return config;
}

Vec<size_t> PositionalEmbeddingImpl::compute_output_shape(const Vec<size_t> &input_shape) const {
  return input_shape;
}

std::shared_ptr<PositionalEmbeddingImpl> PositionalEmbeddingImpl::create_from_config(
    const LayerConfig &config) {
  size_t embed_dim = config.get<size_t>("embed_dim");
  size_t seq_len = config.get<size_t>("seq_len");
  return std::make_shared<PositionalEmbeddingImpl>(embed_dim, seq_len, config.name);
}

}  // namespace internal
}  // namespace tunx
