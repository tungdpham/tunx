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
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace tunx {
namespace internal {

ClassTokenImpl::ClassTokenImpl(size_t embed_dim, const std::string &name)
    : SISOLayerImpl(name),
      embed_dim_(embed_dim) {}

void ClassTokenImpl::init_impl() {
  float bound = static_cast<float>(1.0 / std::sqrt(static_cast<double>(embed_dim_)));
  long long seed = this->use_seed_ ? this->srand_seed_
                                   : std::chrono::system_clock::now().time_since_epoch().count();

  class_token_ = make_param({embed_dim_}, DType_t::FP32);
  fill_normal(class_token_.data(), 0, bound, seed);
}

Tensor ClassTokenImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  if (input.dims() != 3) {
    throw std::runtime_error(
        "ClassTokenImpl: Input tensor must have 3 dimensions (Batch, Seq, Embed)");
  }
  size_t batch_size = input.dim(0);
  size_t seq_len = input.dim(1);
  size_t embed_dim = input.dim(2);

  if (embed_dim != embed_dim_) {
    throw std::runtime_error("ClassTokenImpl: Input embed_dim must match layer embed_dim");
  }

  ClassTokenStats stats{
      .batch_size = batch_size,
      .seq_len = seq_len,
      .embed_dim = embed_dim,
  };

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  WorkspaceReq ws_req = engine_->query_class_token_graph(engine_handle_, stats, type_desc);

  Tensor output = make_tensor({batch_size, seq_len + 1, embed_dim}, input.dtype());
  Tensor ws = make_tensor({ws_req.fwd_workspace}, DType_t::BYTE);

  engine_->class_token_fwd(engine_handle_, stats, input.data_as<void>(),
                           class_token_.data_as<void>(), output.data_as<void>(), ws.data_as<void>(),
                           type_desc);

  return output;
}

Tensor ClassTokenImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  if (grad_output.dims() != 3) {
    throw std::runtime_error(
        "ClassTokenImpl: Gradient tensor must have 3 dimensions (Batch, Seq, Embed)");
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
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  Tensor grad_input = make_tensor({batch_size, seq_len, embed_dim}, grad_output.dtype());
  WorkspaceReq ws_req = engine_->query_class_token_graph(engine_handle_, stats, type_desc);
  Tensor ws = make_tensor({ws_req.bwd_workspace}, DType_t::BYTE);

  engine_->class_token_bwd(engine_handle_, stats, grad_output.data_as<void>(),
                           grad_input.data_as<void>(), class_token_.grad_as<void>(),
                           ws.data_as<void>(), type_desc);

  return grad_input;
}

LayerConfig ClassTokenImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("embed_dim", embed_dim_);
  return config;
}

Vec<size_t> ClassTokenImpl::compute_output_shape(const Vec<size_t> &input_shape) const {
  if (input_shape.size() < 3) {
    throw std::runtime_error("ClassTokenImpl: Input shape must have at least 3 dimensions");
  }
  size_t batch_size = input_shape[0];
  size_t seq_len = input_shape[1];
  size_t embed_dim = input_shape[2];
  return {batch_size, seq_len + 1, embed_dim};
}

std::shared_ptr<ClassTokenImpl> ClassTokenImpl::create_from_config(const LayerConfig &config) {
  size_t embed_dim = config.get<size_t>("embed_dim");
  return std::make_shared<ClassTokenImpl>(embed_dim, config.name);
}

}  // namespace internal
}  // namespace tunx
