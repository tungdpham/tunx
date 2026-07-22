/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/dense.hpp"

#include <cmath>
#include <stdexcept>

#include "nn/engines/iengine.hpp"
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_ops.hpp"
#include "type/type.hpp"

namespace tunx {
namespace internal {

DenseImpl::DenseImpl(size_t input_features, size_t output_features, bool use_bias,
                     const std::string &name)
    : SISOLayerImpl(name),
      input_features_(input_features),
      output_features_(output_features),
      use_bias_(use_bias) {}

DenseImpl::~DenseImpl() {}

void DenseImpl::init_impl() {
  float stddev = static_cast<float>(1.0 / std::sqrt(static_cast<double>(input_features_)));
  long long seed = this->use_seed_ ? this->srand_seed_
                                   : std::chrono::system_clock::now().time_since_epoch().count();

  weights_ = make_param({input_features_, output_features_}, param_dtype_);
  fill_normal(weights_.data(), 0, stddev, seed);

  if (use_bias_) {
    bias_ = make_param({output_features_}, param_dtype_);
    fill_normal(bias_.data(), 0, stddev, seed);
  }
}

Tensor DenseImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  const Vec<size_t> &in_shape = input.shape();
  size_t last_dim = in_shape.back();

  if (last_dim != input_features_) {
    throw std::invalid_argument("Input feature size mismatch in DenseImpl");
  }

  size_t batch_size = 1;
  for (size_t i = 0; i < in_shape.size() - 1; ++i) {
    batch_size *= in_shape[i];
  }

  DenseStats stats{
      .batch_size = batch_size,
      .in_features = input_features_,
      .out_features = output_features_,
  };

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  WorkspaceReq ws_req = engine_->query_dense_graph(engine_handle_, stats, type_desc);

  if (this->is_training_) {
    residuals["input"] = input;
  }

  Vec<size_t> out_shape = in_shape;
  out_shape.back() = output_features_;
  Tensor output = make_tensor(out_shape, io_dtype_);
  Tensor ws = make_tensor({ws_req.fwd_workspace}, DType_t::BYTE);

  engine_->dense_fwd(engine_handle_, stats, input.data_as<void>(), weights_.data_as<void>(),
                     use_bias_ ? bias_.data_as<void>() : nullptr, output.data_as<void>(),
                     ws.data_as<void>(), type_desc);

  return output;
}

Tensor DenseImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  if (grad_output.shape().back() != output_features_) {
    throw std::invalid_argument("Gradient feature size mismatch in DenseImpl.");
  }

  const Tensor &input = residuals["input"];
  const Vec<size_t> &in_shape = input.shape();
  size_t batch_size = 1;
  for (size_t i = 0; i < in_shape.size() - 1; ++i) {
    batch_size *= in_shape[i];
  }

  DenseStats stats{
      .batch_size = batch_size,
      .in_features = input_features_,
      .out_features = output_features_,
  };

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  Tensor grad_input = make_tensor(in_shape, io_dtype_);
  WorkspaceReq ws_req = engine_->query_dense_graph(engine_handle_, stats, type_desc);
  Tensor ws = make_tensor({ws_req.bwd_workspace}, DType_t::BYTE);

  engine_->dense_wgrad(engine_handle_, stats, grad_output.data_as<void>(), input.data_as<void>(),
                       weights_.grad_as<void>(), ws.data_as<void>(), type_desc);

  if (use_bias_) {
    engine_->dense_bgrad(engine_handle_, stats, grad_output.data_as<void>(), bias_.grad_as<void>(),
                         ws.data_as<void>(), type_desc);
  }

  engine_->dense_dgrad(engine_handle_, stats, grad_output.data_as<void>(), weights_.data_as<void>(),
                       grad_input.data_as<void>(), ws.data_as<void>(), type_desc);

  return grad_input;
}

LayerConfig DenseImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("input_features", input_features_);
  config.set("output_features", output_features_);
  config.set("use_bias", use_bias_);
  return config;
}

Vec<size_t> DenseImpl::compute_output_shape(const Vec<size_t> &input_shape) const {
  if (input_shape.empty()) {
    throw std::runtime_error("DenseImpl::compute_output_shape: Input shape is empty.");
  }
  Vec<size_t> out_shape = input_shape;
  out_shape.back() = output_features_;
  return out_shape;
}

std::shared_ptr<DenseImpl> DenseImpl::create_from_config(const LayerConfig &config) {
  size_t input_features = config.get<size_t>("input_features");
  size_t output_features = config.get<size_t>("output_features");
  bool use_bias = config.get<bool>("use_bias");

  return std::make_shared<DenseImpl>(input_features, output_features, use_bias, config.name);
}

}  // namespace internal
}  // namespace tunx
