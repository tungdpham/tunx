/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/relu.hpp"

#include <memory>
#include <stdexcept>

#include "nn/engines/iengine.hpp"
#include "type/type.hpp"

namespace tunx {
namespace internal {

ReLUImpl::ReLUImpl(const std::string &name)
    : SISOLayerImpl(name) {}

Tensor ReLUImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  Tensor output = make_tensor(input.shape(), io_dtype_);

  size_t batch_size = input.dims() > 0 ? input.dim(0) : 1;
  size_t spatial_size = input.dims() > 0 ? input.stride(0) : 1;

  ReLUStats stats{
      .batch_size = batch_size,
      .spatial_size = spatial_size,
  };

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  WorkspaceReq ws_req = engine_->query_relu_graph(engine_handle_, stats, type_desc);
  Tensor ws = make_tensor({ws_req.fwd_workspace}, DType_t::BYTE);

  if (this->is_training_) {
    residuals["output"] = output;

    engine_->relu_fwd(engine_handle_, stats, input.data_as<void>(), output.data_as<void>(),
                      ws.data_as<void>(), type_desc);
  } else {
    engine_->relu_inf(engine_handle_, stats, input.data_as<void>(), output.data_as<void>(),
                      ws.data_as<void>(), type_desc);
  }

  return output;
}

Tensor ReLUImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  Tensor &output = residuals["output"];
  if (!output) {
    throw std::runtime_error("No cached output found for backward pass in ReLUImpl");
  }

  Tensor grad_input = make_tensor(grad_output.shape(), io_dtype_);

  size_t batch_size = grad_output.dims() > 0 ? grad_output.dim(0) : 1;
  size_t spatial_size = grad_output.dims() > 0 ? grad_output.stride(0) : 1;

  ReLUStats stats{
      .batch_size = batch_size,
      .spatial_size = spatial_size,
  };

  DTypeDesc type_desc{
      .io_dtype = io_dtype_,
      .param_dtype = param_dtype_,
      .compute_dtype = compute_dtype_,
  };

  WorkspaceReq ws_req = engine_->query_relu_graph(engine_handle_, stats, type_desc);
  Tensor ws = make_tensor({ws_req.bwd_workspace}, DType_t::BYTE);

  engine_->relu_bwd(engine_handle_, stats, grad_output.data_as<void>(), grad_input.data_as<void>(),
                    output.data_as<void>(), ws.data_as<void>(), type_desc);

  return grad_input;
}

LayerConfig ReLUImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  return config;
}

std::shared_ptr<ReLUImpl> ReLUImpl::create_from_config(const LayerConfig &config) {
  return std::make_shared<ReLUImpl>(config.name);
}

}  // namespace internal
}  // namespace tunx
