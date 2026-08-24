/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/layer.hpp"

#include <fmt/ranges.h>

#include "device/del_allocator_v2.hpp"
#include "device/iallocator.hpp"
#include "device/stream.hpp"
#include "nn/engines/engine_handle.hpp"
#include "nn/param.hpp"
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace tunx {

void LayerImpl::init(IAllocator &param_allocator, InitOptions opts) {
  if (initialized_) {
    throw std::runtime_error("Cannot initalize LayerImpl more than once. ");
  }
  param_allocator_ = &param_allocator;
  engine_ = opts.engine;
  engine_handle_ = opts.handle;
  if (!opts.ws_allocator) {
    ws_allocator_ =
        DELAllocatorV2::instance(param_allocator.device(), engine_handle_.get_stream()).get();
  } else {
    ws_allocator_ = opts.ws_allocator;
  }
  io_dtype_ = opts.io_dtype;
  param_dtype_ = opts.param_dtype;
  compute_dtype_ = opts.compute_dtype;
  if (opts.seed) {
    use_seed_ = true;
    srand_seed_ = opts.seed;
  }
  init_impl();
  for (auto &layer : registered_layers_) {
    layer->init(param_allocator, opts);
  }
  initialized_ = true;
}

Vec<Tensor> LayerImpl::forward(const Vec<Tensor> &inputs) {
  Residuals dummy_residuals;
  return forward(inputs, dummy_residuals);
}

Vec<Tensor> LayerImpl::forward(const Vec<Tensor> &inputs, Residuals &residuals) {
  if (!initialized_) {
    throw std::runtime_error("LayerImpl must be initialized before calling forward");
  }
  Vec<Tensor> device_inputs;
  for (auto &input : inputs) {
    if (input.device() == this->device())
      device_inputs.push_back(input);
    else {
      Tensor device_input = to_device(input, device());
      device_inputs.push_back(device_input);
    }
  }
  Vec<Tensor> outputs = forward_impl(device_inputs, residuals);
#ifndef NDEBUG
  engine_handle_.get_stream().sync();
#endif
  return outputs;
}

Vec<Tensor> LayerImpl::backward(const Vec<Tensor> &grad_outputs, Residuals &residuals) {
  if (!initialized_) {
    throw std::runtime_error("LayerImpl must be initialized before calling backward");
  }
  Vec<Tensor> device_grad_outputs;
  for (auto &grad : grad_outputs) {
    if (grad.device() == this->device())
      device_grad_outputs.push_back(grad);
    else {
      Tensor device_grad = to_device(grad, device());
      device_grad_outputs.push_back(device_grad);
    }
  }
  auto grad_inputs = backward_impl(device_grad_outputs, residuals);
#ifndef NDEBUG
  engine_handle_.get_stream().sync();
#endif
  return grad_inputs;
}

DType_t LayerImpl::get_io_dtype() const { return io_dtype_; }

DType_t LayerImpl::get_param_dtype() const { return param_dtype_; }

DType_t LayerImpl::get_compute_dtype() const { return compute_dtype_; }

void LayerImpl::set_training(bool training) {
  is_training_ = training;
  on_set_training(training);
  for (auto &layer : registered_layers_) {
    layer->set_training(training);
  }
}

bool LayerImpl::is_training() const { return is_training_; }

IAllocator *LayerImpl::workspace_allocator() { return ws_allocator_; }

void LayerImpl::set_workspace_allocator(IAllocator *alloc) {
  ws_allocator_ = alloc;
  for (auto &layer : registered_layers_) {
    layer->set_workspace_allocator(alloc);
  }
}

Engine LayerImpl::get_engine() {
  if (!engine_) {
    throw std::runtime_error("Engine is not set");
  }
  return engine_;
}

engine_handle LayerImpl::get_backend_handle() const { return engine_handle_; }

Vec<Param> LayerImpl::params() {
  Vec<Param> all_params = params_;
  for (auto &layer : registered_layers_) {
    auto layer_params = layer->params();
    all_params.insert(all_params.end(), layer_params.begin(), layer_params.end());
  }
  return all_params;
}

const Vec<Param> LayerImpl::params() const {
  Vec<Param> all_params = params_;
  for (const auto &layer : registered_layers_) {
    auto layer_params = layer->params();
    all_params.insert(all_params.end(), layer_params.begin(), layer_params.end());
  }
  return all_params;
}

void LayerImpl::save_state(std::ostream &out) const {
  auto config = get_config();
  nlohmann::json j = config.to_json();
  std::string j_str = j.dump();
  size_t j_size = j_str.size();
  out.write(reinterpret_cast<const char *>(&j_size), sizeof(size_t));
  out.write(j_str.c_str(), j_size);
  Vec<Param> parameters = this->params();
  for (const auto &param : parameters) {
    save(param.data(), out);
  }
}

Param LayerImpl::make_param(const Vec<size_t> &shape, DType_t dtype) {
  if (!param_allocator_) {
    throw std::runtime_error("LayerImpl::make_param: Param allocator is not set");
  }
  Param param(shape, dtype, *param_allocator_);
  params_.push_back(param);
  return param;
}

Tensor LayerImpl::make_tensor(const Vec<size_t> &shape, DType_t dtype) {
  if (!ws_allocator_) {
    throw std::runtime_error("LayerImpl::make_tensor: Workspace allocator is not set");
  }
  return Tensor(shape, dtype, *ws_allocator_);
}

}  // namespace tunx
