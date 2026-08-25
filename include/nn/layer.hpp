/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <fmt/core.h>
#include <oneapi/tbb/profiling.h>

#include <cstddef>
#include <cstring>
#include <device/stream.hpp>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/config.hpp"
#include "device/del_allocator_v2.hpp"
#include "device/iallocator.hpp"
#include "nn/engine.hpp"
#include "nn/engines/engine_handle.hpp"
#include "nn/param.hpp"
#include "nn/residuals.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace tunx {
using LayerConfig = TConfig;

inline size_t get_shapes_bytes(const Vec<Vec<size_t>> &shapes, DType_t dtype) {
  size_t total_bytes = 0;
  size_t dtype_size = get_dtype_size(dtype);
  for (const auto &shape : shapes) {
    size_t shape_bytes =
        std::accumulate(shape.begin(), shape.end(), dtype_size, std::multiplies<size_t>());
    shape_bytes = align_up(shape_bytes, 256);
    total_bytes += shape_bytes;
  }
  return total_bytes;
}

struct InitOptions {
  IAllocator *ws_allocator = nullptr;
  Engine engine = nullptr;
  engine_handle handle = nullptr;
  unsigned long long seed;
  DType_t io_dtype = DType_t::FP32;
  DType_t param_dtype = DType_t::FP32;
  DType_t compute_dtype = DType_t::FP32;
};

class LayerImpl : public virtual std::enable_shared_from_this<LayerImpl> {
public:
  LayerImpl() = default;
  LayerImpl(const std::string &name)
      : name_(name) {}

  virtual ~LayerImpl() = default;

  void init(IAllocator &param_allocator, InitOptions opts = InitOptions{});

  Vec<Tensor> forward(const Vec<Tensor> &inputs);
  Vec<Tensor> forward(const Vec<Tensor> &inputs, Residuals &residuals);
  Vec<Tensor> backward(const Vec<Tensor> &grad_outputs, Residuals &residuals);

  // Note: have to call init again after changing param dtype
  Engine get_engine();
  engine_handle get_backend_handle() const;
  DType_t get_io_dtype() const;
  DType_t get_param_dtype() const;
  DType_t get_compute_dtype() const;
  void set_training(bool training);
  bool is_training() const;

  IAllocator *workspace_allocator();
  void set_workspace_allocator(IAllocator *alloc);

  virtual Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const = 0;
  std::string name() const { return name_; }
  void save_state(std::ostream &out) const;
  virtual std::string type() const = 0;
  virtual LayerConfig get_config() const = 0;

  virtual Vec<Param> params();
  virtual const Vec<Param> params() const;

  void register_layer(std::shared_ptr<LayerImpl> layer) {
    registered_layers_.push_back(std::move(layer));
  }
  const Vec<std::shared_ptr<LayerImpl>> &layers() const { return registered_layers_; }

  virtual void zero_grads() {
    for (auto &param : params_) {
      param.zero_grad(engine_handle_.get_stream());
    }
    for (auto &layer : registered_layers_) {
      layer->zero_grads();
    }
  }

  Device &device() const {
    if (!param_allocator_) {
      throw std::runtime_error("LayerImpl: Param allocator is not set to get device.");
    }
    return param_allocator_->device();
  }

protected:
  virtual void init_impl() {}
  virtual void on_set_training(bool training) {}
  virtual Vec<Tensor> forward_impl(const Vec<Tensor> &inputs, Residuals &residuals) = 0;
  virtual Vec<Tensor> backward_impl(const Vec<Tensor> &grad_outputs, Residuals &residuals) = 0;

protected:
  bool initialized_ = false;
  Engine engine_ = nullptr;
  engine_handle engine_handle_ = nullptr;
  IAllocator *param_allocator_ = nullptr;
  IAllocator *ws_allocator_ = nullptr;
  bool is_training_ = true;
  bool use_seed_ = false;
  unsigned long long srand_seed_ = 0;
  std::string name_;
  Vec<Param> params_;
  Vec<std::shared_ptr<LayerImpl>> registered_layers_;
  DType_t io_dtype_ = DType_t::FP32;       // data type for input/output tensors
  DType_t param_dtype_ = DType_t::FP32;    // data type for parameters/gradients
  DType_t compute_dtype_ = DType_t::FP32;  // data type for internal computations

public:
  // helpers
  Param make_param(const Vec<size_t> &shape, DType_t dtype);
  Tensor make_tensor(const Vec<size_t> &shape, DType_t dtype);
};

template <typename LayerType>
class LayerRef {
public:
  using impl_type = LayerType;

  template <typename>
  friend class LayerRef;

  LayerRef() = default;

  LayerRef(std::nullptr_t)
      : impl_(nullptr) {}

  LayerRef(std::shared_ptr<LayerType> layer)
      : impl_(layer) {}

  template <typename U, typename = std::enable_if_t<std::is_convertible_v<U *, LayerType *>>>
  LayerRef(std::shared_ptr<U> layer)
      : impl_(std::move(layer)) {}

  template <typename U, typename = std::enable_if_t<std::is_convertible_v<U *, LayerType *>>>
  LayerRef(const LayerRef<U> &other)
      : impl_(std::static_pointer_cast<LayerType>(other.impl_)) {}

  template <typename U, typename = std::enable_if_t<std::is_convertible_v<U *, LayerType *>>>
  LayerRef(LayerRef<U> &&other)
      : impl_(std::static_pointer_cast<LayerType>(std::move(other.impl_))) {}

  template <typename T, typename U = typename std::decay_t<T>::impl_type,
            std::enable_if_t<std::is_convertible_v<U *, LayerType *> &&
                                 std::is_base_of_v<LayerRef<U>, std::decay_t<T>> &&
                                 !std::is_same_v<std::decay_t<T>, LayerRef<U>>,
                             int> = 0>
  LayerRef(T &&other)
      : impl_(std::static_pointer_cast<LayerType>(static_cast<const LayerRef<U> &>(other).impl_)) {}

  template <typename... Args>
  LayerRef(Args &&...args)
      : impl_(std::make_shared<LayerType>(std::forward<Args>(args)...)) {}

  LayerType &operator*() const { return *impl_; }

  operator std::shared_ptr<LayerType>() const { return impl_; }
  LayerType *get() const { return impl_.get(); }
  LayerType *release() { return impl_.release(); }

  explicit operator bool() const { return impl_ != nullptr; }
  bool operator!() const { return impl_ == nullptr; }

  bool operator==(const LayerRef &other) const { return impl_ == other.impl_; }
  bool operator!=(const LayerRef &other) const { return impl_ != other.impl_; }

  template <typename U>
  bool is() const {
    return std::dynamic_pointer_cast<U>(impl_) != nullptr;
  }

  template <typename U>
  auto as() const -> LayerRef<U> {
    auto casted = std::dynamic_pointer_cast<U>(impl_);
    if (!casted) {
      throw std::runtime_error("LayerRef: incompatible layer cast");
    }
    return LayerRef<U>(std::move(casted));
  }

  template <typename... Args>
  decltype(auto) operator()(Args &&...args) const {
    check_layer("operator()");
    return (*impl_)(std::forward<Args>(args)...);
  }

  void init(IAllocator &allocator, InitOptions init_options) {
    check_layer("init");
    impl_->init(allocator, init_options);
  }

  Vec<Tensor> forward(const Vec<Tensor> &inputs) {
    check_layer("forward");
    return impl_->forward(inputs);
  }

  Vec<Tensor> forward(const Vec<Tensor> &inputs, Residuals &residuals) {
    check_layer("forward");
    return impl_->forward(inputs, residuals);
  }

  Vec<Tensor> backward(const Vec<Tensor> &grad_outputs, Residuals &residuals) {
    check_layer("backward");
    return impl_->backward(grad_outputs, residuals);
  }

  IAllocator *get_allocator() const {
    check_layer("get_allocator");
    return impl_->get_allocator();
  }

  DType_t get_io_dtype() const {
    check_layer("get_io_dtype");
    return impl_->get_io_dtype();
  }

  DType_t get_param_dtype() const {
    check_layer("get_param_dtype");
    return impl_->get_param_dtype();
  }

  DType_t get_compute_dtype() const {
    check_layer("get_compute_dtype");
    return impl_->get_compute_dtype();
  }

  LayerRef &set_training(bool training) {
    check_layer("set_training");
    impl_->set_training(training);
    return *this;
  }

  bool is_training() const {
    check_layer("is_training");
    return impl_->is_training();
  }

  LayerRef &set_workspace_allocator(IAllocator *alloc) {
    check_layer("set_workspace_allocator");
    impl_->set_workspace_allocator(alloc);
    return *this;
  }

  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const {
    check_layer("output_shapes");
    return impl_->output_shapes(input_shapes);
  }

  std::string name() const {
    check_layer("name");
    return impl_->name();
  }

  void save_state(std::ofstream &file) {
    check_layer("save_state");
    impl_->save_state(file);
  }

  std::string type() const {
    check_layer("type");
    return impl_->type();
  }

  LayerConfig get_config() const {
    check_layer("get_config");
    return impl_->get_config();
  }

  Vec<Param> params() {
    check_layer("params");
    return impl_->params();
  }

  const Vec<Param> params() const {
    check_layer("params");
    return impl_->params();
  }

  Device &device() const {
    check_layer("device");
    return impl_->device();
  }

  static LayerRef<LayerType> create_from_config(const LayerConfig &config) {
    std::shared_ptr<LayerType> layer = LayerType::create_from_config(config);
    if (!layer) {
      throw std::runtime_error("LayerRef: create_from_config returned nullptr");
    }
    return LayerRef<LayerType>(std::move(layer));
  }

protected:
  std::shared_ptr<LayerType> impl_;

private:
  void check_layer(const char *method_name) const {
    if (!impl_) {
      throw std::runtime_error(
          fmt::format("LayerRef {}: underlying shared_ptr is null", method_name));
    }
  }
};

class Layer : public LayerRef<LayerImpl> {
public:
  using LayerRef<LayerImpl>::LayerRef;
};

template <typename LayerType, typename... Args>
auto make_layer(Args &&...args) -> LayerRef<LayerType> {
  return LayerRef<LayerType>(std::make_shared<LayerType>(std::forward<Args>(args)...));
}

}  // namespace tunx