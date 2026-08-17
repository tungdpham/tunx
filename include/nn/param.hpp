#pragma once

#include <cassert>
#include <memory>
#include <stdexcept>

#include "device/iallocator.hpp"
#include "device/stream.hpp"
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace tunx {
class LayerImpl;

class Param {
private:
  friend class LayerImpl;

  struct Impl {
    Tensor data_;
    Tensor grad_;
    bool requires_grad_ = true;

    Impl(const Vec<size_t> &shape, DType_t dtype, IAllocator &allocator) {
      data_ = Tensor(shape, dtype, allocator);
      grad_ = Tensor(shape, dtype, allocator);
      if (grad_.size() > 0) {
        fill(grad_, 0.0);
      }
    }
  };

  std::shared_ptr<Impl> impl_;

  void check_valid(const std::string method_name = "") const {
    if (!impl_) {
      throw std::runtime_error("Param " + method_name +
                               " error: Accessing an uninitialized Param handle.");
    }
  }

  Param(const Vec<size_t> &shape, DType_t dtype, IAllocator &allocator)
      : impl_(std::make_shared<Impl>(shape, dtype, allocator)) {}

public:
  Param() = default;

  explicit operator bool() const { return impl_ != nullptr; }
  bool is_null() const { return impl_ == nullptr; }

  template <typename T = void>
  T *data_as() {
    check_valid("data_as");
    return impl_->data_.data_as<T>();
  }

  template <typename T = void>
  const T *data_as() const {
    check_valid("data_as");
    return impl_->data_.data_as<T>();
  }

  template <typename T = void>
  T *grad_as() {
    check_valid("grad_as");
    return impl_->grad_.data_as<T>();
  }

  template <typename T = void>
  const T *grad_as() const {
    check_valid("grad_as");
    return impl_->grad_.data_as<T>();
  }

  Tensor &data() {
    check_valid("data");
    return impl_->data_;
  }

  const Tensor &data() const {
    check_valid("data");
    return impl_->data_;
  }

  Tensor &grad() {
    check_valid("grad");
    return impl_->grad_;
  }

  const Tensor &grad() const {
    check_valid("grad");
    return impl_->grad_;
  }

  Vec<size_t> shape() const {
    check_valid("shape");
    return impl_->data_.shape();
  }

  DType_t dtype() const {
    check_valid("dtype");
    return impl_->data_.dtype();
  }

  size_t size() const {
    check_valid("size");
    return impl_->data_.size();
  }

  Device &device() const {
    check_valid("device");
    return impl_->data_.device();
  }

  void zero_grad(stream s = nullptr) {
    check_valid("zero_grad");
    fill(impl_->grad_, 0.0, s);
  }

  bool is_same(const Param &other) const { return impl_ == other.impl_; }

  bool requires_grad() const {
    check_valid("requires_grad");
    return impl_->requires_grad_;
  }

  void set_requires_grad(bool val) {
    check_valid("set_requires_grad");
    impl_->requires_grad_ = val;
  }
};

}  // namespace tunx