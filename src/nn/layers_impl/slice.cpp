/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/slice.hpp"

#include <stdexcept>

namespace tunx {
namespace internal {

SliceImpl::SliceImpl(size_t axis, size_t start, size_t length, const std::string &name)
    : SISOLayerImpl(name),
      axis_(axis),
      start_(start),
      length_(length) {}

Tensor SliceImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  Tensor shape_tensor = Tensor({input.shape().size()});
  std::copy(input.shape().begin(), input.shape().end(), shape_tensor.data_as<size_t>());
  residuals["original_shape"] = shape_tensor;

  Vec<size_t> output_shape = compute_output_shape(input.shape());
  Tensor output = make_tensor(output_shape, io_dtype_);

  // TODO: implement slice forward in engines.
  throw std::runtime_error("SliceImpl: unimplemented");

  return output;
}

Tensor SliceImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  const Tensor &shape_tensor = residuals["original_shape"];
  Vec<size_t> original_shape(shape_tensor.size());
  std::copy(shape_tensor.data_as<size_t>(), shape_tensor.data_as<size_t>() + shape_tensor.size(),
            original_shape.begin());

  Tensor grad_input = make_tensor(original_shape, io_dtype_);

  // TODO: implement slice backward in engines.
  throw std::runtime_error("SliceImpl: unimplemented");

  return grad_input;
}

Vec<size_t> SliceImpl::compute_output_shape(const Vec<size_t> &input_shape) const {
  if (axis_ >= input_shape.size()) {
    throw std::invalid_argument("Slice axis out of bounds");
  }
  if (start_ + length_ > input_shape[axis_]) {
    throw std::invalid_argument("Slice range out of bounds");
  }

  Vec<size_t> output_shape = input_shape;
  output_shape[axis_] = length_;
  return output_shape;
}

LayerConfig SliceImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("axis", (int)axis_);
  config.set("start", (int)start_);
  config.set("length", (int)length_);
  return config;
}

std::shared_ptr<SliceImpl> SliceImpl::create_from_config(const LayerConfig &config) {
  size_t axis = (size_t)config.get<int>("axis", 0);
  size_t start = (size_t)config.get<int>("start", 0);
  size_t length = (size_t)config.get<int>("length", 1);
  return std::make_shared<SliceImpl>(axis, start, length, config.name);
}

}  // namespace internal
}  // namespace tunx
