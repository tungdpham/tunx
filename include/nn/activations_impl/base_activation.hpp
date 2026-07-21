/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>

#include "tensor/tensor.hpp"

namespace tunx {
class ActivationFunction {
public:
  virtual ~ActivationFunction() = default;

  virtual void apply(const Tensor &input, Tensor &output, stream s = nullptr) const = 0;

  virtual void compute_gradient(const Tensor &input, const Tensor &grad_output, Tensor &grad_input,
                                stream s = nullptr) const = 0;

  virtual std::string name() const = 0;
  virtual std::unique_ptr<ActivationFunction> clone() const = 0;
};
}  // namespace tunx