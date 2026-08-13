/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "device/stream.hpp"
#include "nn/activations_impl/base_activation.hpp"
#include "tensor/tensor.hpp"

namespace tunx {
namespace func {
class ReLU : public ActivationFunction {
public:
  explicit ReLU();

  void apply(const Tensor &input, Tensor &output, stream s = nullptr) const override;

  void compute_gradient(const Tensor &input, const Tensor &grad_output, Tensor &grad_input,
                        stream s = nullptr) const override;

  std::string name() const override;
  std::unique_ptr<ActivationFunction> clone() const override;

};

}  // namespace func
}  // namespace tunx
