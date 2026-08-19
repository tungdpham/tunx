/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/sigmoid.hpp"

#include <stdexcept>

#include "nn/activations_impl/sigmoid.hpp"

namespace tunx {

Vec<Vec<size_t>> SigmoidOp::output_shapes(const Vec<Vec<size_t>> &input_shapes,
                                          const Config &config) {
  if (input_shapes.size() != 1) {
    throw std::runtime_error("SigmoidOp: expected exactly 1 input");
  }
  return {input_shapes[0]};
}

Tensor SigmoidOp::forward(OpContext &ctx, const Tensor &input) {
  Tensor output = ctx.make_tensor(input.shape(), ctx.io_dtype);
  func::Sigmoid().apply(input, output, ctx.handle.get_stream());

  if (ctx.is_training) {
    ctx.residuals["output"] = output;
  }

  return output;
}

Tensor SigmoidOp::backward(OpContext &ctx, const Tensor &grad_output) {
  const Tensor &output = ctx.residuals["output"];

  Tensor grad_input = ctx.make_tensor(grad_output.shape(), ctx.io_dtype);

  size_t num_elements = grad_output.size();
  if (grad_output.device_type() == DeviceType::CPU) {
    const float *grad_out_data = grad_output.data_as<float>();
    const float *output_data = output.data_as<float>();
    float *grad_in_data = grad_input.data_as<float>();
    for (size_t i = 0; i < num_elements; ++i) {
      float sig = output_data[i];
      grad_in_data[i] = grad_out_data[i] * sig * (1.0f - sig);
    }
  }
#ifdef TUNX_USE_CUDA
  else if (grad_output.device_type() == DeviceType::CUDA) {
    throw std::runtime_error("SigmoidOp: CUDA backward not yet implemented");
  }
#endif

  return grad_input;
}

LayerConfig SigmoidOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.type = TYPE_NAME;
  lcfg.name = name;
  return lcfg;
}

}  // namespace tunx
