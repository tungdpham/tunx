/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/leaky_relu.hpp"

#include <stdexcept>

#include "nn/activations_impl/leaky_relu.hpp"

namespace tunx {

Vec<Vec<size_t>> LeakyReLUOp::output_shapes(const Vec<Vec<size_t>> &input_shapes,
                                            const Config &config) {
  if (input_shapes.size() != 1) {
    throw std::runtime_error("LeakyReLUOp: expected exactly 1 input");
  }
  return {input_shapes[0]};
}

Tensor LeakyReLUOp::forward(OpContext &ctx, const Tensor &input, const Config &config) {
  Tensor output = ctx.make_tensor(input.shape(), ctx.io_dtype);

  if (ctx.is_training) {
    Tensor mask = ctx.make_tensor(input.shape(), DType_t::UINT8_T);
    ctx.residuals["mask"] = mask;

    func::LeakyReLU(config.negative_slope).apply(input, output);

    size_t num_elements = input.size();
    if (input.device_type() == DeviceType::CPU) {
      const float *input_data = input.data_as<float>();
      uint8_t *mask_data = mask.data_as<uint8_t>();
      for (size_t i = 0; i < num_elements; ++i) {
        mask_data[i] = (input_data[i] > 0.0f) ? 1 : 0;
      }
    }
#ifdef TUNX_USE_CUDA
    else if (input.device_type() == DeviceType::CUDA) {
      throw std::runtime_error("LeakyReLUOp: CUDA mask computation not yet implemented");
    }
#endif
  } else {
    func::LeakyReLU(config.negative_slope).apply(input, output);
  }

  return output;
}

Tensor LeakyReLUOp::backward(OpContext &ctx, const Tensor &grad_output, const Config &config) {
  const Tensor &mask = ctx.residuals["mask"];

  Tensor grad_input = ctx.make_tensor(grad_output.shape(), ctx.io_dtype);

  // Gradient: grad_input = grad_output * (mask ? 1.0 : negative_slope)
  size_t num_elements = grad_output.size();
  if (grad_output.device_type() == DeviceType::CPU) {
    const float *grad_out_data = grad_output.data_as<float>();
    const uint8_t *mask_data = mask.data_as<uint8_t>();
    float *grad_in_data = grad_input.data_as<float>();
    for (size_t i = 0; i < num_elements; ++i) {
      float slope = mask_data[i] ? 1.0f : config.negative_slope;
      grad_in_data[i] = grad_out_data[i] * slope;
    }
  }
#ifdef TUNX_USE_CUDA
  else if (grad_output.device_type() == DeviceType::CUDA) {
    throw std::runtime_error("LeakyReLUOp: CUDA backward not yet implemented");
  }
#endif

  return grad_input;
}

LayerConfig LeakyReLUOp::get_config(const Config &config, const std::string &name) {
  LayerConfig lcfg;
  lcfg.type = TYPE_NAME;
  lcfg.name = name;
  lcfg.set("negative_slope", config.negative_slope);
  return lcfg;
}

}  // namespace tunx
