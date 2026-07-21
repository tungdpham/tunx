#include <gtest/gtest.h>

#include <cstddef>

#include "device/device.hpp"
#include "device/device_manager.hpp"
#include "device/pool_allocator.hpp"
#include "nn/example_graphs.hpp"
#include "nn/graph.hpp"
#include "nn/layer_factory.hpp"
#include "nn/layers_impl/dense.hpp"
#include "nn/loss.hpp"
#include "type/type.hpp"

using namespace std;
using namespace tunx;

class BF16Test : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    initializeDefaultDevices();
    DeviceManager &manager = DeviceManager::instance();
    Vec<DeviceID> device_ids = manager.get_all();

    has_gpu_ = false;
    for (const DeviceID &id : device_ids) {
      Device &device = manager.get(id);
      if (device.device_type() == DeviceType::CUDA) {
        has_gpu_ = true;
        device_ = device;
        break;
      }
    }

    if (!has_gpu_) {
      GTEST_SKIP() << "No CUDA device available, skipping CuDNN engine tests";
    }

    stream_ = device_->default_stream();
  }

  void SetUp() override { ExampleGraphs::register_defaults(); }

  static bool has_gpu_;
  static sref<Device> device_;
  static stream stream_;
};

bool BF16Test::has_gpu_ = false;
sref<Device> BF16Test::device_;
stream BF16Test::stream_ = nullptr;

TEST_F(BF16Test, Dense) {
  constexpr size_t batch_size = 8;
  constexpr size_t input_dim = 32;
  constexpr size_t output_dim = 16;
  auto &allocator = PoolAllocator::instance(device_, stream_);

  auto fp32_dense_layer = Dense(input_dim, output_dim, false, "fp32_dense");
  fp32_dense_layer.set_io_dtype(DType_t::FP32);

  auto bf16_dense_layer = Dense(input_dim, output_dim, false, "bf16_dense");
  bf16_dense_layer.set_io_dtype(DType_t::BF16);
  bf16_dense_layer.set_param_dtype(DType_t::BF16);

  Graph graph;
  Node input = graph.make_node("input");
  Node fp32_output = fp32_dense_layer(input);
  fp32_output->set_uid("fp32_output");
  Node bf16_output = bf16_dense_layer(input);
  bf16_output->set_uid("bf16_output");
  graph.compile(allocator);

  auto bf16_params = bf16_dense_layer.params();
  auto fp32_params = fp32_dense_layer.params();
  for (size_t i = 0; i < bf16_params.size(); ++i) {
    bf16_params[i].data().copy_to(fp32_params[i].data());
  }

  Tensor bf16_input = Tensor({batch_size, input_dim}, DType_t::BF16, getHost());
  fill_normal(bf16_input, 0.0f, 1.0f);
  Tensor fp32_input = Tensor({batch_size, input_dim}, DType_t::FP32, getHost());

  bf16 *input_data = bf16_input.data_as<bf16>();
  float *input_data_fp32 = fp32_input.data_as<float>();
  for (size_t i = 0; i < bf16_input.size(); ++i) {
    input_data_fp32[i] = static_cast<float>(input_data[i]);
  }

  Tensor input_fp32 = fp32_input.to_device(device_);
  Tensor input_bf16 = bf16_input.to_device(device_);

  Residuals fp32_residuals, bf16_residuals;
  Tensor output_fp32 = fp32_dense_layer.forward({input_fp32}, fp32_residuals)[0];
  Tensor output_bf16 = bf16_dense_layer.forward({input_bf16}, bf16_residuals)[0];

  Tensor cpu_output_fp32 = output_fp32.to_host();
  Tensor cpu_output_bf16 = output_bf16.to_host();

  float *output_data_fp32 = cpu_output_fp32.data_as<float>();
  bf16 *output_data_bf16 = cpu_output_bf16.data_as<bf16>();
  constexpr double tolerance = 2e-3;
  for (size_t i = 0; i < cpu_output_fp32.size(); ++i) {
    EXPECT_NEAR(static_cast<double>(output_data_fp32[i]), static_cast<double>(output_data_bf16[i]),
                tolerance)
        << "At index " << i;
  }

  Tensor target_fp32 = Tensor({batch_size, output_dim}, DType_t::FP32);
  Tensor target_bf16 = Tensor({batch_size, output_dim}, DType_t::BF16);
  fill(target_fp32, 0.0f);
  fill(target_bf16, bf16(0.0f));

  for (size_t i = 0; i < batch_size; ++i) {
    target_fp32.at<float>({i, i % output_dim}) = 1.0f;
    target_bf16.at<bf16>({i, i % output_dim}) = bf16(1.0f);
  }

  auto criterion = LossFactory::create_crossentropy();

  auto gradient_fp32 = Tensor({batch_size, output_dim}, DType_t::FP32);
  auto gradient_bf16 = Tensor({batch_size, output_dim}, DType_t::BF16);

  criterion->compute_gradient(cpu_output_fp32, target_fp32, gradient_fp32);
  criterion->compute_gradient(cpu_output_bf16, target_bf16, gradient_bf16);

  auto gpu_gradient_fp32 = gradient_fp32.to_device(device_);
  auto gpu_gradient_bf16 = gradient_bf16.to_device(device_);

  Tensor grad_input_bf16 = bf16_dense_layer.backward({gpu_gradient_bf16}, bf16_residuals)[0];
  Tensor grad_input_fp32 = fp32_dense_layer.backward({gpu_gradient_fp32}, fp32_residuals)[0];

  Tensor cpu_grad_input_fp32 = grad_input_fp32.to_host();
  Tensor cpu_grad_input_bf16 = grad_input_bf16.to_host();
  float *grad_input_data_fp32 = cpu_grad_input_fp32.data_as<float>();
  bf16 *grad_input_data_bf16 = cpu_grad_input_bf16.data_as<bf16>();
  for (size_t i = 0; i < cpu_grad_input_fp32.size(); ++i) {
    EXPECT_NEAR(static_cast<double>(grad_input_data_fp32[i]),
                static_cast<double>(grad_input_data_bf16[i]), tolerance)
        << "At index " << i;
  }
}
