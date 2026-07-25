#include <gtest/gtest.h>

#include "device/device_manager.hpp"
#include "device/pool_allocator.hpp"
#include "nn/graph.hpp"
#include "nn/layer_factory.hpp"
#include "nn/layers_impl/dense.hpp"
#include "tensor/ops.hpp"
#include "tensor_test_utils.hpp"
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

  void SetUp() override {}

  static bool has_gpu_;
  static sref<Device> device_;
  static stream stream_;
};

bool BF16Test::has_gpu_ = false;
sref<Device> BF16Test::device_;
stream BF16Test::stream_ = nullptr;

TEST_F(BF16Test, Dense) {
  auto &allocator = PoolAllocator::instance(device_, stream_);

  Graph fp32_graph;
  auto fp32_dense = Dense(128, 64, false, "fp32_dense");
  {
    Node fp32_input = fp32_graph.input("input");
    Node fp32_output = fp32_dense(fp32_input);
    GraphOpts fp32_opts{
        .io_dtype = DType_t::FP32,
        .param_dtype = DType_t::FP32,
        .compute_dtype = DType_t::FP32,
    };
    fp32_graph.set_output(fp32_output);
    fp32_graph.compile(allocator, fp32_opts);
  }

  Graph bf16_graph;
  auto bf16_dense = Dense(128, 64, false, "bf16_dense");
  {
    Node bf16_input = bf16_graph.input("input");
    Node bf16_output = bf16_dense(bf16_input);
    GraphOpts bf16_opts{
        .io_dtype = DType_t::BF16,
        .param_dtype = DType_t::BF16,
        .compute_dtype = DType_t::FP32,
    };
    bf16_graph.set_output(bf16_output);
    bf16_graph.compile(allocator, bf16_opts);
  }

  auto bf16_params = bf16_dense.params();
  auto fp32_params = fp32_dense.params();
  for (size_t i = 0; i < bf16_params.size(); ++i) {
    Tensor cpu_bf16_param = to_host(bf16_params[i].data());
    Tensor cpu_fp32_param = to_host(fp32_params[i].data());
    bf16 *bf16_data = cpu_bf16_param.data_as<bf16>();
    float *fp32_data = cpu_fp32_param.data_as<float>();
    for (size_t j = 0; j < cpu_bf16_param.size(); ++j) {
      fp32_data[j] = static_cast<float>(bf16_data[j]);
    }
    copy(cpu_fp32_param, fp32_params[i].data());
  }

  Tensor bf16_input = Tensor({32, 128}, DType_t::BF16, getHost());
  fill_normal(bf16_input, 0.0f, 1.0f, 12345ULL);
  Tensor fp32_input = Tensor({32, 128}, DType_t::FP32, getHost());

  bf16 *input_data = bf16_input.data_as<bf16>();
  float *input_data_fp32 = fp32_input.data_as<float>();
  for (size_t i = 0; i < bf16_input.size(); ++i) {
    input_data_fp32[i] = static_cast<float>(input_data[i]);
  }

  Tensor device_fp32_input = to_device(fp32_input, device_);
  Tensor device_bf16_input = to_device(bf16_input, device_);

  Tensor output_fp32 = fp32_dense.forward({device_fp32_input})[0];
  Tensor output_bf16 = bf16_dense.forward({device_bf16_input})[0];

  Tensor cpu_output_fp32 = to_host(output_fp32);
  Tensor cpu_output_bf16 = to_host(output_bf16);

  compare_tensor(cpu_output_bf16, cpu_output_fp32, 1e-2);
}
