#include <gtest/gtest.h>

#include "device/device_manager.hpp"
#include "device/pool_allocator.hpp"
#include "nn/graph.hpp"
#include "nn/layer_factory.hpp"
#include "nn/layers_impl/dense.hpp"
#include "tensor/tensor_ops.hpp"
#include "tensor_test_utils.hpp"
#include "type/type.hpp"

using namespace std;
using namespace tunx;

class FP16Test : public ::testing::Test {
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

bool FP16Test::has_gpu_ = false;
sref<Device> FP16Test::device_;
stream FP16Test::stream_ = nullptr;

TEST_F(FP16Test, Dense) {
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

  Graph fp16_graph;
  auto fp16_dense = Dense(128, 64, false, "fp16_dense");
  {
    Node fp16_input = fp16_graph.input("input");
    Node fp16_output = fp16_dense(fp16_input);

    GraphOpts fp16_opts{
        .io_dtype = DType_t::FP16,
        .param_dtype = DType_t::FP16,
        .compute_dtype = DType_t::FP32,
    };
    fp16_graph.set_output(fp16_output);
    fp16_graph.compile(allocator, fp16_opts);
  }

  auto fp16_params = fp16_dense.params();
  auto fp32_params = fp32_dense.params();
  for (size_t i = 0; i < fp16_params.size(); ++i) {
    Tensor cpu_fp16_param = fp16_params[i].data().to_host();
    Tensor cpu_fp32_param = fp32_params[i].data().to_host();
    fp16 *fp16_data = cpu_fp16_param.data_as<fp16>();
    float *fp32_data = cpu_fp32_param.data_as<float>();
    for (size_t j = 0; j < cpu_fp16_param.size(); ++j) {
      fp32_data[j] = static_cast<float>(fp16_data[j]);
    }
    cpu_fp32_param.copy_to(fp32_params[i].data(), stream_);
  }

  Tensor fp16_input = Tensor({32, 128}, DType_t::FP16, getHost());
  fill_normal(fp16_input, 0.0f, 1.0f);
  Tensor fp32_input = Tensor({32, 128}, DType_t::FP32, getHost());

  fp16 *input_data = fp16_input.data_as<fp16>();
  float *input_data_fp32 = fp32_input.data_as<float>();
  for (size_t i = 0; i < fp16_input.size(); ++i) {
    input_data_fp32[i] = static_cast<float>(input_data[i]);
  }

  Tensor device_fp32_input = fp32_input.to_device(device_);
  Tensor device_fp16_input = fp16_input.to_device(device_);

  Tensor output_fp32 = fp32_dense.forward({device_fp32_input})[0];
  Tensor output_fp16 = fp16_dense.forward({device_fp16_input})[0];

  Tensor host_output_fp32 = output_fp32.to_host();
  Tensor host_output_fp16 = output_fp16.to_host();

  stream_.sync();

  compare_tensor(host_output_fp16, host_output_fp32);
}
