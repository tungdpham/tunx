#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>

#include "device/device.hpp"
#include "device/device_manager.hpp"
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"

using namespace tunx;

class GPUTensorTest : public ::testing::Test {
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

  void SetUp() override {
    small_tensor = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);
    fill(small_tensor, 1.0);

    large_tensor = Tensor({2, 3, 4, 4}, DType_t::FP32, device_);
    fill(large_tensor, 2.0);
  }

  void TearDown() override {
    small_tensor = Tensor();
    large_tensor = Tensor();
  }

  static void TearDownTestSuite() {}

  static bool has_gpu_;
  static sref<Device> device_;
  static stream stream_;
  Tensor small_tensor;
  Tensor large_tensor;
};

bool GPUTensorTest::has_gpu_ = false;
sref<Device> GPUTensorTest::device_;
stream GPUTensorTest::stream_ = nullptr;

TEST_F(GPUTensorTest, Constructor4D) {
  Tensor tensor = Tensor({2, 3, 4, 4}, DType_t::FP32, device_);

  auto shape = tensor.shape();
  EXPECT_EQ(shape[0], 2);
  EXPECT_EQ(shape[1], 3);
  EXPECT_EQ(shape[2], 4);
  EXPECT_EQ(shape[3], 4);
  EXPECT_EQ(tensor.size(), 96);
  EXPECT_TRUE(tensor.device_type() == DeviceType::CUDA);
  EXPECT_FALSE(tensor.device_type() == DeviceType::CPU);
}

TEST_F(GPUTensorTest, ConstructorWithShape) {
  Vec<size_t> shape = {2, 3, 4, 4};
  Tensor tensor = Tensor(shape, DType_t::FP32, device_);

  EXPECT_EQ(tensor.shape(), shape);
  EXPECT_EQ(tensor.size(), 96);
  auto tensor_shape = tensor.shape();
  EXPECT_EQ(tensor_shape[0], 2);
  EXPECT_EQ(tensor_shape[1], 3);
  EXPECT_EQ(tensor_shape[2], 4);
  EXPECT_EQ(tensor_shape[3], 4);
  EXPECT_TRUE(tensor.device_type() == DeviceType::CUDA);
}

TEST_F(GPUTensorTest, DeviceTypeCheck) {
  Tensor tensor = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);

  EXPECT_EQ(tensor.device_type(), DeviceType::CUDA);
  EXPECT_TRUE(tensor.device_type() == DeviceType::CUDA);
  EXPECT_FALSE(tensor.device_type() == DeviceType::CPU);
}

TEST_F(GPUTensorTest, TensorAddition) {
  Tensor tensor1 = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);
  Tensor tensor2 = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);

  fill(tensor1, 2.0);
  fill(tensor2, 3.0);

  Tensor result = tensor1 + tensor2;

  Tensor host_result = to_host(result);

  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 0}), 5.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 1}), 5.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 0}), 5.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 1}), 5.0f);
}

TEST_F(GPUTensorTest, TensorSubtraction) {
  Tensor tensor1 = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);
  Tensor tensor2 = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);

  fill(tensor1, 5.0);
  fill(tensor2, 2.0);

  Tensor result = tensor1 - tensor2;

  Tensor host_result = to_host(result);

  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 0}), 3.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 1}), 3.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 0}), 3.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 1}), 3.0f);
}

TEST_F(GPUTensorTest, TensorMultiplication) {
  Tensor tensor1 = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);
  Tensor tensor2 = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);

  fill(tensor1, 3.0);
  fill(tensor2, 4.0);

  Tensor result = tensor1 * tensor2;

  Tensor host_result = to_host(result);

  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 0}), 12.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 1}), 12.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 0}), 12.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 1}), 12.0f);
}

TEST_F(GPUTensorTest, TensorDivision) {
  Tensor tensor1 = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);
  Tensor tensor2 = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);

  fill(tensor1, 12.0);
  fill(tensor2, 4.0);

  Tensor result = tensor1 / tensor2;

  Tensor host_result = to_host(result);

  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 0}), 3.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 1}), 3.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 0}), 3.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 1}), 3.0f);
}

TEST_F(GPUTensorTest, ScalarMultiplication) {
  Tensor tensor = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);
  fill(tensor, 3.0);

  Tensor result = tensor * 2.0;

  Tensor host_result = to_host(result);

  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 0}), 6.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 1}), 6.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 0}), 6.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 1}), 6.0f);
}

TEST_F(GPUTensorTest, ScalarDivision) {
  Tensor tensor = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);
  fill(tensor, 8.0);

  Tensor result = tensor / 2.0;

  Tensor host_result = to_host(result);

  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 0}), 4.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 1}), 4.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 0}), 4.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 1}), 4.0f);
}

TEST_F(GPUTensorTest, InPlaceAddition) {
  Tensor tensor1 = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);
  Tensor tensor2 = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);

  fill(tensor1, 2.0);
  fill(tensor2, 3.0);

  tensor1 += tensor2;

  Tensor host_result = to_host(tensor1);

  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 0}), 5.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 1}), 5.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 0}), 5.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 1}), 5.0f);
}

TEST_F(GPUTensorTest, InPlaceSubtraction) {
  Tensor tensor1 = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);
  Tensor tensor2 = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);

  fill(tensor1, 5.0);
  fill(tensor2, 2.0);

  tensor1 -= tensor2;

  Tensor host_result = to_host(tensor1);

  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 0}), 3.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 1}), 3.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 0}), 3.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 1}), 3.0f);
}

TEST_F(GPUTensorTest, InPlaceMultiplication) {
  Tensor tensor1 = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);
  Tensor tensor2 = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);

  fill(tensor1, 3.0);
  fill(tensor2, 4.0);

  tensor1 *= tensor2;

  Tensor host_result = to_host(tensor1);

  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 0}), 12.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 1}), 12.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 0}), 12.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 1}), 12.0f);
}

TEST_F(GPUTensorTest, InPlaceScalarMultiplication) {
  Tensor tensor = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);
  fill(tensor, 3.0);

  tensor *= 2.0;

  Tensor host_result = to_host(tensor);

  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 0}), 6.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 1}), 6.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 0}), 6.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 1}), 6.0f);
}

TEST_F(GPUTensorTest, InPlaceScalarDivision) {
  Tensor tensor = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);
  fill(tensor, 8.0);

  tensor /= 2.0;

  Tensor host_result = to_host(tensor);

  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 0}), 4.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 1}), 4.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 0}), 4.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 1}), 4.0f);
}

TEST_F(GPUTensorTest, SameShapeComparison) {
  Tensor tensor1 = Tensor({2, 3, 4, 5}, DType_t::FP32, device_);
  Tensor tensor2 = Tensor({2, 3, 4, 5}, DType_t::FP32, device_);
  Tensor tensor3 = Tensor({2, 3, 4, 6}, DType_t::FP32, device_);

  EXPECT_TRUE(tensor1.shape() == tensor2.shape());
  EXPECT_FALSE(tensor1.shape() == tensor3.shape());
}

TEST_F(GPUTensorTest, AdditionShapeMismatch) {
  Tensor tensor1 = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);
  Tensor tensor2 = Tensor({1, 1, 3, 3}, DType_t::FP32, device_);

  EXPECT_THROW(tensor1 + tensor2, std::invalid_argument);
}

TEST_F(GPUTensorTest, DivisionByZero) {
  Tensor tensor = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);

  EXPECT_THROW(tensor / 0.0, std::invalid_argument);
}

TEST_F(GPUTensorTest, FillOperation) {
  Tensor tensor = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);
  fill(tensor, 42.0);

  Tensor host_result = to_host(tensor);

  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 0}), 42.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 1}), 42.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 0}), 42.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 1}), 42.0f);
}

TEST_F(GPUTensorTest, CloneOperation) {
  Tensor original = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);
  fill(original, 5.0);

  Tensor cloned = clone(original);

  EXPECT_TRUE(original.shape() == cloned.shape());
  EXPECT_TRUE(cloned.device_type() == DeviceType::CUDA);

  Tensor host_result = to_host(cloned);

  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 0}), 5.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 1}), 5.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 0}), 5.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 1, 1}), 5.0f);
}

TEST_F(GPUTensorTest, MoveConstructor) {
  Tensor original = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);
  fill(original, 42.0);

  Tensor moved(std::move(original));

  EXPECT_EQ(moved.size(), 4);
  EXPECT_TRUE(moved.device_type() == DeviceType::CUDA);

  Tensor host_result = to_host(moved);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 0}), 42.0f);
  EXPECT_TRUE(!original);
}

TEST_F(GPUTensorTest, MoveAssignment) {
  Tensor original = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);
  fill(original, 42.0);

  Tensor moved = Tensor({1, 1, 1, 1}, DType_t::FP32, device_);
  moved = std::move(original);

  EXPECT_EQ(moved.size(), 4);
  EXPECT_TRUE(moved.device_type() == DeviceType::CUDA);

  Tensor host_result = to_host(moved);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 0}), 42.0f);
}

TEST_F(GPUTensorTest, MultiBatchAccess) {
  Tensor host_tensor = Tensor({2, 1, 2, 2}, DType_t::FP32);

  host_tensor.at<float>({0, 0, 0, 0}) = 1.0f;
  host_tensor.at<float>({1, 0, 0, 0}) = 2.0f;

  Tensor gpu_tensor = to_device(host_tensor, device_);
  Tensor result = to_host(gpu_tensor);

  EXPECT_FLOAT_EQ(result.at<float>({0, 0, 0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(result.at<float>({1, 0, 0, 0}), 2.0f);
}

TEST_F(GPUTensorTest, MultiChannelAccess) {
  Tensor host_tensor = Tensor({1, 3, 2, 2}, DType_t::FP32);

  host_tensor.at<float>({0, 0, 0, 0}) = 1.0f;
  host_tensor.at<float>({0, 1, 0, 0}) = 2.0f;
  host_tensor.at<float>({0, 2, 0, 0}) = 3.0f;

  Tensor gpu_tensor = to_device(host_tensor, device_);
  Tensor result = to_host(gpu_tensor);

  EXPECT_FLOAT_EQ(result.at<float>({0, 0, 0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(result.at<float>({0, 1, 0, 0}), 2.0f);
  EXPECT_FLOAT_EQ(result.at<float>({0, 2, 0, 0}), 3.0f);
}

TEST_F(GPUTensorTest, CopyConstructor) {
  Tensor original = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);
  fill(original, 42.0);

  Tensor copy = clone(original);

  EXPECT_EQ(copy.size(), original.size());
  EXPECT_TRUE(copy.shape() == original.shape());
  EXPECT_TRUE(copy.device_type() == DeviceType::CUDA);

  Tensor host_copy = to_host(copy);
  EXPECT_FLOAT_EQ(host_copy.at<float>({0, 0, 0, 0}), 42.0f);
}

TEST_F(GPUTensorTest, ToCPU) {
  Tensor gpu_tensor = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);
  fill(gpu_tensor, 42.0);

  Tensor host_tensor = to_host(gpu_tensor);

  EXPECT_TRUE(host_tensor.device_type() == DeviceType::CPU);
  EXPECT_FALSE(host_tensor.device_type() == DeviceType::CUDA);
  EXPECT_EQ(host_tensor.size(), 4);

  EXPECT_FLOAT_EQ(host_tensor.at<float>({0, 0, 0, 0}), 42.0f);
  EXPECT_FLOAT_EQ(host_tensor.at<float>({0, 0, 0, 1}), 42.0f);
  EXPECT_FLOAT_EQ(host_tensor.at<float>({0, 0, 1, 0}), 42.0f);
  EXPECT_FLOAT_EQ(host_tensor.at<float>({0, 0, 1, 1}), 42.0f);
}

TEST_F(GPUTensorTest, ToGPUFromCPU) {
  Tensor host_tensor = Tensor({1, 1, 2, 2}, DType_t::FP32);
  host_tensor.at<float>({0, 0, 0, 0}) = 1.0f;
  host_tensor.at<float>({0, 0, 0, 1}) = 2.0f;
  host_tensor.at<float>({0, 0, 1, 0}) = 3.0f;
  host_tensor.at<float>({0, 0, 1, 1}) = 4.0f;

  Tensor gpu_tensor = to_device(host_tensor, device_);

  EXPECT_TRUE(gpu_tensor.device_type() == DeviceType::CUDA);
  EXPECT_FALSE(gpu_tensor.device_type() == DeviceType::CPU);
  EXPECT_EQ(gpu_tensor.size(), 4);

  Tensor result = to_host(gpu_tensor);
  EXPECT_FLOAT_EQ(result.at<float>({0, 0, 0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(result.at<float>({0, 0, 0, 1}), 2.0f);
  EXPECT_FLOAT_EQ(result.at<float>({0, 0, 1, 0}), 3.0f);
  EXPECT_FLOAT_EQ(result.at<float>({0, 0, 1, 1}), 4.0f);
}

TEST_F(GPUTensorTest, ToGPUIdempotent) {
  Tensor gpu_tensor = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);
  fill(gpu_tensor, 42.0);

  Tensor still_gpu = to_device(gpu_tensor, device_);

  EXPECT_TRUE(still_gpu.device_type() == DeviceType::CUDA);

  Tensor host_result = to_host(still_gpu);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 0}), 42.0f);
}

TEST_F(GPUTensorTest, ToCPUIdempotent) {
  Tensor host_tensor = Tensor({1, 1, 2, 2}, DType_t::FP32);
  host_tensor.at<float>({0, 0, 0, 0}) = 42.0f;

  Tensor still_cpu = to_host(host_tensor);

  EXPECT_TRUE(still_cpu.device_type() == DeviceType::CPU);
  EXPECT_FLOAT_EQ(still_cpu.at<float>({0, 0, 0, 0}), 42.0f);
}

TEST_F(GPUTensorTest, FillRandomUniform) {
  Tensor tensor = Tensor({1, 10, 10, 10}, DType_t::FP32, device_);

  fill_uniform(tensor, 0.0, 1.0, 12345ULL);

  Tensor host_result = to_host(tensor);

  bool all_in_range = true;
  for (size_t i = 0; i < host_result.size(); ++i) {
    float val = host_result.at<float>({i / 4, 0, (i / 2) % 2, i % 2});
    if (val < 0.0f || val > 1.0f) {
      all_in_range = false;
      break;
    }
  }
  EXPECT_TRUE(all_in_range);

  float first_val = host_result.at<float>({0, 0, 0, 0});
  bool has_different = false;
  for (size_t i = 1; i < std::min(host_result.size(), size_t(100)); ++i) {
    if (std::abs(host_result.at<float>({i / 4, 0, (i / 2) % 2, i % 2}) - first_val) > 1e-6f) {
      has_different = true;
      break;
    }
  }
  EXPECT_TRUE(has_different);
}

TEST_F(GPUTensorTest, FillRandomNormal) {
  Tensor tensor = Tensor({1, 10, 10, 10}, DType_t::FP32, device_);
  fill_normal(tensor, 0.0, 1.0, 12345ULL);

  Tensor host_result = to_host(tensor);

  float first_val = host_result.at<float>({0, 0, 0, 0});
  bool has_different = false;
  for (size_t i = 1; i < std::min(host_result.size(), size_t(100)); ++i) {
    if (std::abs(host_result.at<float>({i / 4, 0, (i / 2) % 2, i % 2}) - first_val) > 1e-6f) {
      has_different = true;
      break;
    }
  }
  EXPECT_TRUE(has_different);

  float sum = 0.0f;
  for (size_t i = 0; i < host_result.size(); ++i) {
    sum += host_result.at<float>({i / 4, 0, (i / 2) % 2, i % 2});
  }
  float mean = sum / host_result.size();

  EXPECT_NEAR(mean, 0.0f, 0.2f);
}

class GPUTensorSizeTest
    : public ::testing::TestWithParam<std::tuple<size_t, size_t, size_t, size_t>> {
protected:
  static void SetUpTestSuite() {
    DeviceManager &manager = DeviceManager::instance();
    if (manager.get_all().empty()) {
      initializeDefaultDevices();
    }
    Vec<DeviceID> device_ids = manager.get_all();

    has_gpu_ = false;
    for (const DeviceID &id : device_ids) {
      Device &device = manager.get(id);
      if (device.device_type() == DeviceType::CUDA) {
        has_gpu_ = true;
        device_ = device;
        stream_ = device.default_stream();
        break;
      }
    }

    if (!has_gpu_) {
      GTEST_SKIP() << "No CUDA device available, skipping CUDA tensor tests";
    }
  }

  static void TearDownTestSuite() {}

  static bool has_gpu_;
  static sref<Device> device_;
  static stream stream_;
};

bool GPUTensorSizeTest::has_gpu_ = false;
sref<Device> GPUTensorSizeTest::device_;
stream GPUTensorSizeTest::stream_ = nullptr;

TEST_P(GPUTensorSizeTest, ConstructorAndSize) {
  auto [batch, channels, height, width] = GetParam();
  Tensor tensor = Tensor({batch, channels, height, width}, DType_t::FP32, device_);

  auto tensor_shape = tensor.shape();
  EXPECT_EQ(tensor_shape[0], batch);
  EXPECT_EQ(tensor_shape[1], channels);
  EXPECT_EQ(tensor_shape[2], height);
  EXPECT_EQ(tensor_shape[3], width);
  EXPECT_EQ(tensor.size(), batch * channels * height * width);
  EXPECT_TRUE(tensor.device_type() == DeviceType::CUDA);
}

INSTANTIATE_TEST_SUITE_P(DifferentShapes, GPUTensorSizeTest,
                         ::testing::Values(std::make_tuple(1, 1, 1, 1),
                                           std::make_tuple(1, 3, 32, 32),
                                           std::make_tuple(16, 64, 28, 28),
                                           std::make_tuple(32, 128, 14, 14)));

TEST_F(GPUTensorTest, LargeTensorOperations) {
  Tensor tensor1 = Tensor({4, 16, 64, 64}, DType_t::FP32, device_);
  Tensor tensor2 = Tensor({4, 16, 64, 64}, DType_t::FP32, device_);

  fill(tensor1, 1.5);
  fill(tensor2, 2.5);

  Tensor result = tensor1 + tensor2;

  EXPECT_EQ(result.size(), 4 * 16 * 64 * 64);
  EXPECT_TRUE(result.device_type() == DeviceType::CUDA);

  Tensor host_result = to_host(result);
  EXPECT_FLOAT_EQ(host_result.at<float>({0, 0, 0, 0}), 4.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({1, 5, 10, 20}), 4.0f);
  EXPECT_FLOAT_EQ(host_result.at<float>({3, 15, 63, 63}), 4.0f);
}

TEST_F(GPUTensorTest, FloatingPointComparisons) {
  Tensor tensor1 = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);
  Tensor tensor2 = Tensor({1, 1, 2, 2}, DType_t::FP32, device_);

  fill(tensor1, 0.1 + 0.2);
  fill(tensor2, 0.3);

  Tensor diff = tensor1 - tensor2;
  Tensor host_diff = to_host(diff);

  for (size_t i = 0; i < host_diff.size(); ++i) {
    EXPECT_NEAR(host_diff.at<float>({i / 4, 0, (i / 2) % 2, i % 2}), 0.0f, 1e-6f);
  }
}
