/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>

#include "device/device_manager.hpp"
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"
#include "tensor_test_utils.hpp"

using namespace tunx;

#ifdef TUNX_USE_CUDA

class GPUOpsTest : public ::testing::Test {
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

  void TearDown() override {}

  static void TearDownTestSuite() {}

  static bool has_gpu_;
  static sref<Device> device_;
  static stream stream_;
};

bool GPUOpsTest::has_gpu_ = false;
sref<Device> GPUOpsTest::device_;
stream GPUOpsTest::stream_ = nullptr;

TEST_F(GPUOpsTest, PadBasic) {
  Tensor host_tensor = Tensor({1, 1, 3, 3}, DType_t::FP32);
  auto host_data = host_tensor.data_as<float>();
  for (size_t i = 0; i < 9; ++i) {
    host_data[i] = static_cast<float>(i + 1);
  }

  Tensor host_padded = Tensor({1, 1, 5, 5}, DType_t::FP32);
  pad(host_tensor, host_padded, 1, 1);

  Tensor gpu_tensor = to_device(host_tensor, getGPU());
  Tensor gpu_padded = Tensor({1, 1, 5, 5}, DType_t::FP32, getGPU());
  pad(gpu_tensor, gpu_padded, 1, 1);

  compare_tensor(host_padded, gpu_padded);
}

TEST_F(GPUOpsTest, PadMultiChannel) {
  Tensor host_tensor = Tensor({2, 3, 4, 4}, DType_t::FP32);
  fill_normal(host_tensor, 0.0, 10.0f, 12345ULL);

  Tensor host_padded = Tensor({2, 3, 8, 8}, DType_t::FP32);
  pad(host_tensor, host_padded, 2, 2, -1.0f);

  Tensor gpu_tensor = to_device(host_tensor, getGPU());
  Tensor gpu_padded = Tensor({2, 3, 8, 8}, DType_t::FP32, getGPU());
  pad(gpu_tensor, gpu_padded, 2, 2, -1.0f);

  compare_tensor(host_padded, gpu_padded);
}

TEST_F(GPUOpsTest, PadAsymmetric) {
  Tensor host_tensor = Tensor({1, 2, 5, 7}, DType_t::FP32);
  fill_normal(host_tensor, 0.0, 5.0f, 12345ULL);

  Tensor host_padded = Tensor({1, 2, 11, 9}, DType_t::FP32);
  pad(host_tensor, host_padded, 3, 1, 2.5f);

  Tensor gpu_tensor = to_device(host_tensor, getGPU());
  Tensor gpu_padded = Tensor({1, 2, 11, 9}, DType_t::FP32, getGPU());
  pad(gpu_tensor, gpu_padded, 3, 1, 2.5f);

  compare_tensor(host_padded, gpu_padded);
}

TEST_F(GPUOpsTest, UnpadBasic) {
  Tensor host_tensor = Tensor({1, 1, 5, 5}, DType_t::FP32);
  fill_normal(host_tensor, 0.0, 10.0f, 12345ULL);

  Tensor host_unpadded = Tensor({1, 1, 3, 3}, DType_t::FP32);
  unpad(host_tensor, host_unpadded, 1, 1);

  Tensor gpu_tensor = to_device(host_tensor, getGPU());
  Tensor gpu_unpadded = Tensor({1, 1, 3, 3}, DType_t::FP32, getGPU());
  unpad(gpu_tensor, gpu_unpadded, 1, 1);

  compare_tensor(host_unpadded, gpu_unpadded);
}

TEST_F(GPUOpsTest, UnpadMultiChannel) {
  Tensor host_tensor = Tensor({2, 3, 8, 8}, DType_t::FP32);
  fill_normal(host_tensor, 0.0, 15.0f, 12345ULL);

  Tensor host_unpadded = Tensor({2, 3, 4, 4}, DType_t::FP32);
  unpad(host_tensor, host_unpadded, 2, 2);

  Tensor gpu_tensor = to_device(host_tensor, getGPU());
  Tensor gpu_unpadded = Tensor({2, 3, 4, 4}, DType_t::FP32, getGPU());
  unpad(gpu_tensor, gpu_unpadded, 2, 2);

  compare_tensor(host_unpadded, gpu_unpadded);
}

TEST_F(GPUOpsTest, PadUnpadRoundTrip) {
  Tensor host_original = Tensor({1, 2, 4, 4}, DType_t::FP32);
  fill_normal(host_original, 0.0, 8.0f, 12345ULL);

  Tensor host_padded = Tensor({1, 2, 8, 8}, DType_t::FP32);
  pad(host_original, host_padded, 2, 2);
  Tensor host_restored = Tensor({1, 2, 4, 4}, DType_t::FP32);
  unpad(host_padded, host_restored, 2, 2);

  Tensor gpu_original = to_device(host_original, getGPU());
  Tensor gpu_padded = Tensor({1, 2, 8, 8}, DType_t::FP32, getGPU());
  pad(gpu_original, gpu_padded, 2, 2);
  Tensor gpu_restored = Tensor({1, 2, 4, 4}, DType_t::FP32, getGPU());
  unpad(gpu_padded, gpu_restored, 2, 2);

  compare_tensor(host_original, host_restored);
  compare_tensor(host_original, gpu_restored);
  compare_tensor(host_restored, gpu_restored);
}

TEST_F(GPUOpsTest, CropBasic) {
  Tensor host_tensor = Tensor({1, 1, 5, 5}, DType_t::FP32);
  auto host_data = host_tensor.data_as<float>();
  for (size_t i = 0; i < 25; ++i) {
    host_data[i] = static_cast<float>(i);
  }

  Tensor host_cropped = Tensor({1, 1, 3, 3}, DType_t::FP32);
  crop(host_tensor, host_cropped, 1, 1, 3, 3);

  Tensor gpu_tensor = to_device(host_tensor, getGPU());
  Tensor gpu_cropped = Tensor({1, 1, 3, 3}, DType_t::FP32, getGPU());
  crop(gpu_tensor, gpu_cropped, 1, 1, 3, 3);

  compare_tensor(host_cropped, gpu_cropped);
}

TEST_F(GPUOpsTest, CropMultiChannel) {
  Tensor host_tensor = Tensor({2, 3, 10, 10}, DType_t::FP32);
  fill_normal(host_tensor, 0.0, 20.0f, 12345ULL);

  Tensor host_cropped = Tensor({2, 3, 6, 6}, DType_t::FP32);
  crop(host_tensor, host_cropped, 2, 3, 7, 8);

  Tensor gpu_tensor = to_device(host_tensor, getGPU());
  Tensor gpu_cropped = Tensor({2, 3, 6, 6}, DType_t::FP32, getGPU());
  crop(gpu_tensor, gpu_cropped, 2, 3, 7, 8);

  compare_tensor(host_cropped, gpu_cropped);
}

TEST_F(GPUOpsTest, CropCorner) {
  Tensor host_tensor = Tensor({1, 2, 8, 8}, DType_t::FP32);
  fill_normal(host_tensor, 0.0, 12.0f, 12345ULL);

  Tensor host_cropped = Tensor({1, 2, 4, 4}, DType_t::FP32);
  crop(host_tensor, host_cropped, 0, 0, 3, 3);

  Tensor gpu_tensor = to_device(host_tensor, getGPU());
  Tensor gpu_cropped = Tensor({1, 2, 4, 4}, DType_t::FP32, getGPU());
  crop(gpu_tensor, gpu_cropped, 0, 0, 3, 3);

  compare_tensor(host_cropped, gpu_cropped);
}

TEST_F(GPUOpsTest, CropBottomRight) {
  Tensor host_tensor = Tensor({1, 1, 6, 6}, DType_t::FP32);
  fill_normal(host_tensor, 0.0, 10.0f, 12345ULL);

  Tensor host_cropped = Tensor({1, 1, 3, 3}, DType_t::FP32);
  crop(host_tensor, host_cropped, 3, 3, 5, 5);

  Tensor gpu_tensor = to_device(host_tensor, getGPU());
  Tensor gpu_cropped = Tensor({1, 1, 3, 3}, DType_t::FP32, getGPU());
  crop(gpu_tensor, gpu_cropped, 3, 3, 5, 5);

  compare_tensor(host_cropped, gpu_cropped);
}

TEST_F(GPUOpsTest, SliceBatchBasic) {
  Tensor host_tensor = Tensor({4, 2, 3, 3}, DType_t::FP32);
  fill_normal(host_tensor, 0.0, 15.0f, 12345ULL);

  Tensor host_sliced = Tensor({2, 2, 3, 3}, DType_t::FP32);
  slice_batch(host_tensor, host_sliced, 1, 3);

  Tensor gpu_tensor = to_device(host_tensor, getGPU());
  Tensor gpu_sliced = Tensor({2, 2, 3, 3}, DType_t::FP32, getGPU());
  slice_batch(gpu_tensor, gpu_sliced, 1, 3);

  compare_tensor(host_sliced, gpu_sliced);
}

TEST_F(GPUOpsTest, SliceBatchSingle) {
  Tensor host_tensor = Tensor({5, 3, 4, 4}, DType_t::FP32);
  fill_normal(host_tensor, 0.0, 10.0f, 12345ULL);

  Tensor host_sliced = Tensor({1, 3, 4, 4}, DType_t::FP32);
  slice_batch(host_tensor, host_sliced, 2, 3);

  Tensor gpu_tensor = to_device(host_tensor, getGPU());
  Tensor gpu_sliced = Tensor({1, 3, 4, 4}, DType_t::FP32, getGPU());
  slice_batch(gpu_tensor, gpu_sliced, 2, 3);

  compare_tensor(host_sliced, gpu_sliced);
}

TEST_F(GPUOpsTest, SliceBatchFirstBatch) {
  Tensor host_tensor = Tensor({3, 2, 5, 5}, DType_t::FP32);
  fill_normal(host_tensor, 0.0, 8.0f, 12345ULL);

  Tensor host_sliced = Tensor({1, 2, 5, 5}, DType_t::FP32);
  slice_batch(host_tensor, host_sliced, 0, 1);

  Tensor gpu_tensor = to_device(host_tensor, getGPU());
  Tensor gpu_sliced = Tensor({1, 2, 5, 5}, DType_t::FP32, getGPU());
  slice_batch(gpu_tensor, gpu_sliced, 0, 1);

  compare_tensor(host_sliced, gpu_sliced);
}

TEST_F(GPUOpsTest, SplitBasic) {
  Tensor host_tensor = Tensor({4, 2, 3, 3}, DType_t::FP32);
  fill_normal(host_tensor, 0.0, 10.0f, 12345ULL);

  Vec<Tensor> host_splits, gpu_splits;
  split(host_tensor, host_splits, 2);

  Tensor gpu_tensor = to_device(host_tensor, getGPU());
  split(gpu_tensor, gpu_splits, 2);
  ASSERT_EQ(host_splits.size(), gpu_splits.size());

  for (size_t i = 0; i < host_splits.size(); ++i) {
    compare_tensor(host_splits[i], gpu_splits[i]);
  }
}

TEST_F(GPUOpsTest, SplitMultiple) {
  Tensor host_tensor = Tensor({8, 3, 4, 4}, DType_t::FP32);
  fill_normal(host_tensor, 0.0, 15.0f, 12345ULL);
  Vec<Tensor> host_splits;
  split(host_tensor, host_splits, 4);

  Tensor gpu_tensor = to_device(host_tensor, getGPU());
  Vec<Tensor> gpu_splits;
  split(gpu_tensor, gpu_splits, 4);

  ASSERT_EQ(host_splits.size(), gpu_splits.size());

  float *original_data = host_tensor.data_as<float>();
  int original_idx = 0;
  for (size_t i = 0; i < host_splits.size(); ++i) {
    float *split_data = host_splits[i].data_as<float>();
    for (size_t idx = 0; idx < host_splits[i].size(); ++idx) {
      EXPECT_EQ(original_data[original_idx++], split_data[idx])
          << "Mismatch in CPU split at index " << idx << " of split " << i;
    }
  }

  ASSERT_EQ(original_idx, host_tensor.size());

  for (size_t i = 0; i < host_splits.size(); ++i) {
    compare_tensor(host_splits[i], gpu_splits[i]);
  }
}

TEST_F(GPUOpsTest, SplitSingleBatch) {
  Tensor host_tensor = Tensor({6, 2, 5, 5}, DType_t::FP32);
  fill_normal(host_tensor, 0.0, 12.0f, 12345ULL);

  Vec<Tensor> host_splits, gpu_splits;
  split(host_tensor, host_splits, 6);

  Tensor gpu_tensor = to_device(host_tensor, getGPU());
  split(gpu_tensor, gpu_splits, 6);

  ASSERT_EQ(host_splits.size(), gpu_splits.size());

  for (size_t i = 0; i < host_splits.size(); ++i) {
    compare_tensor(host_splits[i], gpu_splits[i]);
  }
}

TEST_F(GPUOpsTest, Im2colBasicKernel3x3) {
  Tensor host_input = Tensor({1, 1, 5, 5}, DType_t::FP32);
  auto host_input_data = host_input.data_as<float>();
  for (size_t i = 0; i < 25; ++i) {
    host_input_data[i] = static_cast<float>(i + 1);
  }

  size_t kernel_h = 3, kernel_w = 3;
  size_t stride_h = 1, stride_w = 1;
  size_t pad_h = 0, pad_w = 0;

  auto input_shape = host_input.shape();
  size_t output_h = (input_shape[2] - kernel_h) / stride_h + 1;
  size_t output_w = (input_shape[3] - kernel_w) / stride_w + 1;
  size_t col_size = input_shape[0] * input_shape[1] * kernel_h * kernel_w * output_h * output_w;

  Tensor host_col_data = Tensor({col_size}, DType_t::FP32);
  im2col(host_input, host_col_data, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  Tensor gpu_input = to_device(host_input, getGPU());
  Tensor gpu_col_data = Tensor({col_size}, DType_t::FP32, getGPU());
  im2col(gpu_input, gpu_col_data, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  Tensor host_col_cpu = to_host(host_col_data);
  Tensor gpu_col_cpu = to_host(gpu_col_data);

  for (size_t i = 0; i < col_size; ++i) {
    EXPECT_NEAR(host_col_cpu.data_as<float>()[i], gpu_col_cpu.data_as<float>()[i], 1e-5f)
        << "Mismatch at index " << i;
  }
}

TEST_F(GPUOpsTest, Im2colWithPadding) {
  Tensor host_input = Tensor({1, 2, 4, 4}, DType_t::FP32);
  fill_normal(host_input, 0.0, 10.0f, 12345ULL);

  size_t kernel_h = 3, kernel_w = 3;
  size_t stride_h = 1, stride_w = 1;
  size_t pad_h = 1, pad_w = 1;

  auto input_shape = host_input.shape();
  size_t padded_h = input_shape[2] + 2 * pad_h;
  size_t padded_w = input_shape[3] + 2 * pad_w;
  size_t output_h = (padded_h - kernel_h) / stride_h + 1;
  size_t output_w = (padded_w - kernel_w) / stride_w + 1;
  size_t col_size = input_shape[0] * input_shape[1] * kernel_h * kernel_w * output_h * output_w;

  Tensor host_col_data = Tensor({col_size}, DType_t::FP32);
  im2col(host_input, host_col_data, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  Tensor gpu_input = to_device(host_input, getGPU());
  Tensor gpu_col_data = Tensor({col_size}, DType_t::FP32, getGPU());
  im2col(gpu_input, gpu_col_data, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  Tensor host_col_cpu = to_host(host_col_data);
  Tensor gpu_col_cpu = to_host(gpu_col_data);

  for (size_t i = 0; i < col_size; ++i) {
    EXPECT_NEAR(host_col_cpu.data_as<float>()[i], gpu_col_cpu.data_as<float>()[i], 1e-5f)
        << "Mismatch at index " << i;
  }
}

TEST_F(GPUOpsTest, Im2colWithStride) {
  Tensor host_input = Tensor({1, 1, 8, 8}, DType_t::FP32);
  fill_normal(host_input, 0.0, 15.0f, 12345ULL);

  size_t kernel_h = 3, kernel_w = 3;
  size_t stride_h = 2, stride_w = 2;
  size_t pad_h = 0, pad_w = 0;

  auto input_shape = host_input.shape();
  size_t output_h = (input_shape[2] - kernel_h) / stride_h + 1;
  size_t output_w = (input_shape[3] - kernel_w) / stride_w + 1;
  size_t col_size = input_shape[0] * input_shape[1] * kernel_h * kernel_w * output_h * output_w;

  Tensor host_col = Tensor({col_size}, DType_t::FP32);
  im2col(host_input, host_col, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  Tensor gpu_input = to_device(host_input, getGPU());
  Tensor gpu_col_data = Tensor({col_size}, DType_t::FP32, getGPU());
  im2col(gpu_input, gpu_col_data, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  Tensor host_col_cpu = to_host(host_col);
  Tensor gpu_col_cpu = to_host(gpu_col_data);

  for (size_t i = 0; i < col_size; ++i) {
    EXPECT_NEAR(host_col_cpu.data_as<float>()[i], gpu_col_cpu.data_as<float>()[i], 1e-5f)
        << "Mismatch at index " << i;
  }
}

TEST_F(GPUOpsTest, Im2colMultiBatch) {
  Tensor host_input = Tensor({4, 3, 6, 6}, DType_t::FP32);
  fill_normal(host_input, 0.0, 12.0f, 12345ULL);

  size_t kernel_h = 3, kernel_w = 3;
  size_t stride_h = 1, stride_w = 1;
  size_t pad_h = 1, pad_w = 1;

  auto input_shape = host_input.shape();
  size_t padded_h = input_shape[2] + 2 * pad_h;
  size_t padded_w = input_shape[3] + 2 * pad_w;
  size_t output_h = (padded_h - kernel_h) / stride_h + 1;
  size_t output_w = (padded_w - kernel_w) / stride_w + 1;
  size_t col_size = input_shape[0] * input_shape[1] * kernel_h * kernel_w * output_h * output_w;

  Tensor host_col = Tensor({col_size}, DType_t::FP32);
  im2col(host_input, host_col, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  Tensor gpu_input = to_device(host_input, getGPU());
  Tensor gpu_col_data = Tensor({col_size}, DType_t::FP32, getGPU());
  im2col(gpu_input, gpu_col_data, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  Tensor host_col_cpu = to_host(host_col);
  Tensor gpu_col_cpu = to_host(gpu_col_data);

  for (size_t i = 0; i < col_size; ++i) {
    EXPECT_NEAR(host_col_cpu.data_as<float>()[i], gpu_col_cpu.data_as<float>()[i], 1e-5f)
        << "Mismatch at index " << i;
  }
}

TEST_F(GPUOpsTest, Col2imBasic) {
  size_t batch_size = 1, channels = 1, height = 5, width = 5;
  size_t kernel_h = 3, kernel_w = 3;
  size_t stride_h = 1, stride_w = 1;
  size_t pad_h = 0, pad_w = 0;

  size_t output_h = (height - kernel_h) / stride_h + 1;
  size_t output_w = (width - kernel_w) / stride_w + 1;
  size_t col_size = batch_size * channels * kernel_h * kernel_w * output_h * output_w;

  Tensor host_col_data = Tensor({col_size}, DType_t::FP32);
  auto host_col_ptr = host_col_data.data_as<float>();
  for (size_t i = 0; i < col_size; ++i) {
    host_col_ptr[i] = static_cast<float>(i % 10);
  }

  Tensor host_result = Tensor({batch_size * channels * height * width}, DType_t::FP32);
  fill(host_result, 0.0f);
  col2im(host_col_data, host_result, batch_size, channels, height, width, kernel_h, kernel_w,
         stride_h, stride_w, pad_h, pad_w);

  Tensor gpu_col_data = to_device(host_col_data, getGPU());

  Tensor gpu_result = Tensor({batch_size * channels * height * width}, DType_t::FP32, getGPU());
  fill(gpu_result, 0.0f);

  col2im(gpu_col_data, gpu_result, batch_size, channels, height, width, kernel_h, kernel_w,
         stride_h, stride_w, pad_h, pad_w);

  Tensor host_result_cpu = to_host(host_result);
  Tensor gpu_result_cpu = to_host(gpu_result);

  for (size_t i = 0; i < host_result_cpu.size(); ++i) {
    EXPECT_NEAR(host_result_cpu.data_as<float>()[i], gpu_result_cpu.data_as<float>()[i], 1e-4f)
        << "Mismatch at index " << i;
  }
}

TEST_F(GPUOpsTest, Col2imWithPadding) {
  size_t batch_size = 1, channels = 2, height = 4, width = 4;
  size_t kernel_h = 3, kernel_w = 3;
  size_t stride_h = 1, stride_w = 1;
  size_t pad_h = 1, pad_w = 1;

  size_t padded_h = height + 2 * pad_h;
  size_t padded_w = width + 2 * pad_w;
  size_t output_h = (padded_h - kernel_h) / stride_h + 1;
  size_t output_w = (padded_w - kernel_w) / stride_w + 1;
  size_t col_size = batch_size * channels * kernel_h * kernel_w * output_h * output_w;

  Tensor host_col_data = Tensor({col_size}, DType_t::FP32);
  auto host_col_ptr = host_col_data.data_as<float>();
  for (size_t i = 0; i < col_size; ++i) {
    host_col_ptr[i] = static_cast<float>((i % 20) - 10);
  }

  Tensor host_result = Tensor({batch_size * channels * height * width}, DType_t::FP32);
  fill(host_result, 0.0f);
  col2im(host_col_data, host_result, batch_size, channels, height, width, kernel_h, kernel_w,
         stride_h, stride_w, pad_h, pad_w);

  Tensor gpu_col_data = to_device(host_col_data, getGPU());

  Tensor gpu_result = Tensor({batch_size * channels * height * width}, DType_t::FP32, getGPU());
  fill(gpu_result, 0.0f);

  col2im(gpu_col_data, gpu_result, batch_size, channels, height, width, kernel_h, kernel_w,
         stride_h, stride_w, pad_h, pad_w);

  Tensor host_result_cpu = to_host(host_result);
  Tensor gpu_result_cpu = to_host(gpu_result);

  for (size_t i = 0; i < host_result_cpu.size(); ++i) {
    EXPECT_NEAR(host_result_cpu.data_as<float>()[i], gpu_result_cpu.data_as<float>()[i], 1e-4f)
        << "Mismatch at index " << i;
  }
}

TEST_F(GPUOpsTest, Im2colCol2imRoundTrip) {
  Tensor host_input = Tensor({1, 1, 6, 6}, DType_t::FP32);
  fill_normal(host_input, 0.0, 10.0f, 12345ULL);

  size_t kernel_h = 3, kernel_w = 3;
  size_t stride_h = 1, stride_w = 1;
  size_t pad_h = 0, pad_w = 0;

  auto input_shape = host_input.shape();
  size_t output_h = (input_shape[2] - kernel_h) / stride_h + 1;
  size_t output_w = (input_shape[3] - kernel_w) / stride_w + 1;
  size_t col_size = input_shape[0] * input_shape[1] * kernel_h * kernel_w * output_h * output_w;

  Tensor host_col_data = Tensor({col_size}, DType_t::FP32);
  im2col(host_input, host_col_data, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  Tensor host_reconstructed = Tensor({1, 1, 6, 6}, DType_t::FP32);
  fill(host_reconstructed, 0.0f);
  col2im(host_col_data, host_reconstructed, input_shape[0], input_shape[1], input_shape[2],
         input_shape[3], kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  Tensor gpu_input = to_device(host_input, getGPU());
  Tensor gpu_col_data = Tensor({col_size}, DType_t::FP32, getGPU());
  im2col(gpu_input, gpu_col_data, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  Tensor gpu_reconstructed = Tensor({1, 1, 6, 6}, DType_t::FP32, getGPU());
  fill(gpu_reconstructed, 0.0f);
  auto gpu_input_shape = gpu_input.shape();
  col2im(gpu_col_data, gpu_reconstructed, gpu_input_shape[0], gpu_input_shape[1],
         gpu_input_shape[2], gpu_input_shape[3], kernel_h, kernel_w, stride_h, stride_w, pad_h,
         pad_w);

  compare_tensor(host_reconstructed, gpu_reconstructed, 1e-4f);
}

TEST_F(GPUOpsTest, CombinedPadCropSlice) {
  Tensor host_original = Tensor({4, 3, 8, 8}, DType_t::FP32);
  fill_normal(host_original, 0.0, 15.0f, 12345ULL);

  Tensor host_padded = Tensor({4, 3, 12, 12}, DType_t::FP32);
  pad(host_original, host_padded, 2, 2);
  Tensor host_cropped = Tensor({4, 3, 6, 6}, DType_t::FP32);
  crop(host_padded, host_cropped, 3, 3, 8, 8);
  Tensor host_sliced = Tensor({2, 3, 6, 6}, DType_t::FP32);
  slice_batch(host_cropped, host_sliced, 1, 3);

  Tensor gpu_original = to_device(host_original, getGPU());
  Tensor gpu_padded = Tensor({4, 3, 12, 12}, DType_t::FP32, getGPU());
  pad(gpu_original, gpu_padded, 2, 2);
  Tensor gpu_cropped = Tensor({4, 3, 6, 6}, DType_t::FP32, getGPU());
  crop(gpu_padded, gpu_cropped, 3, 3, 8, 8);
  Tensor gpu_sliced = Tensor({2, 3, 6, 6}, DType_t::FP32, getGPU());
  slice_batch(gpu_cropped, gpu_sliced, 1, 3);

  compare_tensor(host_sliced, gpu_sliced);
}

TEST_F(GPUOpsTest, LargeTensorOperations) {
  Tensor host_tensor = Tensor({8, 16, 32, 32}, DType_t::FP32);
  fill_normal(host_tensor, 0.0, 20.0f, 12345ULL);

  Tensor host_padded = Tensor({8, 16, 36, 36}, DType_t::FP32);
  pad(host_tensor, host_padded, 2, 2);
  Tensor gpu_tensor = to_device(host_tensor, getGPU());
  Tensor gpu_padded = Tensor({8, 16, 36, 36}, DType_t::FP32, getGPU());
  pad(gpu_tensor, gpu_padded, 2, 2);
  compare_tensor(host_padded, gpu_padded);

  Tensor host_cropped = Tensor({8, 16, 22, 22}, DType_t::FP32);
  crop(host_tensor, host_cropped, 5, 5, 26, 26);
  Tensor gpu_cropped = Tensor({8, 16, 22, 22}, DType_t::FP32, getGPU());
  crop(gpu_tensor, gpu_cropped, 5, 5, 26, 26);
  compare_tensor(host_cropped, gpu_cropped);

  Tensor host_sliced = Tensor({4, 16, 32, 32}, DType_t::FP32);
  slice_batch(host_tensor, host_sliced, 2, 6);
  Tensor gpu_sliced = Tensor({4, 16, 32, 32}, DType_t::FP32, getGPU());
  slice_batch(gpu_tensor, gpu_sliced, 2, 6);
  compare_tensor(host_sliced, gpu_sliced);
}

TEST_F(GPUOpsTest, MinimalTensor) {
  Tensor host_tensor = Tensor({1, 1, 1, 1}, DType_t::FP32);
  auto host_data = host_tensor.data_as<float>();
  host_data[0] = 42.0f;

  Tensor host_padded = Tensor({1, 1, 3, 3}, DType_t::FP32);
  pad(host_tensor, host_padded, 1, 1);
  Tensor gpu_tensor = to_device(host_tensor, getGPU());
  Tensor gpu_padded = Tensor({1, 1, 3, 3}, DType_t::FP32, getGPU());
  pad(gpu_tensor, gpu_padded, 1, 1);

  compare_tensor(host_padded, gpu_padded);
}

TEST_F(GPUOpsTest, SinglePixelPadding) {
  Tensor host_tensor = Tensor({1, 1, 3, 3}, DType_t::FP32);
  fill_normal(host_tensor, 0.0, 5.0f, 12345ULL);

  Tensor host_padded = Tensor({1, 1, 5, 5}, DType_t::FP32);
  pad(host_tensor, host_padded, 1, 1);
  Tensor gpu_tensor = to_device(host_tensor, getGPU());
  Tensor gpu_padded = Tensor({1, 1, 5, 5}, DType_t::FP32, getGPU());
  pad(gpu_tensor, gpu_padded, 1, 1);

  compare_tensor(host_padded, gpu_padded);
}

TEST_F(GPUOpsTest, AsymmetricDimensions) {
  Tensor host_tensor = Tensor({1, 1, 20, 3}, DType_t::FP32);
  fill_normal(host_tensor, 0.0, 10.0f, 12345ULL);

  Tensor host_padded = Tensor({1, 1, 24, 13}, DType_t::FP32);
  pad(host_tensor, host_padded, 2, 5, 1.0f);
  Tensor gpu_tensor = to_device(host_tensor, getGPU());
  Tensor gpu_padded = Tensor({1, 1, 24, 13}, DType_t::FP32, getGPU());
  pad(gpu_tensor, gpu_padded, 2, 5, 1.0f);

  compare_tensor(host_padded, gpu_padded);

  Tensor host_tensor2 = Tensor({1, 1, 3, 20}, DType_t::FP32);
  fill_normal(host_tensor2, 0.0, 10.0f, 12345ULL);

  Tensor host_padded2 = Tensor({1, 1, 13, 24}, DType_t::FP32);
  pad(host_tensor2, host_padded2, 5, 2, -2.0f);
  Tensor gpu_tensor2 = to_device(host_tensor2, getGPU());
  Tensor gpu_padded2 = Tensor({1, 1, 13, 24}, DType_t::FP32, getGPU());
  pad(gpu_tensor2, gpu_padded2, 5, 2, -2.0f);

  compare_tensor(host_padded2, gpu_padded2);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif