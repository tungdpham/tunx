/*
 * Copyright (c) 2026 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#ifdef TUNX_USE_CUDA

#include <gtest/gtest.h>

#include "device/device_manager.hpp"
#include "engine_test_utils.hpp"
#include "nn/engines/cuda_engine.hpp"
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"
#include "tensor_test_utils.hpp"

using namespace tunx;

class CUDAEngineTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    initializeDefaultDevices();
    DeviceManager& manager = DeviceManager::instance();
    Vec<DeviceID> device_ids = manager.get_all();

    has_gpu_ = false;
    for (const DeviceID& id : device_ids) {
      Device& device = manager.get(id);
      if (device.device_type() == DeviceType::CUDA) {
        has_gpu_ = true;
        break;
      }
    }

    if (!has_gpu_) {
      GTEST_SKIP() << "No CUDA device available, skipping CUDA engine tests";
    }

    engine_ = std::make_unique<CUDAEngine>();
    stream_ = getGPU().default_stream();
    handle_ = engine_->create_handle(stream_);
  }

  static void TearDownTestSuite() { engine_.reset(); }

  static bool has_gpu_;
  static std::unique_ptr<CUDAEngine> engine_;
  static stream stream_;
  static engine_handle handle_;
};

bool CUDAEngineTest::has_gpu_ = false;
std::unique_ptr<CUDAEngine> CUDAEngineTest::engine_;
stream CUDAEngineTest::stream_ = nullptr;
engine_handle CUDAEngineTest::handle_;

TEST_F(CUDAEngineTest, DenseFwdReturnsCorrectResults) {
  size_t batch_size = 16;
  size_t in_features = 64;
  size_t out_features = 32;

  DenseStats stats{
      .batch_size = batch_size,
      .in_features = in_features,
      .out_features = out_features,
      .use_bias = true,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };

  Tensor input({batch_size, in_features}, DType_t::FP32, getGPU());
  fill_normal(input, 0.0, 0.5, 12345ULL);
  Tensor weight({out_features, in_features}, DType_t::FP32, getGPU());
  fill_normal(weight, 0.0, 0.1, 12345ULL);
  Tensor bias({out_features}, DType_t::FP32, getGPU());
  fill_normal(bias, 0.0, 0.1, 12345ULL);
  Tensor output({batch_size, out_features}, DType_t::FP32, getGPU());

  WorkspaceReq req = engine_->query_dense_graph(handle_, stats, type_desc);

  size_t ws_size = req.fwd_workspace > 0 ? req.fwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getGPU());

  engine_->dense_fwd(handle_, stats, input.data_as<void>(), weight.data_as<void>(),
                     bias.data_as<void>(), output.data_as<void>(), workspace.data_as<void>(),
                     type_desc);
  cudaDeviceSynchronize();

  Tensor expected_output({batch_size, out_features}, DType_t::FP32, getHost());

  Tensor input_host = to_host(input);
  Tensor weight_host = to_host(weight);
  Tensor bias_host = to_host(bias);

  math_dense_fwd(input_host.data_as<float>(), weight_host.data_as<float>(),
                 bias_host.data_as<float>(), expected_output.data_as<float>(), batch_size,
                 in_features, out_features);

  compare_tensor(to_host(output), expected_output);
}

TEST_F(CUDAEngineTest, DenseWgradReturnsCorrectResults) {
  size_t batch_size = 16;
  size_t in_features = 64;
  size_t out_features = 32;

  DenseStats stats{
      .batch_size = batch_size,
      .in_features = in_features,
      .out_features = out_features,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };

  Tensor input({batch_size, in_features}, DType_t::FP32, getGPU());
  fill_normal(input, 0.0, 0.5, 12345ULL);
  Tensor grad_output({batch_size, out_features}, DType_t::FP32, getGPU());
  fill_normal(grad_output, 0.0, 0.2, 12345ULL);
  Tensor grad_weight_temp({out_features, in_features}, DType_t::FP32, getGPU());
  fill(grad_weight_temp, 0.0f);
  Tensor grad_weight({out_features, in_features}, DType_t::FP32, getGPU());
  fill(grad_weight, 0.0f);

  WorkspaceReq req = engine_->query_dense_graph(handle_, stats, type_desc);

  size_t ws_size = req.bwd_workspace > 0 ? req.bwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getGPU());

  engine_->dense_wgrad(handle_, stats, grad_output.data_as<void>(), input.data_as<void>(),
                       grad_weight.data_as<void>(), workspace.data_as<void>(), type_desc);
  cudaDeviceSynchronize();

  Tensor expected_grad_weight({out_features, in_features}, DType_t::FP32, getHost());

  Tensor input_host = to_host(input);
  Tensor grad_out_host = to_host(grad_output);

  math_dense_wgrad(input_host.data_as<float>(), grad_out_host.data_as<float>(),
                   expected_grad_weight.data_as<float>(), batch_size, in_features, out_features);

  compare_tensor(to_host(grad_weight), expected_grad_weight);
}

TEST_F(CUDAEngineTest, DenseDgradReturnsCorrectResults) {
  size_t batch_size = 16;
  size_t in_features = 64;
  size_t out_features = 32;

  DenseStats stats{
      .batch_size = batch_size,
      .in_features = in_features,
      .out_features = out_features,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };

  Tensor grad_output({batch_size, out_features}, DType_t::FP32, getGPU());
  fill_normal(grad_output, 0.0, 0.5, 12345ULL);
  Tensor weight({out_features, in_features}, DType_t::FP32, getGPU());
  fill_normal(weight, 0.0, 0.1, 12345ULL);
  Tensor grad_input({batch_size, in_features}, DType_t::FP32, getGPU());

  WorkspaceReq req = engine_->query_dense_graph(handle_, stats, type_desc);

  size_t ws_size = req.bwd_workspace > 0 ? req.bwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getGPU());

  engine_->dense_dgrad(handle_, stats, grad_output.data_as<void>(), weight.data_as<void>(),
                       grad_input.data_as<void>(), workspace.data_as<void>(), type_desc);
  cudaDeviceSynchronize();

  Tensor expected_grad_input({batch_size, in_features}, DType_t::FP32, getHost());

  Tensor grad_out_host = to_host(grad_output);
  Tensor weight_host = to_host(weight);

  math_dense_dgrad(grad_out_host.data_as<float>(), weight_host.data_as<float>(),
                   expected_grad_input.data_as<float>(), batch_size, in_features, out_features);

  compare_tensor(to_host(grad_input), expected_grad_input);
}

TEST_F(CUDAEngineTest, DenseBgradReturnsCorrectResults) {
  size_t batch_size = 16;
  size_t in_features = 64;
  size_t out_features = 32;

  DenseStats stats{
      .batch_size = batch_size,
      .in_features = in_features,
      .out_features = out_features,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };

  Tensor grad_output({batch_size, out_features}, DType_t::FP32, getGPU());
  fill_normal(grad_output, 0.0, 0.5, 12345ULL);
  Tensor grad_bias_temp({out_features}, DType_t::FP32, getGPU());
  fill(grad_bias_temp, 0.0f);
  Tensor grad_bias({out_features}, DType_t::FP32, getGPU());
  fill(grad_bias, 0.0f);

  WorkspaceReq req = engine_->query_dense_graph(handle_, stats, type_desc);

  size_t ws_size = req.bwd_workspace > 0 ? req.bwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getGPU());

  engine_->dense_bgrad(handle_, stats, grad_output.data_as<void>(), grad_bias.data_as<void>(),
                       workspace.data_as<void>(), type_desc);
  cudaDeviceSynchronize();

  Tensor expected_grad_bias({out_features}, DType_t::FP32, getHost());

  Tensor grad_out_host = to_host(grad_output);

  math_dense_bgrad(grad_out_host.data_as<float>(), expected_grad_bias.data_as<float>(), batch_size,
                   out_features);

  compare_tensor(to_host(grad_bias), expected_grad_bias);
}

TEST_F(CUDAEngineTest, TransposeReturnsCorrectResults) {
  size_t batch_size = 2;
  size_t num_heads = 4;
  size_t seq_len = 8;
  size_t head_dim = 16;

  TransposeStats stats{
      .shape = {batch_size, num_heads, seq_len, head_dim, 0, 0, 0, 0},
      .ndim = 4,
      .dim0 = 1,
      .dim1 = 2,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };

  Tensor input({batch_size, num_heads, seq_len, head_dim}, DType_t::FP32, getGPU());
  fill_normal(input, 0.0, 1.0, 12345ULL);
  Tensor output({batch_size, seq_len, num_heads, head_dim}, DType_t::FP32, getGPU());

  WorkspaceReq req = engine_->query_transpose_graph(handle_, stats, type_desc);
  size_t ws_size = req.fwd_workspace > 0 ? req.fwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getGPU());

  engine_->transpose(handle_, stats, input.data_as<void>(), output.data_as<void>(),
                     workspace.data_as<void>(), type_desc);
  cudaDeviceSynchronize();

  Tensor expected_output({batch_size, seq_len, num_heads, head_dim}, DType_t::FP32, getHost());
  Tensor host_input = to_host(input);
  math_transpose(host_input.data_as<float>(), expected_output.data_as<float>(), stats.shape,
                 stats.ndim, stats.dim0, stats.dim1);

  compare_tensor(to_host(output), expected_output);
}

TEST_F(CUDAEngineTest, SliceFwdReturnsCorrectResults) {
  SliceStats stats{
      .outer_size = 2,
      .inner_size = 8,
      .axis_size = 10,
      .start = 2,
      .length = 4,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };

  Tensor input({stats.outer_size, stats.axis_size, stats.inner_size}, DType_t::FP32, getGPU());
  fill_normal(input, 0.0, 1.0, 12345ULL);
  Tensor output({stats.outer_size, stats.length, stats.inner_size}, DType_t::FP32, getGPU());

  WorkspaceReq req = engine_->query_slice_graph(handle_, stats, type_desc);
  size_t ws_size = req.fwd_workspace > 0 ? req.fwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getGPU());

  engine_->slice_fwd(handle_, stats, input.data_as<void>(), output.data_as<void>(),
                     workspace.data_as<void>(), type_desc);
  cudaDeviceSynchronize();

  Tensor expected_output({stats.outer_size, stats.length, stats.inner_size}, DType_t::FP32, getHost());
  Tensor input_host = to_host(input);
  math_slice_fwd(input_host.data_as<float>(), expected_output.data_as<float>(), stats.outer_size,
                 stats.inner_size, stats.axis_size, stats.start, stats.length);

  compare_tensor(to_host(output), expected_output);
}

TEST_F(CUDAEngineTest, SliceBwdReturnsCorrectResults) {
  SliceStats stats{
      .outer_size = 2,
      .inner_size = 8,
      .axis_size = 10,
      .start = 2,
      .length = 4,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };

  Tensor grad_output({stats.outer_size, stats.length, stats.inner_size}, DType_t::FP32, getGPU());
  fill_normal(grad_output, 0.0, 1.0, 12345ULL);
  
  Tensor grad_input({stats.outer_size, stats.axis_size, stats.inner_size}, DType_t::FP32, getGPU());
  fill(grad_input, 0.0f); // Initialize to 0

  WorkspaceReq req = engine_->query_slice_graph(handle_, stats, type_desc);
  size_t ws_size = req.bwd_workspace > 0 ? req.bwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getGPU());

  engine_->slice_bwd(handle_, stats, grad_output.data_as<void>(), grad_input.data_as<void>(),
                     workspace.data_as<void>(), type_desc);
  cudaDeviceSynchronize();

  Tensor expected_grad_input({stats.outer_size, stats.axis_size, stats.inner_size}, DType_t::FP32, getHost());
  fill(expected_grad_input, 0.0f);
  
  Tensor grad_output_host = to_host(grad_output);
  math_slice_bwd(grad_output_host.data_as<float>(), expected_grad_input.data_as<float>(), stats.outer_size,
                 stats.inner_size, stats.axis_size, stats.start, stats.length);

  compare_tensor(to_host(grad_input), expected_grad_input);
}

#endif  // TUNX_USE_CUDA
