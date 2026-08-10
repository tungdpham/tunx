/*
 * Copyright (c) 2026 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#ifdef TUNX_USE_CUDNN
#include <cudnn.h>
#include <cudnn_graph.h>
#include <fmt/core.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <stdexcept>

#include "device/device_manager.hpp"
#include "device/stream.hpp"
#include "engine_test_utils.hpp"
#include "nn/engines/cudnn_engine.hpp"
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"
#include "tensor_test_utils.hpp"
#include "type/type.hpp"

using namespace tunx;

class CuDNNEngineTest : public ::testing::Test {
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
        device_ = device;
        break;
      }
    }

    if (!has_gpu_) {
      GTEST_SKIP() << "No CUDA device available, skipping CuDNN engine tests";
    }

    stream_ = device_->default_stream();
    engine_ = std::make_unique<CuDNNEngine>();
    engine_handle_ = engine_->create_handle(stream_);
  }

  void SetUp() override {}

  void TearDown() override {}

  static void TearDownTestSuite() { engine_.reset(); }

  void check_dense_fwd(const Tensor& input, const Tensor& weight, const Tensor& bias,
                       const Tensor& output) {
    Tensor expected_output(output.shape(), output.dtype(), DeviceAllocator::instance(getHost()));
    size_t batch_size = input.shape()[0];
    size_t in_features = input.shape()[1];
    size_t out_features = output.shape()[1];

    Tensor input_host = to_host(input);
    Tensor weight_host = to_host(weight);
    Tensor bias_host = to_host(bias);

    math_dense_fwd(input_host.data_as<float>(), weight_host.data_as<float>(),
                   bias_host.data_as<float>(), expected_output.data_as<float>(), batch_size,
                   in_features, out_features);

    Tensor output_host = to_host(output);

    compare_tensor(output_host, expected_output);
  }

  void check_dense_wgrad(const Tensor& input, const Tensor& grad_output,
                         const Tensor& grad_weight) {
    Tensor expected_grad_weight(grad_weight.shape(), grad_weight.dtype(),
                                DeviceAllocator::instance(getHost()));
    fill(expected_grad_weight, 0.0);
    size_t batch_size = input.shape()[0];
    size_t in_features = input.shape()[1];
    size_t out_features = grad_output.shape()[1];

    Tensor input_host = to_host(input);
    Tensor grad_out_host = to_host(grad_output);

    math_dense_wgrad(input_host.data_as<float>(), grad_out_host.data_as<float>(),
                     expected_grad_weight.data_as<float>(), batch_size, in_features, out_features);

    Tensor output_host = to_host(grad_weight);

    compare_tensor(output_host, expected_grad_weight);
  }

  void check_dense_dgrad(const Tensor& grad_output, const Tensor& weight,
                         const Tensor& grad_input) {
    Tensor expected_grad_input(grad_input.shape(), grad_input.dtype(),
                               DeviceAllocator::instance(getHost()));
    size_t batch_size = grad_input.shape()[0];
    size_t in_features = grad_input.shape()[1];
    size_t out_features = grad_output.shape()[1];

    Tensor grad_out_host = to_host(grad_output);
    Tensor weight_host = to_host(weight);

    math_dense_dgrad(grad_out_host.data_as<float>(), weight_host.data_as<float>(),
                     expected_grad_input.data_as<float>(), batch_size, in_features, out_features);

    Tensor output_host = to_host(grad_input);

    compare_tensor(output_host, expected_grad_input);
  }

  void check_dense_bgrad(const Tensor& grad_output, const Tensor& grad_bias) {
    Tensor expected_grad_bias(grad_bias.shape(), grad_bias.dtype(),
                              DeviceAllocator::instance(getHost()));
    fill(expected_grad_bias, 0.0f);
    size_t batch_size = grad_output.shape()[0];
    size_t out_features = grad_output.shape()[1];

    Tensor grad_out_host = to_host(grad_output);

    math_dense_bgrad(grad_out_host.data_as<float>(), expected_grad_bias.data_as<float>(),
                     batch_size, out_features);

    Tensor output_host = to_host(grad_bias);

    compare_tensor(output_host, expected_grad_bias);
  }

  // Tensors for conv2d/batchnorm/avgpool are NHWC in memory:
  // input shape {N, H, W, C}, weight shape {OC, KH, KW, IC}

  void check_conv2d_wgrad(const Tensor& input, const Tensor& grad_output, const Tensor& grad_weight,
                          const Conv2DStats& stats) {
    Tensor expected_grad_weight(grad_weight.shape(), grad_weight.dtype(),
                                DeviceAllocator::instance(getHost()));
    fill(expected_grad_weight, 0.0);
    // NHWC: input shape is {N, H, W, C}, grad_output shape is {N, OH, OW, OC}
    size_t output_h = (stats.input_h + 2 * stats.pad_h - stats.kernel_h) / stats.stride_h + 1;
    size_t output_w = (stats.input_w + 2 * stats.pad_w - stats.kernel_w) / stats.stride_w + 1;

    Tensor input_host = to_host(input);
    Tensor grad_out_host = to_host(grad_output);

    math_conv2d_wgrad_naive(grad_out_host.data_as<float>(), input_host.data_as<float>(),
                            expected_grad_weight.data_as<float>(), stats.batch_size, stats.input_h,
                            stats.input_w, stats.in_channels, stats.out_channels, stats.kernel_h,
                            stats.kernel_w, stats.stride_h, stats.stride_w, stats.pad_h,
                            stats.pad_w, output_h, output_w);

    compare_tensor(to_host(grad_weight), expected_grad_weight);
  }

  void check_conv2d_bgrad(const Tensor& grad_output, const Tensor& grad_bias,
                          const Conv2DStats& stats) {
    Tensor expected_grad_bias(grad_bias.shape(), grad_bias.dtype(),
                              DeviceAllocator::instance(getHost()));
    fill(expected_grad_bias, 0.0);
    // NHWC: grad_output shape is {N, OH, OW, OC}
    size_t output_h = (stats.input_h + 2 * stats.pad_h - stats.kernel_h) / stats.stride_h + 1;
    size_t output_w = (stats.input_w + 2 * stats.pad_w - stats.kernel_w) / stats.stride_w + 1;

    Tensor grad_out_host = to_host(grad_output);

    math_conv2d_bgrad_naive(grad_out_host.data_as<float>(), expected_grad_bias.data_as<float>(),
                            stats.batch_size, stats.out_channels, output_h, output_w);

    compare_tensor(to_host(grad_bias), expected_grad_bias);
  }

  void check_batchnorm_bwd(const Tensor& grad_output, const Tensor& input, const Tensor& gamma,
                           const Tensor& mean, const Tensor& invar, const Tensor& grad_input,
                           const Tensor& grad_gamma, const Tensor& grad_beta,
                           const BatchNormStats& stats) {
    Tensor expected_grad_input(grad_input.shape(), grad_input.dtype(),
                               DeviceAllocator::instance(getHost()));
    Tensor expected_grad_gamma(grad_gamma.shape(), grad_gamma.dtype(),
                               DeviceAllocator::instance(getHost()));
    Tensor expected_grad_beta(grad_beta.shape(), grad_beta.dtype(),
                              DeviceAllocator::instance(getHost()));

    fill(expected_grad_gamma, 0.0f);
    fill(expected_grad_beta, 0.0f);

    size_t N = stats.batch_size;
    size_t C = stats.channels;
    size_t S = stats.height * stats.width;

    Tensor grad_out_host = to_host(grad_output);
    Tensor input_host = to_host(input);
    Tensor gamma_host = to_host(gamma);
    Tensor mean_host = to_host(mean);
    Tensor invar_host = to_host(invar);

    // NHWC batchnorm backward: input layout is {N, H, W, C}
    math_batchnorm_bwd(grad_out_host.data_as<float>(), input_host.data_as<float>(),
                       mean_host.data_as<float>(), invar_host.data_as<float>(),
                       gamma_host.data_as<float>(), expected_grad_gamma.data_as<float>(),
                       expected_grad_beta.data_as<float>(), expected_grad_input.data_as<float>(),
                       (const bool*)nullptr, N, C, S, true, false);

    compare_tensor(to_host(grad_input), expected_grad_input);
    compare_tensor(to_host(grad_gamma), expected_grad_gamma);
    compare_tensor(to_host(grad_beta), expected_grad_beta);
  }

  void check_layernorm_bwd(const Tensor& grad_output, const Tensor& input, const Tensor& gamma,
                           const Tensor& grad_input, const Tensor& grad_gamma,
                           const Tensor& grad_beta, const LayerNormStats& stats) {
    Tensor expected_grad_input(grad_input.shape(), grad_input.dtype(),
                               DeviceAllocator::instance(getHost()));
    Tensor expected_grad_gamma(grad_gamma.shape(), grad_gamma.dtype(),
                               DeviceAllocator::instance(getHost()));
    Tensor expected_grad_beta(grad_beta.shape(), grad_beta.dtype(),
                              DeviceAllocator::instance(getHost()));

    fill(expected_grad_gamma, 0.0f);
    fill(expected_grad_beta, 0.0f);

    Tensor grad_out_host = to_host(grad_output);
    Tensor input_host = to_host(input);
    Tensor gamma_host = to_host(gamma);

    math_layernorm_bwd(grad_out_host.data_as<float>(), input_host.data_as<float>(),
                       gamma_host.data_as<float>(), expected_grad_input.data_as<float>(),
                       expected_grad_gamma.data_as<float>(), expected_grad_beta.data_as<float>(),
                       stats.batch_size, stats.channels, static_cast<float>(stats.epsilon));

    compare_tensor(to_host(grad_input), expected_grad_input);
    compare_tensor(to_host(grad_gamma), expected_grad_gamma);
    compare_tensor(to_host(grad_beta), expected_grad_beta);
  }

  void check_avgpool_fwd(const Tensor& input, const Tensor& output, const AvgPool2DStats& stats) {
    Tensor expected_output(output.shape(), output.dtype(), DeviceAllocator::instance(getHost()));
    // NHWC: input shape is {N, H, W, C}
    Tensor input_host = to_host(input);

    size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
    size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;

    math_avgpool_fwd(input_host.data_as<float>(), expected_output.data_as<float>(),
                     stats.batch_size, stats.height, stats.width, stats.channels, stats.pool_h,
                     stats.pool_w, stats.stride_h, stats.stride_w, stats.pad_h, stats.pad_w,
                     output_h, output_w);

    compare_tensor(to_host(output), expected_output);
  }

  void check_avgpool_bwd(const Tensor& grad_output, const Tensor& grad_input,
                         const AvgPool2DStats& stats) {
    Tensor expected_grad_input(grad_input.shape(), grad_input.dtype(),
                               DeviceAllocator::instance(getHost()));
    fill(expected_grad_input, 0.0f);
    // NHWC: grad_output shape is {N, OH, OW, C}
    Tensor grad_out_host = to_host(grad_output);

    size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
    size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;

    math_avgpool_bwd(grad_out_host.data_as<float>(), expected_grad_input.data_as<float>(),
                     stats.batch_size, stats.height, stats.width, stats.channels, stats.pool_h,
                     stats.pool_w, stats.stride_h, stats.stride_w, stats.pad_h, stats.pad_w,
                     output_h, output_w);

    compare_tensor(to_host(grad_input), expected_grad_input);
    compare_tensor(to_host(grad_input), expected_grad_input);
  }

  void check_maxpool_fwd(const Tensor& input, const Tensor& output, const Tensor& mask,
                         const MaxPool2DStats& stats) {
    Tensor expected_output(output.shape(), output.dtype(), DeviceAllocator::instance(getHost()));
    Tensor expected_mask(mask.shape(), mask.dtype(), DeviceAllocator::instance(getHost()));
    Tensor input_host = to_host(input);

    size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
    size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;

    math_maxpool2d_fwd(input_host.data_as<float>(), expected_output.data_as<float>(),
                       expected_mask.data_as<int32_t>(), stats.batch_size, stats.height,
                       stats.width, stats.channels, stats.pool_h, stats.pool_w, stats.stride_h,
                       stats.stride_w, stats.pad_h, stats.pad_w, output_h, output_w);

    compare_tensor(to_host(output), expected_output);
    compare_tensor(to_host(mask), expected_mask);
  }

  void check_maxpool_bwd(const Tensor& grad_output, const Tensor& grad_input, const Tensor& mask,
                         const MaxPool2DStats& stats) {
    Tensor expected_grad_input(grad_input.shape(), grad_input.dtype(),
                               DeviceAllocator::instance(getHost()));
    fill(expected_grad_input, 0.0f);

    Tensor grad_out_host = to_host(grad_output);
    Tensor mask_host = to_host(mask);

    size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
    size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;

    math_maxpool2d_bwd(grad_out_host.data_as<float>(), expected_grad_input.data_as<float>(),
                       mask_host.data_as<int32_t>(), stats.batch_size, stats.channels, output_h,
                       output_w, stats.height, stats.width, stats.pool_w, stats.stride_h,
                       stats.stride_w, stats.pad_h, stats.pad_w);

    compare_tensor(to_host(grad_input), expected_grad_input);
  }

  void check_class_token_fwd(const Tensor& input, const Tensor& token, const Tensor& output,
                             const ClassTokenStats& stats) {
    Tensor expected_output(output.shape(), output.dtype(), DeviceAllocator::instance(getHost()));

    Tensor input_host = to_host(input);
    Tensor token_host = to_host(token);

    math_class_token_fwd(input_host.data_as<float>(), token_host.data_as<float>(),
                         expected_output.data_as<float>(), stats.batch_size, stats.seq_len,
                         stats.embed_dim);

    compare_tensor(to_host(output), expected_output);
  }

  void check_class_token_bwd(const Tensor& grad_output, const Tensor& grad_input,
                             const Tensor& grad_token, const ClassTokenStats& stats) {
    Tensor expected_grad_input(grad_input.shape(), grad_input.dtype(),
                               DeviceAllocator::instance(getHost()));
    Tensor expected_grad_token(grad_token.shape(), grad_token.dtype(),
                               DeviceAllocator::instance(getHost()));
    fill(expected_grad_token, 0.0f);

    Tensor grad_out_host = to_host(grad_output);

    math_class_token_bwd(grad_out_host.data_as<float>(), expected_grad_input.data_as<float>(),
                         expected_grad_token.data_as<float>(), stats.batch_size, stats.seq_len,
                         stats.embed_dim);

    compare_tensor(to_host(grad_input), expected_grad_input);
    compare_tensor(to_host(grad_token), expected_grad_token);
  }

  void check_dropout_fwd(const Tensor& input, const Tensor& mask, const Tensor& output,
                         const DropoutStats& stats) {
    Tensor expected_output(output.shape(), output.dtype(), DeviceAllocator::instance(getHost()));

    Tensor input_host = to_host(input);
    Tensor mask_host = to_host(mask);

    math_dropout_fwd(input_host.data_as<float>(), expected_output.data_as<float>(),
                     mask_host.data_as<bool>(), stats.batch_size, stats.channels,
                     stats.spatial_size, static_cast<float>(stats.dropout_rate));

    compare_tensor(to_host(output), expected_output);
  }

  void check_dropout_bwd(const Tensor& grad_output, const Tensor& mask, const Tensor& grad_input,
                         const DropoutStats& stats) {
    Tensor expected_grad_input(grad_input.shape(), grad_input.dtype(),
                               DeviceAllocator::instance(getHost()));

    Tensor grad_out_host = to_host(grad_output);
    Tensor mask_host = to_host(mask);

    math_dropout_bwd(grad_out_host.data_as<float>(), expected_grad_input.data_as<float>(),
                     mask_host.data_as<bool>(), stats.batch_size, stats.channels,
                     stats.spatial_size, static_cast<float>(1.0 / (1.0 - stats.dropout_rate)));

    compare_tensor(to_host(grad_input), expected_grad_input);
  }

  void check_embedding_fwd(const Tensor& input, const Tensor& weight, const Tensor& output,
                           const EmbeddingStats& stats) {
    Tensor expected_output(output.shape(), output.dtype(), DeviceAllocator::instance(getHost()));

    Tensor input_host = to_host(input);
    Tensor weight_host = to_host(weight);

    math_embedding_fwd(input_host.data_as<float>(), weight_host.data_as<float>(),
                       expected_output.data_as<float>(), stats.num_indices, stats.vocab_size,
                       stats.embed_dim, stats.padding_idx);

    compare_tensor(to_host(output), expected_output);
  }

  void check_embedding_bwd(const Tensor& input, const Tensor& grad_output,
                           const Tensor& grad_weight, const EmbeddingStats& stats) {
    Tensor expected_grad_weight(grad_weight.shape(), grad_weight.dtype(),
                                DeviceAllocator::instance(getHost()));
    fill(expected_grad_weight, 0.0f);

    Tensor input_host = to_host(input);
    Tensor grad_out_host = to_host(grad_output);

    math_embedding_bwd(input_host.data_as<float>(), grad_out_host.data_as<float>(),
                       expected_grad_weight.data_as<float>(), stats.num_indices, stats.vocab_size,
                       stats.embed_dim, stats.padding_idx);

    compare_tensor(to_host(grad_weight), expected_grad_weight);
  }

  static bool has_gpu_;
  static sref<Device> device_;
  static stream stream_;
  static engine_handle engine_handle_;
  static std::unique_ptr<CuDNNEngine> engine_;
};

bool CuDNNEngineTest::has_gpu_ = false;
sref<Device> CuDNNEngineTest::device_;
stream CuDNNEngineTest::stream_ = nullptr;
engine_handle CuDNNEngineTest::engine_handle_;
std::unique_ptr<CuDNNEngine> CuDNNEngineTest::engine_;

TEST_F(CuDNNEngineTest, DenseFwdThrowsWhenUncached) {
  DenseStats stats{
      .batch_size = 32,
      .in_features = 128,
      .out_features = 64,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };

  EXPECT_THROW(
      {
        engine_->dense_fwd(engine_handle_, stats, nullptr, nullptr, nullptr, nullptr, nullptr,
                           type_desc);
      },
      std::runtime_error);
}

TEST_F(CuDNNEngineTest, QueryDenseGraphReturnsValidWorkspace) {
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

  WorkspaceReq req = engine_->query_dense_graph(engine_handle_, stats, type_desc);

  EXPECT_GE(req.fwd_workspace, 0);
  EXPECT_GE(req.bwd_workspace, 0);
  EXPECT_EQ(req.inf_workspace, 0);
}

TEST_F(CuDNNEngineTest, DenseFwdReturnsCorrectResults) {
  DenseStats stats{
      .batch_size = 16,
      .in_features = 64,
      .out_features = 32,
      .use_bias = true,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };

  Tensor input({16, 64}, DType_t::FP32, getGPU());
  fill_normal(input, 0.0, 0.5, 12345ULL);
  Tensor weight({32, 64}, DType_t::FP32, getGPU());  // [output_features, input_features]
  fill_normal(weight, 0.0, 0.1, 12345ULL);
  Tensor bias({32}, DType_t::FP32, getGPU());
  fill_normal(bias, 0.0, 0.1, 12345ULL);
  Tensor output({16, 32}, DType_t::FP32, getGPU());

  WorkspaceReq req = engine_->query_dense_graph(engine_handle_, stats, type_desc);

  Tensor workspace({req.fwd_workspace}, DType_t::BYTE, getGPU());
  engine_->dense_fwd(engine_handle_, stats, input.data_as<void>(), weight.data_as<void>(),
                     bias.data_as<void>(), output.data_as<void>(), workspace.data_as<void>(),
                     type_desc);

  check_dense_fwd(input, weight, bias, output);
}

TEST_F(CuDNNEngineTest, DenseWgradReturnsCorrectResults) {
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
  Tensor grad_weight({out_features, in_features}, DType_t::FP32, getGPU());
  fill(grad_weight, 0.0f);

  WorkspaceReq req = engine_->query_dense_graph(engine_handle_, stats, type_desc);

  Tensor workspace({req.bwd_workspace}, DType_t::BYTE, getGPU());
  engine_->dense_wgrad(engine_handle_, stats, grad_output.data_as<void>(), input.data_as<void>(),
                       grad_weight.data_as<void>(), workspace.data_as<void>(), type_desc);

  check_dense_wgrad(input, grad_output, grad_weight);
}

TEST_F(CuDNNEngineTest, DenseDgradReturnsCorrectResults) {
  DenseStats stats{
      .batch_size = 16,
      .in_features = 64,
      .out_features = 32,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };

  Tensor grad_output({16, 32}, DType_t::FP32, getGPU());
  fill_normal(grad_output, 0.0, 0.5, 12345ULL);
  Tensor weight({32, 64}, DType_t::FP32, getGPU());  // [output_features, input_features]
  fill_normal(weight, 0.0, 0.1, 12345ULL);
  Tensor grad_input({16, 64}, DType_t::FP32, getGPU());

  WorkspaceReq req = engine_->query_dense_graph(engine_handle_, stats, type_desc);

  Tensor workspace({req.bwd_workspace}, DType_t::BYTE, getGPU());
  engine_->dense_dgrad(engine_handle_, stats, grad_output.data_as<void>(), weight.data_as<void>(),
                       grad_input.data_as<void>(), workspace.data_as<void>(), type_desc);

  check_dense_dgrad(grad_output, weight, grad_input);
}

TEST_F(CuDNNEngineTest, DenseBgradReturnsCorrectResults) {
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
  Tensor grad_bias({out_features}, DType_t::FP32, getGPU());
  fill(grad_bias, 0.0f);
  cudaDeviceSynchronize();

  WorkspaceReq req = engine_->query_dense_graph(engine_handle_, stats, type_desc);

  Tensor workspace({req.bwd_workspace}, DType_t::BYTE, getGPU());
  engine_->dense_bgrad(engine_handle_, stats, grad_output.data_as<void>(),
                       grad_bias.data_as<void>(), workspace.data_as<void>(), type_desc);

  check_dense_bgrad(grad_output, grad_bias);
}

TEST_F(CuDNNEngineTest, Conv2DWgradReturnsCorrectResult) {
  Conv2DStats stats{
      .batch_size = 2,
      .in_channels = 3,
      .out_channels = 4,
      .input_h = 8,
      .input_w = 8,
      .kernel_h = 3,
      .kernel_w = 3,
      .stride_h = 1,
      .stride_w = 1,
      .pad_h = 1,
      .pad_w = 1,
      .use_bias = true,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };

  // NHWC layout: {N, H, W, C}
  Tensor input({2, 8, 8, 3}, DType_t::FP32, getGPU());
  fill_normal(input, 0.0, 0.5, 12345ULL);
  Tensor grad_output({2, 8, 8, 4}, DType_t::FP32, getGPU());
  fill_normal(grad_output, 0.0, 0.2, 12345ULL);

  // Weight layout: {OC, KH, KW, IC}
  Tensor grad_weight({4, 3, 3, 3}, DType_t::FP32, getGPU());
  fill(grad_weight, 0.0f);

  WorkspaceReq req = engine_->query_conv2d_graph(engine_handle_, stats, type_desc);
  Tensor workspace({req.bwd_workspace}, DType_t::BYTE, getGPU());

  engine_->conv2d_wgrad(engine_handle_, stats, grad_output.data_as<void>(), input.data_as<void>(),
                        grad_weight.data_as<void>(), workspace.data_as<void>(), type_desc);

  check_conv2d_wgrad(input, grad_output, grad_weight, stats);
}

TEST_F(CuDNNEngineTest, Conv2DBgradReturnsCorrectResult) {
  size_t batch_size = 2;
  size_t in_channels = 3;
  size_t out_channels = 4;
  size_t input_h = 8;
  size_t input_w = 8;
  size_t kernel_h = 3;
  size_t kernel_w = 3;
  size_t stride_h = 1;
  size_t stride_w = 1;
  size_t pad_h = 1;
  size_t pad_w = 1;

  Conv2DStats stats{
      .batch_size = batch_size,
      .in_channels = in_channels,
      .out_channels = out_channels,
      .input_h = input_h,
      .input_w = input_w,
      .kernel_h = kernel_h,
      .kernel_w = kernel_w,
      .stride_h = stride_h,
      .stride_w = stride_w,
      .pad_h = pad_h,
      .pad_w = pad_w,
      .use_bias = true,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };

  size_t output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
  size_t output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;

  // NHWC layout: {N, H, W, C}
  Tensor grad_output({batch_size, output_h, output_w, out_channels}, DType_t::FP32, getGPU());
  fill_normal(grad_output, 0.0, 0.2, 12345ULL);

  Tensor grad_bias({out_channels}, DType_t::FP32, getGPU());
  fill(grad_bias, 0.0f);

  WorkspaceReq req = engine_->query_conv2d_graph(engine_handle_, stats, type_desc);
  Tensor workspace({req.bwd_workspace}, DType_t::BYTE, getGPU());

  engine_->conv2d_bgrad(engine_handle_, stats, grad_output.data_as<void>(),
                        grad_bias.data_as<void>(), workspace.data_as<void>(), type_desc);

  check_conv2d_bgrad(grad_output, grad_bias, stats);
}

TEST_F(CuDNNEngineTest, BatchNormBwdReturnsCorrectResult) {
  BatchNormStats stats{
      .batch_size = 2,
      .height = 8,
      .width = 8,
      .channels = 4,
      .epsilon = 1e-5,
      .momentum = 0.1,
      .use_relu = false,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };

  // NHWC layout: {N, H, W, C}
  Tensor input({2, 8, 8, 4}, DType_t::FP32, getGPU());
  fill_normal(input, 0.0, 0.5, 12345ULL);
  Tensor grad_output({2, 8, 8, 4}, DType_t::FP32, getGPU());
  fill_normal(grad_output, 0.0, 0.2, 12345ULL);
  Tensor gamma({4}, DType_t::FP32, getGPU());
  fill_normal(gamma, 1.0, 0.1, 12345ULL);
  Tensor mean({4}, DType_t::FP32, getGPU());
  fill(mean, 0.0f);
  Tensor invar({4}, DType_t::FP32, getGPU());
  fill(invar, 1.0f);

  Tensor grad_input({2, 8, 8, 4}, DType_t::FP32, getGPU());
  Tensor grad_gamma({4}, DType_t::FP32, getGPU());
  Tensor grad_beta({4}, DType_t::FP32, getGPU());
  fill(grad_gamma, 0.0);
  fill(grad_beta, 0.0);

  WorkspaceReq req = engine_->query_batchnorm_graph(engine_handle_, stats, type_desc);
  Tensor workspace({req.bwd_workspace}, DType_t::BYTE, getGPU());

  engine_->batchnorm_bwd(engine_handle_, stats, grad_output.data_as<void>(), input.data_as<void>(),
                         nullptr, gamma.data_as<void>(), grad_input.data_as<void>(),
                         grad_gamma.data_as<void>(), grad_beta.data_as<void>(),
                         mean.data_as<void>(), invar.data_as<void>(), workspace.data_as<void>(),
                         type_desc);

  check_batchnorm_bwd(grad_output, input, gamma, mean, invar, grad_input, grad_gamma, grad_beta,
                      stats);
}

TEST_F(CuDNNEngineTest, LayerNormBwdReturnsCorrectResult) {
  size_t batch_size = 2;
  size_t channels = 4;
  float epsilon = 1e-5f;

  LayerNormStats stats{
      .batch_size = batch_size,
      .channels = channels,
      .epsilon = epsilon,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };

  Tensor input({batch_size, channels}, DType_t::FP32, getGPU());
  fill_normal(input, 0.0, 0.5, 12345ULL);
  Tensor grad_output({batch_size, channels}, DType_t::FP32, getGPU());
  fill_normal(grad_output, 0.0, 0.2, 12345ULL);
  Tensor gamma({channels}, DType_t::FP32, getGPU());
  fill_normal(gamma, 1.0, 0.1, 12345ULL);

  // Run layernorm forward first to capture real mean and inv_variance
  Tensor beta({channels}, DType_t::FP32, getGPU());
  fill(beta, 0.0f);
  Tensor ln_output({batch_size, channels}, DType_t::FP32, getGPU());
  Tensor mean({batch_size, 1}, DType_t::FP32, getGPU());
  Tensor invar({batch_size, 1}, DType_t::FP32, getGPU());
  {
    WorkspaceReq fwd_req = engine_->query_layernorm_graph(engine_handle_, stats, type_desc);
    Tensor fwd_workspace({fwd_req.fwd_workspace > 0 ? fwd_req.fwd_workspace : 1}, DType_t::BYTE,
                         getGPU());
    engine_->layernorm_fwd(engine_handle_, stats, input.data_as<void>(), gamma.data_as<void>(),
                           beta.data_as<void>(), ln_output.data_as<void>(), mean.data_as<void>(),
                           invar.data_as<void>(), fwd_workspace.data_as<void>(), type_desc);
  }

  Tensor grad_input({batch_size, channels}, DType_t::FP32, getGPU());
  Tensor grad_gamma({channels}, DType_t::FP32, getGPU());
  Tensor grad_beta({channels}, DType_t::FP32, getGPU());
  fill(grad_gamma, 0.0f);
  fill(grad_beta, 0.0f);

  WorkspaceReq req = engine_->query_layernorm_graph(engine_handle_, stats, type_desc);
  Tensor workspace({req.bwd_workspace}, DType_t::BYTE, getGPU());

  engine_->layernorm_bwd(engine_handle_, stats, grad_output.data_as<void>(), input.data_as<void>(),
                         gamma.data_as<void>(), mean.data_as<void>(), invar.data_as<void>(),
                         grad_input.data_as<void>(), grad_gamma.data_as<void>(),
                         grad_beta.data_as<void>(), workspace.data_as<void>(), type_desc);

  check_layernorm_bwd(grad_output, input, gamma, grad_input, grad_gamma, grad_beta, stats);
}

TEST_F(CuDNNEngineTest, AvgPoolFwdReturnsCorrectResult) {
  AvgPool2DStats stats{
      .batch_size = 2,
      .height = 4,
      .width = 4,
      .channels = 3,
      .pool_h = 2,
      .pool_w = 2,
      .stride_h = 2,
      .stride_w = 2,
      .pad_h = 0,
      .pad_w = 0,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  WorkspaceReq req = engine_->query_avgpool_graph(engine_handle_, stats, type_desc);
  Tensor workspace({req.fwd_workspace > 0 ? req.fwd_workspace : 1}, DType_t::BYTE, getGPU());
  // NHWC layout: {N, H, W, C}
  Tensor input({2, 4, 4, 3}, DType_t::FP32, getGPU());
  fill_normal(input, 0.0, 0.5, 12345ULL);
  Tensor output({2, 2, 2, 3}, DType_t::FP32, getGPU());

  EXPECT_NO_THROW({
    engine_->avgpool_fwd(engine_handle_, stats, input.data_as<void>(), output.data_as<void>(),
                         workspace.data_as<void>(), type_desc);
  });

  check_avgpool_fwd(input, output, stats);
}

TEST_F(CuDNNEngineTest, AvgPoolBwdReturnsCorrectResult) {
  size_t batch_size = 2;
  size_t height = 4;
  size_t width = 4;
  size_t channels = 3;
  size_t pool_h = 2;
  size_t pool_w = 2;
  size_t stride_h = 2;
  size_t stride_w = 2;
  size_t pad_h = 0;
  size_t pad_w = 0;
  size_t output_h = (height + 2 * pad_h - pool_h) / stride_h + 1;
  size_t output_w = (width + 2 * pad_w - pool_w) / stride_w + 1;

  AvgPool2DStats stats{
      .batch_size = batch_size,
      .height = height,
      .width = width,
      .channels = channels,
      .pool_h = pool_h,
      .pool_w = pool_w,
      .stride_h = stride_h,
      .stride_w = stride_w,
      .pad_h = pad_h,
      .pad_w = pad_w,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  WorkspaceReq req = engine_->query_avgpool_graph(engine_handle_, stats, type_desc);
  Tensor workspace({req.bwd_workspace}, DType_t::BYTE, getGPU());
  // NHWC layout: {N, H, W, C}
  Tensor grad_output({batch_size, output_h, output_w, channels}, DType_t::FP32, getGPU());
  fill_normal(grad_output, 0.0, 0.5, 12345ULL);
  Tensor grad_input({batch_size, height, width, channels}, DType_t::FP32, getGPU());
  fill(grad_input, 0.0f);  // zero before accumulation

  EXPECT_NO_THROW({
    engine_->avgpool_bwd(engine_handle_, stats, grad_output.data_as<void>(),
                         grad_input.data_as<void>(), workspace.data_as<void>(), type_desc);
  });

  check_avgpool_bwd(grad_output, grad_input, stats);
}

TEST_F(CuDNNEngineTest, MaxPoolFwdReturnsCorrectResult) {
  size_t batch_size = 1;
  size_t height = 4;
  size_t width = 4;
  size_t channels = 1;
  size_t pool_h = 2;
  size_t pool_w = 2;
  size_t stride_h = 2;
  size_t stride_w = 2;
  size_t pad_h = 0;
  size_t pad_w = 0;
  size_t output_h = (height + 2 * pad_h - pool_h) / stride_h + 1;
  size_t output_w = (width + 2 * pad_w - pool_w) / stride_w + 1;

  MaxPool2DStats stats{
      .batch_size = batch_size,
      .height = height,
      .width = width,
      .channels = channels,
      .pool_h = pool_h,
      .pool_w = pool_w,
      .stride_h = stride_h,
      .stride_w = stride_w,
      .pad_h = pad_h,
      .pad_w = pad_w,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  WorkspaceReq req = engine_->query_maxpool2d_graph(engine_handle_, stats, type_desc);
  Tensor workspace({req.fwd_workspace > 0 ? req.fwd_workspace : 1}, DType_t::BYTE, getGPU());
  Tensor input({batch_size, height, width, channels}, DType_t::FP32, getGPU());
  float input_host[] = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  cudaMemcpy(input.data_as(), input_host, sizeof(input_host), cudaMemcpyHostToDevice);
  Tensor output({batch_size, output_h, output_w, channels}, DType_t::FP32, getGPU());
  Tensor mask({batch_size, output_h, output_w, channels}, DType_t::INT32, getGPU());

  EXPECT_NO_THROW({
    engine_->maxpool2d_fwd(engine_handle_, stats, input.data_as<void>(), output.data_as<void>(),
                           mask.data_as<void>(), workspace.data_as<void>(), type_desc);
  });

  check_maxpool_fwd(input, output, mask, stats);
}

TEST_F(CuDNNEngineTest, MaxPoolBwdReturnsCorrectResult) {
  size_t batch_size = 2;
  size_t height = 4;
  size_t width = 4;
  size_t channels = 3;
  size_t pool_h = 2;
  size_t pool_w = 2;
  size_t stride_h = 2;
  size_t stride_w = 2;
  size_t pad_h = 0;
  size_t pad_w = 0;
  size_t output_h = (height + 2 * pad_h - pool_h) / stride_h + 1;
  size_t output_w = (width + 2 * pad_w - pool_w) / stride_w + 1;

  MaxPool2DStats stats{
      .batch_size = batch_size,
      .height = height,
      .width = width,
      .channels = channels,
      .pool_h = pool_h,
      .pool_w = pool_w,
      .stride_h = stride_h,
      .stride_w = stride_w,
      .pad_h = pad_h,
      .pad_w = pad_w,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  WorkspaceReq req = engine_->query_maxpool2d_graph(engine_handle_, stats, type_desc);
  Tensor workspace({req.bwd_workspace > 0 ? req.bwd_workspace : 1}, DType_t::BYTE, getGPU());

  Tensor input({batch_size, height, width, channels}, DType_t::FP32, getGPU());
  fill_normal(input, 0.0, 0.5, 12345ULL);
  Tensor output({batch_size, output_h, output_w, channels}, DType_t::FP32, getGPU());
  Tensor mask({batch_size, output_h, output_w, channels}, DType_t::INT32, getGPU());

  engine_->maxpool2d_fwd(engine_handle_, stats, input.data_as<void>(), output.data_as<void>(),
                         mask.data_as<void>(), workspace.data_as<void>(), type_desc);

  Tensor grad_output({batch_size, output_h, output_w, channels}, DType_t::FP32, getGPU());
  fill_normal(grad_output, 0.0, 0.5, 12345ULL);
  Tensor grad_input({batch_size, height, width, channels}, DType_t::FP32, getGPU());
  fill(grad_input, 0.0f);

  EXPECT_NO_THROW({
    engine_->maxpool2d_bwd(engine_handle_, stats, grad_output.data_as<void>(),
                           grad_input.data_as<void>(), mask.data_as<void>(),
                           workspace.data_as<void>(), type_desc);
  });

  check_maxpool_bwd(grad_output, grad_input, mask, stats);
}

TEST_F(CuDNNEngineTest, ClassTokenFwdReturnsCorrectResult) {
  ClassTokenStats stats{
      .batch_size = 2,
      .seq_len = 3,
      .embed_dim = 4,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  WorkspaceReq req = engine_->query_class_token_graph(engine_handle_, stats, type_desc);
  Tensor workspace({req.fwd_workspace > 0 ? req.fwd_workspace : 1}, DType_t::BYTE, getGPU());

  Tensor input({2, 3, 4}, DType_t::FP32, getGPU());
  fill_normal(input, 0.0, 0.5, 12345ULL);
  Tensor token({2, 4}, DType_t::FP32, getGPU());
  Tensor output({2, 4, 4}, DType_t::FP32, getGPU());

  EXPECT_NO_THROW({
    engine_->class_token_fwd(engine_handle_, stats, input.data_as<void>(), token.data_as<void>(),
                             output.data_as<void>(), workspace.data_as<void>(), type_desc);
  });

  check_class_token_fwd(input, token, output, stats);
}

TEST_F(CuDNNEngineTest, ClassTokenBwdReturnsCorrectResult) {
  size_t batch_size = 2;
  size_t seq_len = 3;
  size_t embed_dim = 4;

  ClassTokenStats stats{
      .batch_size = batch_size,
      .seq_len = seq_len,
      .embed_dim = embed_dim,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  WorkspaceReq req = engine_->query_class_token_graph(engine_handle_, stats, type_desc);
  Tensor workspace({req.bwd_workspace > 0 ? req.bwd_workspace : 1}, DType_t::BYTE, getGPU());

  // Output seq_len is seq_len + 1 (prepended class token)
  size_t output_seq_len = seq_len + 1;
  Tensor grad_output({batch_size, output_seq_len, embed_dim}, DType_t::FP32, getGPU());
  fill_normal(grad_output, 0.0, 0.2, 12345ULL);
  Tensor grad_input({batch_size, seq_len, embed_dim}, DType_t::FP32, getGPU());
  Tensor grad_token({embed_dim}, DType_t::FP32, getGPU());
  fill(grad_token, 0.0f);

  engine_->class_token_bwd(engine_handle_, stats, grad_output.data_as<void>(),
                           grad_input.data_as<void>(), grad_token.data_as<void>(),
                           workspace.data_as<void>(), type_desc);

  check_class_token_bwd(grad_output, grad_input, grad_token, stats);
}

TEST_F(CuDNNEngineTest, DropoutFwdReturnsCorrectResult) {
  DropoutStats stats{
      .batch_size = 2,
      .channels = 3,
      .spatial_size = 4,
      .dropout_rate = 0.5,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  WorkspaceReq req = engine_->query_dropout_graph(engine_handle_, stats, type_desc);
  Tensor workspace({req.fwd_workspace > 0 ? req.fwd_workspace : 1}, DType_t::BYTE, getGPU());
  Tensor input({2, 3, 2, 2}, DType_t::FP32, getGPU());
  fill_normal(input, 0.0, 0.5, 12345ULL);
  Tensor output({2, 3, 2, 2}, DType_t::FP32, getGPU());
  Tensor mask({2, 3, 2, 2}, DType_t::BOOL, getGPU());

  EXPECT_NO_THROW({
    engine_->dropout_fwd(engine_handle_, stats, input.data_as<void>(), output.data_as<void>(),
                         mask.data_as<bool>(), workspace.data_as<void>(), type_desc);
  });

  // Copy mask to host via cudaMemcpy since DType_t::BOOL is unsupported by to_host()
  size_t mask_elements = 2 * 3 * 2 * 2;
  std::vector<bool> mask_host_vec(mask_elements);
  std::vector<uint8> mask_raw(mask_elements);
  cudaMemcpy(mask_raw.data(), mask.data_as<bool>(), mask_elements * sizeof(bool),
             cudaMemcpyDeviceToHost);
  Tensor input_host = to_host(input);
  Tensor output_host = to_host(output);
  float scale = 1.0f / (1.0f - static_cast<float>(stats.dropout_rate));
  const float* in_data = input_host.data_as<float>();
  const float* out_data = output_host.data_as<float>();
  for (size_t i = 0; i < mask_elements; ++i) {
    bool kept = static_cast<bool>(mask_raw[i]);
    float expected = kept ? in_data[i] * scale : 0.0f;
    EXPECT_NEAR(out_data[i], expected, 1e-3f) << fmt::format(
        "Dropout fwd mismatch at index {}: output={}, expected={}", i, out_data[i], expected);
  }
}

TEST_F(CuDNNEngineTest, DropoutBwdReturnsCorrectResult) {
  size_t batch_size = 2;
  size_t channels = 3;
  size_t spatial_size = 4;
  double dropout_rate = 0.5;
  float bwd_scale = 2.0f;
  // Spatial dims: spatial_size = spatial_h * spatial_w
  size_t spatial_h = 2;
  size_t spatial_w = 2;

  DropoutStats stats{
      .batch_size = batch_size,
      .channels = channels,
      .spatial_size = spatial_size,
      .dropout_rate = dropout_rate,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  WorkspaceReq req = engine_->query_dropout_graph(engine_handle_, stats, type_desc);
  Tensor workspace({req.bwd_workspace > 0 ? req.bwd_workspace : 1}, DType_t::BYTE, getGPU());

  Tensor grad_output({batch_size, channels, spatial_h, spatial_w}, DType_t::FP32, getGPU());
  fill_normal(grad_output, 0.0, 0.5, 12345ULL);
  Tensor grad_input({batch_size, channels, spatial_h, spatial_w}, DType_t::FP32, getGPU());
  Tensor mask({batch_size, channels, spatial_h, spatial_w}, DType_t::BOOL, getGPU());
  // Fill mask: alternate true/false
  size_t mask_elements = batch_size * channels * spatial_h * spatial_w;
  {
    std::vector<uint8> mask_raw(mask_elements);
    for (size_t i = 0; i < mask_elements; ++i) mask_raw[i] = static_cast<uint8>(i % 2 == 0);
    cudaMemcpy(mask.data_as<bool>(), mask_raw.data(), mask_elements * sizeof(bool),
               cudaMemcpyHostToDevice);
  }

  EXPECT_NO_THROW({
    engine_->dropout_bwd(engine_handle_, stats, grad_output.data_as<void>(),
                         grad_input.data_as<void>(), mask.data_as<bool>(), bwd_scale,
                         workspace.data_as<void>(), type_desc);
  });

  // Verify backward: grad_input[i] = mask[i] ? grad_output[i] * scale : 0
  std::vector<uint8> mask_raw(mask_elements);
  cudaMemcpy(mask_raw.data(), mask.data_as<bool>(), mask_elements * sizeof(bool),
             cudaMemcpyDeviceToHost);
  Tensor grad_output_host = to_host(grad_output);
  Tensor grad_input_host = to_host(grad_input);
  const float* go_data = grad_output_host.data_as<float>();
  const float* gi_data = grad_input_host.data_as<float>();
  for (size_t i = 0; i < mask_elements; ++i) {
    bool kept = static_cast<bool>(mask_raw[i]);
    float expected = kept ? go_data[i] * bwd_scale : 0.0f;
    EXPECT_NEAR(gi_data[i], expected, 1e-3f) << fmt::format(
        "Dropout bwd mismatch at index {}: grad_input={}, expected={}", i, gi_data[i], expected);
  }
}

TEST_F(CuDNNEngineTest, EmbeddingFwdReturnsCorrectResult) {
  EmbeddingStats stats{
      .num_indices = 4,
      .vocab_size = 10,
      .embed_dim = 8,
      .padding_idx = 0,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  WorkspaceReq req = engine_->query_embedding_graph(engine_handle_, stats, type_desc);
  Tensor workspace({req.fwd_workspace > 0 ? req.fwd_workspace : 1}, DType_t::BYTE, getGPU());

  Tensor input({4}, DType_t::FP32, getGPU());
  Tensor weight({10, 8}, DType_t::FP32, getGPU());
  Tensor output({4, 8}, DType_t::FP32, getGPU());

  EXPECT_NO_THROW({
    engine_->embedding_fwd(engine_handle_, stats, input.data_as<void>(), weight.data_as<void>(),
                           output.data_as<void>(), workspace.data_as<void>(), type_desc);
  });

  check_embedding_fwd(input, weight, output, stats);
}

TEST_F(CuDNNEngineTest, EmbeddingBwdReturnsCorrectResult) {
  size_t num_indices = 4;
  size_t vocab_size = 10;
  size_t embed_dim = 8;
  size_t padding_idx = 0;
  float host_input[] = {1.0f, 2.0f, 1.0f, 0.0f};

  EmbeddingStats stats{
      .num_indices = num_indices,
      .vocab_size = vocab_size,
      .embed_dim = embed_dim,
      .padding_idx = padding_idx,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  WorkspaceReq req = engine_->query_embedding_graph(engine_handle_, stats, type_desc);
  Tensor workspace({req.bwd_workspace > 0 ? req.bwd_workspace : 1}, DType_t::BYTE, getGPU());

  Tensor input({num_indices}, DType_t::FP32, getGPU());
  cudaMemcpy(input.data_as<void>(), host_input, sizeof(host_input), cudaMemcpyHostToDevice);
  Tensor grad_output({num_indices, embed_dim}, DType_t::FP32, getGPU());
  fill_normal(grad_output, 0.0, 0.2, 12345ULL);
  Tensor grad_weight({vocab_size, embed_dim}, DType_t::FP32, getGPU());
  fill(grad_weight, 0.0f);

  engine_->embedding_bwd(engine_handle_, stats, grad_output.data_as<void>(), input.data_as<void>(),
                         grad_weight.data_as<void>(), workspace.data_as<void>(), type_desc);

  check_embedding_bwd(input, grad_output, grad_weight, stats);
}

TEST_F(CuDNNEngineTest, ReLUFwdReturnsCorrectResult) {
  size_t batch_size = 16;
  size_t spatial_size = 256;  // must be >= 8 for cuDNN bool tensor alignment
  ReLUStats stats{
      .batch_size = batch_size,
      .spatial_size = spatial_size,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  WorkspaceReq req = engine_->query_relu_graph(engine_handle_, stats, type_desc);
  size_t ws_size = std::max(req.fwd_workspace, req.bwd_workspace);
  Tensor workspace({ws_size > 0 ? ws_size : 1}, DType_t::BYTE, getGPU());

  Tensor input({batch_size, spatial_size}, DType_t::FP32, getGPU());
  fill_normal(input, 0.0, 1.0, 12345ULL);

  Tensor output({batch_size, spatial_size}, DType_t::FP32, getGPU());

  EXPECT_NO_THROW({
    engine_->relu_fwd(engine_handle_, stats, input.data_as<void>(), output.data_as<void>(),
                      workspace.data_as<void>(), type_desc);
  });

  Tensor input_host = to_host(input);
  Tensor output_host = to_host(output);
  const float* in_data = input_host.data_as<float>();
  const float* out_data = output_host.data_as<float>();
  for (size_t i = 0; i < input.size(); ++i) {
    float expected = std::max(0.0f, in_data[i]);
    EXPECT_NEAR(out_data[i], expected, 1e-3f) << fmt::format(
        "ReLU fwd mismatch at index {}: output={}, expected={}", i, out_data[i], expected);
  }
}

TEST_F(CuDNNEngineTest, ReLUBwdReturnsCorrectResult) {
  size_t batch_size = 16;
  size_t spatial_size = 256;  // must be >= 8 for cuDNN bool tensor alignment

  ReLUStats stats{
      .batch_size = batch_size,
      .spatial_size = spatial_size,
  };
  DTypeDesc type_desc{
      .io_dtype = DType_t::FP32,
      .param_dtype = DType_t::FP32,
      .compute_dtype = DType_t::FP32,
  };
  WorkspaceReq req = engine_->query_relu_graph(engine_handle_, stats, type_desc);
  size_t ws_size = std::max(req.fwd_workspace, req.bwd_workspace);

  Tensor workspace({ws_size > 0 ? ws_size : 1}, DType_t::BYTE, getGPU());
  Tensor grad_output({batch_size, spatial_size}, DType_t::FP32, getGPU());
  fill(grad_output, 1.0f);

  // Construct a fake cached output on host then upload: every other element is positive.
  size_t mask_elements = batch_size * spatial_size;
  std::vector<float> cached_output_host(mask_elements);
  for (size_t i = 0; i < mask_elements; ++i) cached_output_host[i] = (i % 2 == 0) ? 1.0f : -1.0f;
  Tensor cached_output({batch_size, spatial_size}, DType_t::FP32, getGPU());
  cudaMemcpy(cached_output.data_as<float>(), cached_output_host.data(),
             mask_elements * sizeof(float), cudaMemcpyHostToDevice);

  Tensor grad_input({batch_size, spatial_size}, DType_t::FP32, getGPU());

  EXPECT_NO_THROW({
    engine_->relu_bwd(engine_handle_, stats, grad_output.data_as<void>(),
                      grad_input.data_as<void>(), cached_output.data_as<void>(),
                      workspace.data_as<void>(), type_desc);
  });

  Tensor grad_input_host = to_host(grad_input);
  const float* gi_data = grad_input_host.data_as<float>();
  for (size_t i = 0; i < mask_elements; ++i) {
    float expected = (i % 2 == 0) ? 1.0f : 0.0f;
    EXPECT_NEAR(gi_data[i], expected, 1e-3f) << fmt::format(
        "ReLU bwd mismatch at index {}: grad_input={}, expected={}", i, gi_data[i], expected);
  }
}

TEST_F(CuDNNEngineTest, TransposeReturnsCorrectResults) {
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

  WorkspaceReq req = engine_->query_transpose_graph(engine_handle_, stats, type_desc);
  size_t ws_size = req.fwd_workspace > 0 ? req.fwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getGPU());

  engine_->transpose(engine_handle_, stats, input.data_as<void>(), output.data_as<void>(),
                     workspace.data_as<void>(), type_desc);
  cudaDeviceSynchronize();

  Tensor expected_output({batch_size, seq_len, num_heads, head_dim}, DType_t::FP32, getHost());
  Tensor host_input = to_host(input);
  math_transpose(host_input.data_as<float>(), expected_output.data_as<float>(), stats.shape,
                 stats.ndim, stats.dim0, stats.dim1);

  compare_tensor(output, expected_output, 1e-4);
}

TEST_F(CuDNNEngineTest, SDPAFwdReturnsCorrectResults) {
  cudaGetLastError();
  size_t batch_size = 2;
  size_t num_heads = 4;
  size_t seq_len = 8;
  size_t head_dim = 16;

  AttentionStats stats{
      .batch_size = batch_size,
      .num_heads = num_heads,
      .seq_len = seq_len,
      .head_dim = head_dim,
      .attn_scale = 1.0f / static_cast<float>(sqrt(head_dim)),
      .is_causal = true,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::BF16,
      .param_dtype = DType_t::BF16,
      .compute_dtype = DType_t::FP32,
  };

  Tensor q({batch_size, num_heads, seq_len, head_dim}, DType_t::BF16, getGPU());
  Tensor k({batch_size, num_heads, seq_len, head_dim}, DType_t::BF16, getGPU());
  Tensor v({batch_size, num_heads, seq_len, head_dim}, DType_t::BF16, getGPU());
  fill_normal(q, 0.0, 1.0, 12345ULL);
  fill_normal(k, 0.0, 1.0, 12346ULL);
  fill_normal(v, 0.0, 1.0, 12347ULL);
  Tensor output({batch_size, num_heads, seq_len, head_dim}, DType_t::BF16, getGPU());
  Tensor stats_tensor({batch_size, num_heads, seq_len}, DType_t::FP32, getGPU());

  WorkspaceReq req = engine_->query_sdpa_graph(engine_handle_, stats, type_desc);
  size_t ws_size = req.fwd_workspace > 0 ? req.fwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getGPU());

  engine_->sdpa_fwd(engine_handle_, stats, q.data_as<void>(), k.data_as<void>(), v.data_as<void>(),
                    output.data_as<void>(), stats_tensor.data_as<void>(), workspace.data_as<void>(),
                    type_desc);
  cudaDeviceSynchronize();

  Tensor expected_output({batch_size, num_heads, seq_len, head_dim}, DType_t::FP32, getHost());
  Tensor expected_stats_tensor({batch_size, num_heads, seq_len}, DType_t::FP32, getHost());

  Tensor host_q = to_host(q);
  Tensor host_k = to_host(k);
  Tensor host_v = to_host(v);
  Tensor host_q_fp32({batch_size, num_heads, seq_len, head_dim}, DType_t::FP32, getHost());
  Tensor host_k_fp32({batch_size, num_heads, seq_len, head_dim}, DType_t::FP32, getHost());
  Tensor host_v_fp32({batch_size, num_heads, seq_len, head_dim}, DType_t::FP32, getHost());
  cast(host_q, host_q_fp32);
  cast(host_k, host_k_fp32);
  cast(host_v, host_v_fp32);

  math_sdpa_fwd(host_q_fp32.data_as<float>(), host_k_fp32.data_as<float>(),
                host_v_fp32.data_as<float>(), expected_output.data_as<float>(),
                expected_stats_tensor.data_as<float>(), batch_size, num_heads, seq_len, head_dim,
                1.0f / sqrt(head_dim), true);

  Tensor expected_output_bf16({batch_size, num_heads, seq_len, head_dim}, DType_t::BF16, getHost());
  cast(expected_output, expected_output_bf16);

  compare_tensor(output, expected_output_bf16, 5e-2);
  compare_tensor(stats_tensor, expected_stats_tensor, 5e-2);
}
TEST_F(CuDNNEngineTest, SDPABwdReturnsCorrectResults) {
  size_t batch_size = 2;
  size_t num_heads = 4;
  size_t seq_len = 8;
  size_t head_dim = 16;

  AttentionStats stats{
      .batch_size = batch_size,
      .num_heads = num_heads,
      .seq_len = seq_len,
      .head_dim = head_dim,
      .attn_scale = 1.0f / static_cast<float>(sqrt(head_dim)),
      .is_causal = true,
  };

  DTypeDesc type_desc{
      .io_dtype = DType_t::BF16,
      .param_dtype = DType_t::BF16,
      .compute_dtype = DType_t::FP32,
  };

  Tensor q({batch_size, num_heads, seq_len, head_dim}, DType_t::BF16, getGPU());
  Tensor k({batch_size, num_heads, seq_len, head_dim}, DType_t::BF16, getGPU());
  Tensor v({batch_size, num_heads, seq_len, head_dim}, DType_t::BF16, getGPU());
  Tensor do_({batch_size, num_heads, seq_len, head_dim}, DType_t::BF16, getGPU());
  fill_normal(q, 0.0, 1.0, 12345ULL);
  fill_normal(k, 0.0, 1.0, 12346ULL);
  fill_normal(v, 0.0, 1.0, 12347ULL);
  fill_normal(do_, 0.0, 1.0, 12348ULL);
  Tensor output({batch_size, num_heads, seq_len, head_dim}, DType_t::BF16, getGPU());
  Tensor stats_tensor({batch_size, num_heads, seq_len}, DType_t::FP32, getGPU());

  WorkspaceReq req = engine_->query_sdpa_graph(engine_handle_, stats, type_desc);
  size_t ws_size = req.fwd_workspace > 0 ? req.fwd_workspace : 1;
  Tensor workspace({ws_size}, DType_t::BYTE, getGPU());

  engine_->sdpa_fwd(engine_handle_, stats, q.data_as<void>(), k.data_as<void>(), v.data_as<void>(),
                    output.data_as<void>(), stats_tensor.data_as<void>(), workspace.data_as<void>(),
                    type_desc);
  cudaDeviceSynchronize();

  Tensor dq({batch_size, num_heads, seq_len, head_dim}, DType_t::BF16, getGPU());
  Tensor dk({batch_size, num_heads, seq_len, head_dim}, DType_t::BF16, getGPU());
  Tensor dv({batch_size, num_heads, seq_len, head_dim}, DType_t::BF16, getGPU());

  size_t bwd_ws_size = req.bwd_workspace > 0 ? req.bwd_workspace : 1;
  Tensor bwd_workspace({bwd_ws_size}, DType_t::BYTE, getGPU());

  engine_->sdpa_bwd(engine_handle_, stats, q.data_as<void>(), k.data_as<void>(), v.data_as<void>(),
                    output.data_as<void>(), do_.data_as<void>(), stats_tensor.data_as<void>(),
                    dq.data_as<void>(), dk.data_as<void>(), dv.data_as<void>(),
                    bwd_workspace.data_as<void>(), type_desc);
  cudaDeviceSynchronize();

  Tensor expected_dq({batch_size, num_heads, seq_len, head_dim}, DType_t::FP32, getHost());
  Tensor expected_dk({batch_size, num_heads, seq_len, head_dim}, DType_t::FP32, getHost());
  Tensor expected_dv({batch_size, num_heads, seq_len, head_dim}, DType_t::FP32, getHost());

  Tensor host_q = to_host(q);
  Tensor host_k = to_host(k);
  Tensor host_v = to_host(v);
  Tensor host_do = to_host(do_);
  Tensor host_output = to_host(output);

  Tensor host_q_fp32({batch_size, num_heads, seq_len, head_dim}, DType_t::FP32, getHost());
  Tensor host_k_fp32({batch_size, num_heads, seq_len, head_dim}, DType_t::FP32, getHost());
  Tensor host_v_fp32({batch_size, num_heads, seq_len, head_dim}, DType_t::FP32, getHost());
  Tensor host_do_fp32({batch_size, num_heads, seq_len, head_dim}, DType_t::FP32, getHost());
  Tensor host_output_fp32({batch_size, num_heads, seq_len, head_dim}, DType_t::FP32, getHost());

  cast(host_q, host_q_fp32);
  cast(host_k, host_k_fp32);
  cast(host_v, host_v_fp32);
  cast(host_do, host_do_fp32);
  cast(host_output, host_output_fp32);

  math_sdpa_bwd(host_q_fp32.data_as<float>(), host_k_fp32.data_as<float>(),
                host_v_fp32.data_as<float>(), host_output_fp32.data_as<float>(),
                host_do_fp32.data_as<float>(), expected_dq.data_as<float>(),
                expected_dk.data_as<float>(), expected_dv.data_as<float>(), batch_size, num_heads,
                seq_len, head_dim, 1.0f / sqrt(head_dim), true);

  Tensor expected_dq_bf16({batch_size, num_heads, seq_len, head_dim}, DType_t::BF16, getHost());
  Tensor expected_dk_bf16({batch_size, num_heads, seq_len, head_dim}, DType_t::BF16, getHost());
  Tensor expected_dv_bf16({batch_size, num_heads, seq_len, head_dim}, DType_t::BF16, getHost());

  cast(expected_dq, expected_dq_bf16);
  cast(expected_dk, expected_dk_bf16);
  cast(expected_dv, expected_dv_bf16);

  compare_tensor(dq, expected_dq_bf16, 5e-2);
  compare_tensor(dk, expected_dk_bf16, 5e-2);
  compare_tensor(dv, expected_dv_bf16, 5e-2);
}
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif
