/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include <gtest/gtest.h>

#include "device/device_manager.hpp"
#include "device/pool_allocator.hpp"
#include "nn/blocks_impl/flash_attention_block.hpp"
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"
#include "test_graph_utils.hpp"
#include "type/type.hpp"

using namespace tunx;

class FlashAttentionBlockTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { initializeDefaultDevices(); }

  void SetUp() override {
    DeviceManager &manager = DeviceManager::instance();
    Vec<DeviceID> device_ids = manager.get_all();

    has_cuda_ = false;

    for (const DeviceID &id : device_ids) {
      Device &device = manager.get(id);
      if (device.device_type() == DeviceType::CUDA) {
        has_cuda_ = true;
        device_ = device;
        stream_ = device.default_stream();
        break;
      }
    }

    if (!has_cuda_) {
      GTEST_SKIP() << "No CUDA device available";
    }
  }

  bool has_cuda_;
  sref<Device> device_;
  stream stream_;
};

TEST_F(FlashAttentionBlockTest, ForwardShapeTest) {
  auto block = FlashAttentionBlock(64, 4, true, "test_attention");

  auto &allocator = PoolAllocator::instance(device_, stream_);
  GraphOpts opts{
      .io_dtype = DType_t::BF16,
      .param_dtype = DType_t::BF16,
      .compute_dtype = DType_t::FP32,
  };
  Graph graph = test::compile_single_layer(block, allocator, "input", "output", opts);

  Tensor input = Tensor({2, 10, 64}, DType_t::BF16, device_);
  fill(input, 1.0f);

  auto output = block.forward({input})[0];

  EXPECT_EQ(output.shape()[0], 2);
  EXPECT_EQ(output.shape()[1], 10);
  EXPECT_EQ(output.shape()[2], 64);
}

TEST_F(FlashAttentionBlockTest, BackwardShapeTest) {
  auto block = FlashAttentionBlock(64, 4, true, "test_attention");

  auto &allocator = PoolAllocator::instance(device_, stream_);
  GraphOpts opts{
      .io_dtype = DType_t::BF16,
      .param_dtype = DType_t::BF16,
      .compute_dtype = DType_t::FP32,
  };
  Graph graph = test::compile_single_layer(block, allocator, "input", "output", opts);

  Tensor input = Tensor({2, 10, 64}, DType_t::BF16, device_);
  fill(input, 1.0f);

  Residuals residuals;
  auto output = block.forward({input}, residuals)[0];

  Tensor grad_output = Tensor({2, 10, 64}, DType_t::BF16, device_);
  fill(grad_output, 1.0f);

  auto grad_input = block.backward({grad_output}, residuals)[0];

  EXPECT_EQ(grad_input.shape()[0], 2);
  EXPECT_EQ(grad_input.shape()[1], 10);
  EXPECT_EQ(grad_input.shape()[2], 64);
}

#include <random>

void ref_linear(const float* in, const float* w, const float* b, float* out,
                size_t batch, size_t seq_len, size_t in_dim, size_t out_dim) {
  for (size_t b_idx = 0; b_idx < batch; ++b_idx) {
    for (size_t s_idx = 0; s_idx < seq_len; ++s_idx) {
      for (size_t o_idx = 0; o_idx < out_dim; ++o_idx) {
        float sum = b ? b[o_idx] : 0.0f;
        for (size_t i_idx = 0; i_idx < in_dim; ++i_idx) {
          // Assume W is [out_dim, in_dim] or [in_dim, out_dim].
          // Standard PyTorch is [out_dim, in_dim] -> w[o_idx * in_dim + i_idx].
          // If it's [in_dim, out_dim], then w[i_idx * out_dim + o_idx].
          // We will try [out_dim, in_dim] first. If tunx uses [in_dim, out_dim], 
          // we'd need to transpose. But actually let's just do standard [out_dim, in_dim].
          sum += in[b_idx * seq_len * in_dim + s_idx * in_dim + i_idx] * 
                 w[o_idx * in_dim + i_idx]; // PyTorch style
        }
        out[b_idx * seq_len * out_dim + s_idx * out_dim + o_idx] = sum;
      }
    }
  }
}

Tensor compute_ref_mha(FlashAttentionBlock& block, const Tensor& input, size_t num_heads, bool is_causal) {
  Tensor host_in = to_host(input);
  auto params = block.params();
  Tensor w_q = to_host(params[0].data()), b_q = to_host(params[1].data());
  Tensor w_k = to_host(params[2].data()), b_k = to_host(params[3].data());
  Tensor w_v = to_host(params[4].data()), b_v = to_host(params[5].data());
  Tensor w_o = to_host(params[6].data()), b_o = to_host(params[7].data());

  size_t batch = input.shape()[0];
  size_t seq_len = input.shape()[1];
  size_t embed_dim = input.shape()[2];
  size_t head_dim = embed_dim / num_heads;

  std::vector<float> in_f32(batch * seq_len * embed_dim);
  const bf16* in_data = host_in.data_as<bf16>();
  for (size_t i = 0; i < in_f32.size(); ++i) {
    in_f32[i] = static_cast<float>(in_data[i]);
  }

  auto to_f32 = [](const Tensor& t) {
    std::vector<float> res(t.size());
    const bf16* ptr = t.data_as<bf16>();
    for(size_t i=0; i<res.size(); ++i) res[i] = static_cast<float>(ptr[i]);
    return res;
  };

  auto w_q_f = to_f32(w_q), b_q_f = to_f32(b_q);
  auto w_k_f = to_f32(w_k), b_k_f = to_f32(b_k);
  auto w_v_f = to_f32(w_v), b_v_f = to_f32(b_v);
  auto w_o_f = to_f32(w_o), b_o_f = to_f32(b_o);

  std::vector<float> Q(batch * seq_len * embed_dim);
  std::vector<float> K(batch * seq_len * embed_dim);
  std::vector<float> V(batch * seq_len * embed_dim);

  auto linear_fwd = [](const std::vector<float>& X, const std::vector<float>& W, const std::vector<float>& B,
                       std::vector<float>& Y, size_t B_, size_t S, size_t I, size_t O) {
    for (size_t b = 0; b < B_; ++b) {
      for (size_t s = 0; s < S; ++s) {
        for (size_t o = 0; o < O; ++o) {
          float sum = B[o];
          for (size_t i = 0; i < I; ++i) {
             sum += X[b * S * I + s * I + i] * W[o * I + i];
          }
          Y[b * S * O + s * O + o] = sum;
        }
      }
    }
  };

  linear_fwd(in_f32, w_q_f, b_q_f, Q, batch, seq_len, embed_dim, embed_dim);
  linear_fwd(in_f32, w_k_f, b_k_f, K, batch, seq_len, embed_dim, embed_dim);
  linear_fwd(in_f32, w_v_f, b_v_f, V, batch, seq_len, embed_dim, embed_dim);

  std::vector<float> Out(batch * seq_len * embed_dim);
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  for (size_t b = 0; b < batch; ++b) {
    for (size_t h = 0; h < num_heads; ++h) {
      for (size_t sq = 0; sq < seq_len; ++sq) {
        std::vector<float> scores(seq_len, 0.0f);
        float max_score = -1e9f;
        for (size_t sk = 0; sk < seq_len; ++sk) {
          if (is_causal && sk > sq) {
            scores[sk] = -1e9f;
          } else {
            float dot = 0.0f;
            for (size_t d = 0; d < head_dim; ++d) {
              float q_val = Q[b * seq_len * embed_dim + sq * embed_dim + h * head_dim + d];
              float k_val = K[b * seq_len * embed_dim + sk * embed_dim + h * head_dim + d];
              dot += q_val * k_val;
            }
            scores[sk] = dot * scale;
          }
          max_score = std::max(max_score, scores[sk]);
        }

        float sum_exp = 0.0f;
        for (size_t sk = 0; sk < seq_len; ++sk) {
          if (scores[sk] > -1e8f) {
            scores[sk] = std::exp(scores[sk] - max_score);
            sum_exp += scores[sk];
          } else {
            scores[sk] = 0.0f;
          }
        }
        for (size_t sk = 0; sk < seq_len; ++sk) {
          scores[sk] /= sum_exp;
        }

        for (size_t d = 0; d < head_dim; ++d) {
          float out_val = 0.0f;
          for (size_t sk = 0; sk < seq_len; ++sk) {
            float v_val = V[b * seq_len * embed_dim + sk * embed_dim + h * head_dim + d];
            out_val += scores[sk] * v_val;
          }
          Out[b * seq_len * embed_dim + sq * embed_dim + h * head_dim + d] = out_val;
        }
      }
    }
  }

  std::vector<float> FinalOut(batch * seq_len * embed_dim);
  linear_fwd(Out, w_o_f, b_o_f, FinalOut, batch, seq_len, embed_dim, embed_dim);

  Tensor ref_out = Tensor({batch, seq_len, embed_dim}, DType_t::BF16);
  bf16* ref_data = ref_out.data_as<bf16>();
  for (size_t i = 0; i < FinalOut.size(); ++i) {
    ref_data[i] = static_cast<bf16>(FinalOut[i]);
  }

  return to_device(ref_out, input.device());
}

TEST_F(FlashAttentionBlockTest, CompareAgainstReferenceForward) {
  size_t batch = 2;
  size_t seq_len = 16;
  size_t embed_dim = 64;
  size_t num_heads = 4;
  bool is_causal = true;

  auto block = FlashAttentionBlock(embed_dim, num_heads, is_causal, "test_attention_correctness");

  auto &allocator = PoolAllocator::instance(device_, stream_);
  GraphOpts opts{
      .io_dtype = DType_t::BF16,
      .param_dtype = DType_t::BF16,
      .compute_dtype = DType_t::FP32,
  };
  Graph graph = test::compile_single_layer(block, allocator, "input", "output", opts);

  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 0.1f);

  auto randomize_param = [&](Param& p) {
    Tensor host_t = to_host(p.data());
    bf16* data = host_t.data_as<bf16>();
    for (size_t i = 0; i < host_t.size(); ++i) {
      data[i] = static_cast<bf16>(dist(gen));
    }
    copy(host_t, p.data());
  };

  auto params = block.params();
  for (auto& p : params) {
    randomize_param(p);
  }

  Tensor host_input = Tensor({batch, seq_len, embed_dim}, DType_t::BF16);
  bf16* in_data = host_input.data_as<bf16>();
  for (size_t i = 0; i < host_input.size(); ++i) {
    in_data[i] = static_cast<bf16>(dist(gen));
  }
  Tensor input = to_device(host_input, device_);

  auto flash_out = block.forward({input})[0];
  Tensor ref_out = compute_ref_mha(block, input, num_heads, is_causal);

  Tensor host_flash = to_host(flash_out);
  Tensor host_ref = to_host(ref_out);

  const bf16* flash_data = host_flash.data_as<bf16>();
  const bf16* ref_data = host_ref.data_as<bf16>();

  float max_diff = 0.0f;
  for (size_t i = 0; i < host_flash.size(); ++i) {
    float f_val = static_cast<float>(flash_data[i]);
    float r_val = static_cast<float>(ref_data[i]);
    float diff = std::abs(f_val - r_val);
    max_diff = std::max(max_diff, diff);
    EXPECT_NEAR(f_val, r_val, 1e-2f) << "Mismatch at index " << i << " (f_val: " << f_val << ", r_val: " << r_val << ")";
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
