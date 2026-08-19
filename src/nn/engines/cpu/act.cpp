#include <random>

#include "internal.hpp"
#include "nn/engines/cpu_engine.hpp"
#include "nn/engines/engine_handle.hpp"
#include "nn/engines/iengine.hpp"
#include "nn/stats/stats.hpp"
#include "threading/thread_handler.hpp"

namespace tunx {
namespace {

constexpr size_t DROPOUT_BLOCK_SIZE = 1024;

template <typename T>
void dropout_fwd_impl(const T* input_data, T* output_data, bool* mask_data, size_t batch_size,
                      size_t channels, size_t spatial_size, T dropout_rate) {
  T scale = T(1) / (T(1) - dropout_rate);
  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    size_t offset = (n * channels + c) * spatial_size;
    const T* input_ptr = input_data + offset;
    bool* mask_ptr = mask_data + offset;
    T* output_ptr = output_data + offset;
    thread_local std::mt19937 local_generator(std::random_device{}());
    thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    T rng_buffer[DROPOUT_BLOCK_SIZE];
    for (size_t i = 0; i < spatial_size; i += DROPOUT_BLOCK_SIZE) {
      size_t current_block_size = std::min(DROPOUT_BLOCK_SIZE, spatial_size - i);
      for (size_t j = 0; j < current_block_size; ++j) {
        rng_buffer[j] = static_cast<T>(dist(local_generator));
      }
      for (size_t j = 0; j < current_block_size; ++j) {
        T r = rng_buffer[j];
        T keep_mask = static_cast<T>(r >= dropout_rate);
        T final_mask = keep_mask * scale;
        mask_ptr[i + j] = static_cast<bool>(keep_mask);
        output_ptr[i + j] = input_ptr[i + j] * final_mask;
      }
    }
  });
}

template <typename T>
void dropout_bwd_impl(const T* grad_output_data, T* grad_input_data, const bool* mask_data,
                      size_t batch_size, size_t channels, size_t spatial_size, T scale) {
  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    size_t offset = (n * channels + c) * spatial_size;
    const T* grad_out_ptr = grad_output_data + offset;
    const bool* mask_ptr = mask_data + offset;
    T* grad_in_ptr = grad_input_data + offset;
    for (size_t i = 0; i < spatial_size; ++i) {
      grad_in_ptr[i] = mask_ptr[i] ? grad_out_ptr[i] * scale : T(0);
    }
  });
}

template <typename T>
void relu_fwd_impl(const T* input_data, T* output_data, size_t num_elements) {
  parallel_for<size_t>(0, num_elements, [&](size_t i) {
    output_data[i] = input_data[i] > T(0) ? input_data[i] : T(0);
  });
}

template <typename T>
void relu_inf_impl(const T* input_data, T* output_data, size_t num_elements) {
  parallel_for<size_t>(0, num_elements, [&](size_t i) {
    output_data[i] = input_data[i] > T(0) ? input_data[i] : T(0);
  });
}

template <typename T>
void relu_bwd_impl(const T* grad_output_data, T* grad_input_data, const T* output_data,
                   size_t num_elements) {
  parallel_for<size_t>(0, num_elements, [&](size_t i) {
    grad_input_data[i] = output_data[i] > T(0) ? grad_output_data[i] : T(0);
  });
}

}  // namespace

void CPUEngine::dropout_fwd(engine_handle backend_handle, const DropoutStats& stats,
                            const void* input, void* output, bool* mask, void* workspace,
                            DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    dropout_fwd_impl<T>(static_cast<const T*>(input), static_cast<T*>(output), mask,
                        stats.batch_size, stats.channels, stats.spatial_size,
                        static_cast<T>(stats.dropout_rate));
  });
}

void CPUEngine::dropout_bwd(engine_handle backend_handle, const DropoutStats& stats,
                            const void* grad_output, void* grad_input, const bool* mask,
                            double scale, void* workspace, DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    dropout_bwd_impl<T>(static_cast<const T*>(grad_output), static_cast<T*>(grad_input), mask,
                        stats.batch_size, stats.channels, stats.spatial_size,
                        static_cast<T>(scale));
  });
}

void CPUEngine::relu_fwd(engine_handle backend_handle, const ReLUStats& stats, const void* input,
                         void* output, void* workspace, DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    relu_fwd_impl<T>(static_cast<const T*>(input), static_cast<T*>(output),
                     stats.batch_size * stats.spatial_size);
  });
}

void CPUEngine::relu_inf(engine_handle backend_handle, const ReLUStats& stats, const void* input,
                         void* output, void* workspace, DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    relu_inf_impl<T>(static_cast<const T*>(input), static_cast<T*>(output),
                     stats.batch_size * stats.spatial_size);
  });
}

void CPUEngine::relu_bwd(engine_handle backend_handle, const ReLUStats& stats,
                         const void* grad_output, void* grad_input, const void* output,
                         void* workspace, DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    relu_bwd_impl<T>(static_cast<const T*>(grad_output), static_cast<T*>(grad_input),
                     static_cast<const T*>(output), stats.batch_size * stats.spatial_size);
  });
}

WorkspaceReq CPUEngine::query_dropout_graph(engine_handle backend_handle, const DropoutStats&,
                                            DTypeDesc) {
  return {0, 0, 0};
}

WorkspaceReq CPUEngine::query_relu_graph(engine_handle backend_handle, const ReLUStats&,
                                         DTypeDesc) {
  return {0, 0, 0};
}

}  // namespace tunx
