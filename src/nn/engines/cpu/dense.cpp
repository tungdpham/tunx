
#include "internal.hpp"
#include "math/cpu/gemm.hpp"
#include "nn/engines/cpu_engine.hpp"
#include "threading/thread_handler.hpp"

namespace tunx {
namespace {

template <typename T>
void dense_fwd_impl(const T* input_data, const T* weight_data, T* output_data, size_t batch_size,
                    size_t input_features, size_t output_features) {
  cpu::legacy_gemm(input_data, weight_data, output_data, batch_size, output_features, input_features,
                   false, true, T(1.0), T(0.0));
}

template <typename T>
void dense_wgrad_impl(const T* input_data, const T* gradient_data, T* grad_weight_data,
                      size_t batch_size, size_t input_features, size_t output_features) {
  cpu::legacy_gemm(gradient_data, input_data, grad_weight_data, output_features, input_features,
                   batch_size, true, false, T(1.0), T(1.0));
}

template <typename T>
void dense_dgrad_impl(const T* gradient_data, const T* weight_data, T* grad_input_data,
                      size_t batch_size, size_t input_features, size_t output_features) {
  cpu::legacy_gemm(gradient_data, weight_data, grad_input_data, batch_size, input_features,
                   output_features, false, false, T(1.0), T(0.0));
}

template <typename T>
void dense_bgrad_impl(const T* current_grad_data, T* grad_bias_data, size_t batch_size,
                      size_t output_features) {
  parallel_for<size_t>(0, output_features, [&](size_t out_f) {
    T grad_sum = T(0);
    for (size_t n = 0; n < batch_size; ++n) {
      grad_sum += current_grad_data[n * output_features + out_f];
    }
    grad_bias_data[out_f] += grad_sum;
  });
}

template <typename T>
void add_bias_impl(T* output_data, const T* bias_data, size_t batch_size, size_t output_features) {
  parallel_for_2d(batch_size, output_features, [&](size_t n, size_t out_f) {
    output_data[n * output_features + out_f] += bias_data[out_f];
  });
}

}  // namespace

void CPUEngine::dense_fwd(engine_handle backend_handle, const DenseStats& stats, const void* input,
                          const void* weight, const void* bias, void* output, void* workspace,
                          DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    dense_fwd_impl<T>(static_cast<const T*>(input), static_cast<const T*>(weight),
                      static_cast<T*>(output), stats.batch_size, stats.in_features,
                      stats.out_features);
    if (bias) {
      add_bias_impl<T>(static_cast<T*>(output), static_cast<const T*>(bias), stats.batch_size,
                       stats.out_features);
    }
  });
}

void CPUEngine::dense_wgrad(engine_handle backend_handle, const DenseStats& stats,
                            const void* grad_output, const void* input, void* grad_weight,
                            void* workspace, DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    dense_wgrad_impl<T>(static_cast<const T*>(input), static_cast<const T*>(grad_output),
                        static_cast<T*>(grad_weight), stats.batch_size, stats.in_features,
                        stats.out_features);
  });
}

void CPUEngine::dense_dgrad(engine_handle backend_handle, const DenseStats& stats,
                            const void* grad_output, const void* weight, void* grad_input,
                            void* workspace, DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    dense_dgrad_impl<T>(static_cast<const T*>(grad_output), static_cast<const T*>(weight),
                        static_cast<T*>(grad_input), stats.batch_size, stats.in_features,
                        stats.out_features);
  });
}

void CPUEngine::dense_bgrad(engine_handle backend_handle, const DenseStats& stats,
                            const void* grad_output, void* grad_bias, void* workspace,
                            DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    dense_bgrad_impl<T>(static_cast<const T*>(grad_output), static_cast<T*>(grad_bias),
                        stats.batch_size, stats.out_features);
  });
}

void CPUEngine::legacy_dense_fwd(engine_handle backend_handle, const void* input,
                                 const void* weight, void* output, size_t batch_size,
                                 size_t in_features, size_t out_features, DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    cpu::legacy_gemm(static_cast<const T*>(input), static_cast<const T*>(weight),
                     static_cast<T*>(output), batch_size, out_features, in_features, false, true,
                     T(1.0), T(0.0));
  });
}

void CPUEngine::legacy_dense_wgrad(engine_handle backend_handle, const void* input,
                                   const void* grad_output, void* grad_weight, size_t batch_size,
                                   size_t in_features, size_t out_features, DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    cpu::legacy_gemm(static_cast<const T*>(grad_output), static_cast<const T*>(input),
                     static_cast<T*>(grad_weight), out_features, in_features, batch_size, true, false,
                     T(1.0), T(0.0));
  });
}

void CPUEngine::legacy_dense_dgrad(engine_handle backend_handle, const void* grad_output,
                                   const void* weight, void* grad_input, size_t batch_size,
                                   size_t in_features, size_t out_features, DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    cpu::legacy_gemm(static_cast<const T*>(grad_output), static_cast<const T*>(weight),
                     static_cast<T*>(grad_input), batch_size, in_features, out_features, false, false,
                     T(1.0), T(0.0));
  });
}

void CPUEngine::legacy_dense_bgrad(engine_handle backend_handle, const void* grad_output,
                                   void* grad_bias, size_t batch_size, size_t out_features,
                                   DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    const T* go = static_cast<const T*>(grad_output);
    T* gb = static_cast<T*>(grad_bias);
    for (size_t j = 0; j < out_features; ++j) {
      T sum = 0;
      for (size_t i = 0; i < batch_size; ++i) {
        sum += go[i * out_features + j];
      }
      gb[j] = sum;
    }
  });
}

void CPUEngine::legacy_dense_add_bias(engine_handle backend_handle, void* output, const void* bias,
                                      size_t batch_size, size_t out_features, DTypeDesc type_desc) {
  CHECK_HOMOGENEOUS_DTYPE(type_desc);
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    add_bias_impl<T>(static_cast<T*>(output), static_cast<const T*>(bias), batch_size,
                     out_features);
  });
}

WorkspaceReq CPUEngine::query_dense_graph(engine_handle backend_handle, const DenseStats& stats,
                                          DTypeDesc type_desc) {
  return {0, 0, 0};
}

}  // namespace tunx
