/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/optimizers_impl/cuda/sgd_kernels.hpp"
#include "type/type.hpp"

#ifdef TUNX_USE_CUDA

namespace tunx {
namespace cuda {
namespace sgd {

template <typename T>
__global__ void update_sgd_kernel(T* params_data, const T* grads_data, size_t size,
                                  const float learning_rate, const float weight_decay) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float grad = static_cast<float>(grads_data[idx]);
    if (weight_decay > 0.0f) {
      grad += weight_decay * static_cast<float>(params_data[idx]);
    }
    params_data[idx] -= learning_rate * grad;
  }
}

template <typename T>
__global__ void update_sgd_momentum_kernel(T* params_data, const T* grads_data, T* velocity_data,
                                           size_t size, const float learning_rate,
                                           const float momentum, const float weight_decay) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float grad = static_cast<float>(grads_data[idx]);
    if (weight_decay > 0.0f) {
      grad += weight_decay * static_cast<float>(params_data[idx]);
    }
    velocity_data[idx] = momentum * static_cast<float>(velocity_data[idx]) - learning_rate * grad;
    params_data[idx] += velocity_data[idx];
  }
}

template <typename T>
void update_sgd(T* params_data, const T* grads_data, size_t size, const float learning_rate,
                const float weight_decay, cudaStream_t stream) {
  if (size == 0) return;
  const int threads_per_block = 256;
  const int num_blocks = (size + threads_per_block - 1) / threads_per_block;

  update_sgd_kernel<<<num_blocks, threads_per_block, 0, stream>>>(params_data, grads_data, size,
                                                                  learning_rate, weight_decay);
}

template <typename T>
void update_sgd_momentum(T* params_data, const T* grads_data, T* velocity_data, size_t size,
                         const float learning_rate, const float momentum, const float weight_decay,
                         cudaStream_t stream) {
  if (size == 0) return;
  const int threads_per_block = 256;
  const int num_blocks = (size + threads_per_block - 1) / threads_per_block;

  update_sgd_momentum_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      params_data, grads_data, velocity_data, size, learning_rate, momentum, weight_decay);
}

#define INSTANTIATE(T)                                                                         \
  template void update_sgd<T>(T * params_data, const T* grads_data, size_t size,               \
                              const float learning_rate, const float weight_decay,             \
                              cudaStream_t stream);                                            \
  template void update_sgd_momentum<T>(T * params_data, const T* grads_data, T* velocity_data, \
                                       size_t size, const float learning_rate,                 \
                                       const float momentum, const float weight_decay,         \
                                       cudaStream_t stream);
INSTANTIATE(fp16)
INSTANTIATE(bf16)
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

}  // namespace sgd
}  // namespace cuda
}  // namespace tunx

#endif
