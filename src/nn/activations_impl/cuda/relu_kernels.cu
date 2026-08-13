#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "nn/activations_impl/cuda/relu_kernels.hpp"
#include "type/cuda/vectorized_types.cuh"
#include "type/type.hpp"

#ifdef TUNX_USE_CUDA

namespace tunx {
namespace cuda {

constexpr int BLOCK_SIZE = 256;

template <typename T>
__global__ void relu_vec_kernel(const T* input, T* output, size_t n_vectors) {
  using VecT = typename VectoredTraits<T>::type;
  constexpr int VecSize = VectoredTraits<T>::size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_vectors) return;

  const VecT* input_vec = reinterpret_cast<const VecT*>(input);
  VecT* output_vec = reinterpret_cast<VecT*>(output);

  VecT in_val = input_vec[idx];
  VecT out_val;

  const T* in_ptr = reinterpret_cast<const T*>(&in_val);
  T* out_ptr = reinterpret_cast<T*>(&out_val);

  T zero = static_cast<T>(0);
  for (int i = 0; i < VecSize; ++i) {
    out_ptr[i] = (in_ptr[i] > zero) ? in_ptr[i] : zero;
  }

  output_vec[idx] = out_val;
}

template <typename T>
__global__ void relu_tail_kernel(const T* input, T* output, size_t offset, size_t tail_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tail_elements) return;

  size_t pos = offset + idx;
  T val = input[pos];
  T zero = static_cast<T>(0);
  output[pos] = (val > zero) ? val : zero;
}

template <typename T>
__global__ void relu_gradient_vec_kernel(const T* input, const T* grad_output, T* grad_input,
                                         size_t n_vectors) {
  using VecT = typename VectoredTraits<T>::type;
  constexpr int VecSize = VectoredTraits<T>::size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_vectors) return;

  const VecT* input_vec = reinterpret_cast<const VecT*>(input);
  const VecT* grad_out_vec = reinterpret_cast<const VecT*>(grad_output);
  VecT* grad_in_vec = reinterpret_cast<VecT*>(grad_input);

  VecT in_val = input_vec[idx];
  VecT grad_out_val = grad_out_vec[idx];
  VecT grad_in_val;

  const T* in_ptr = reinterpret_cast<const T*>(&in_val);
  const T* grad_out_ptr = reinterpret_cast<const T*>(&grad_out_val);
  T* grad_in_ptr = reinterpret_cast<T*>(&grad_in_val);

  T zero = static_cast<T>(0);
  for (int i = 0; i < VecSize; ++i) {
    grad_in_ptr[i] = (in_ptr[i] > zero) ? grad_out_ptr[i] : zero;
  }

  grad_in_vec[idx] = grad_in_val;
}

template <typename T>
__global__ void relu_gradient_tail_kernel(const T* input, const T* grad_output, T* grad_input,
                                          size_t offset, size_t tail_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tail_elements) return;

  size_t pos = offset + idx;
  T val = input[pos];
  T zero = static_cast<T>(0);
  grad_input[pos] = (val > zero) ? grad_output[pos] : zero;
}

template <typename T>
void relu_impl(const T* input, T* output, size_t size, cudaStream_t stream) {
  if (size == 0) return;

  constexpr int VecSize = VectoredTraits<T>::size;
  size_t n_vectors = size / VecSize;

  if (n_vectors > 0) {
    int num_blocks = (n_vectors + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_vec_kernel<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, n_vectors);
  }

  size_t tail_offset = n_vectors * VecSize;
  size_t tail_elements = size - tail_offset;
  if (tail_elements > 0) {
    int num_blocks = (tail_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_tail_kernel<T>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, tail_offset, tail_elements);
  }
}

template <typename T>
void relu_gradient_impl(const T* input, const T* grad_output, T* grad_input, size_t size,
                        cudaStream_t stream) {
  if (size == 0) return;

  constexpr int VecSize = VectoredTraits<T>::size;
  size_t n_vectors = size / VecSize;

  if (n_vectors > 0) {
    int num_blocks = (n_vectors + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_gradient_vec_kernel<T>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, grad_output, grad_input, n_vectors);
  }

  size_t tail_offset = n_vectors * VecSize;
  size_t tail_elements = size - tail_offset;
  if (tail_elements > 0) {
    int num_blocks = (tail_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_gradient_tail_kernel<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        input, grad_output, grad_input, tail_offset, tail_elements);
  }
}

void relu(DType_t dtype, const void* input, void* output, size_t size, cudaStream_t stream) {
  DISPATCH_DTYPE(dtype, T,
                 relu_impl<T>(static_cast<const T*>(input), static_cast<T*>(output), size, stream));
}

void relu_gradient(DType_t dtype, const void* input, const void* grad_output, void* grad_input,
                   size_t size, cudaStream_t stream) {
  DISPATCH_DTYPE(
      dtype, T,
      relu_gradient_impl<T>(static_cast<const T*>(input), static_cast<const T*>(grad_output),
                            static_cast<T*>(grad_input), size, stream));
}

}  // namespace cuda
}  // namespace tunx

#endif