#include "nn/activations_impl/cuda/elu_kernels.hpp"
#include "type/type.hpp"

#ifdef TUNX_USE_CUDA

namespace tunx {
namespace cuda {

template <typename T>
void elu_impl(const T* input, T* output, size_t size, T alpha, cudaStream_t stream);

template <typename T>
void elu_gradient_impl(const T* input, const T* grad_output, T* grad_input, size_t size, T alpha,
                       cudaStream_t stream);

constexpr int BLOCK_SIZE = 256;

__global__ void elu_kernel(const float* input, float* output, size_t size, float alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx] > 0.0f ? input[idx] : alpha * (expf(input[idx]) - 1.0f);
  }
}

__global__ void elu_gradient_kernel(const float* input, const float* grad_output, float* grad_input,
                                    size_t size, float alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    grad_input[idx] =
        input[idx] > 0.0f ? grad_output[idx] : grad_output[idx] * alpha * expf(input[idx]);
  }
}

__global__ void elu_kernel_double(const double* input, double* output, size_t size, double alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx] > 0.0 ? input[idx] : alpha * (exp(input[idx]) - 1.0);
  }
}

__global__ void elu_gradient_kernel_double(const double* input, const double* grad_output,
                                           double* grad_input, size_t size, double alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    grad_input[idx] =
        input[idx] > 0.0 ? grad_output[idx] : grad_output[idx] * alpha * exp(input[idx]);
  }
}

template <>
void elu_impl<float>(const float* input, float* output, size_t size, float alpha,
                     cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  elu_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, output, size, alpha);
}

template <>
void elu_gradient_impl<float>(const float* input, const float* grad_output, float* grad_input,
                              size_t size, float alpha, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  elu_gradient_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, grad_output, grad_input, size,
                                                            alpha);
}

template <>
void elu_impl<double>(const double* input, double* output, size_t size, double alpha,
                      cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  elu_kernel_double<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, output, size, alpha);
}

template <>
void elu_gradient_impl<double>(const double* input, const double* grad_output, double* grad_input,
                               size_t size, double alpha, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  elu_gradient_kernel_double<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, grad_output, grad_input,
                                                                   size, alpha);
}

__global__ void elu_half_scalar_kernel(const fp16* input, fp16* output, size_t size, fp16 alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    fp16 zero = __float2half(0.0f);
    if (__hgt(input[idx], zero)) {
      output[idx] = input[idx];
    } else {
      output[idx] = __hmul(alpha, (hexp(input[idx]) - __float2half(1.0f)));
    }
  }
}

__global__ void elu_gradient_half_scalar_kernel(const fp16* input, const fp16* grad_output,
                                                fp16* grad_input, size_t size, fp16 alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    fp16 zero = __float2half(0.0f);
    if (__hgt(input[idx], zero)) {
      grad_input[idx] = grad_output[idx];
    } else {
      grad_input[idx] = __hmul(grad_output[idx], __hmul(alpha, hexp(input[idx])));
    }
  }
}

template <>
void elu_impl<fp16>(const fp16* input, fp16* output, size_t size, fp16 alpha, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  elu_half_scalar_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, output, size, alpha);
}

template <>
void elu_gradient_impl<fp16>(const fp16* input, const fp16* grad_output, fp16* grad_input,
                             size_t size, fp16 alpha, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  elu_gradient_half_scalar_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, grad_output,
                                                                        grad_input, size, alpha);
}

__global__ void elu_bf16_scalar_kernel(const bf16* input, bf16* output, size_t size, bf16 alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    bf16 zero = __float2bfloat16(0.0f);
    if (input[idx] > zero) {
      output[idx] = input[idx];
    } else {
      output[idx] = alpha * (hexp(input[idx]) - __float2bfloat16(1.0f));
    }
  }
}

__global__ void elu_gradient_bf16_scalar_kernel(const bf16* input, const bf16* grad_output,
                                                bf16* grad_input, size_t size, bf16 alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    bf16 zero = __float2bfloat16(0.0f);
    if (input[idx] > zero) {
      grad_input[idx] = grad_output[idx];
    } else {
      grad_input[idx] = grad_output[idx] * (alpha * hexp(input[idx]));
    }
  }
}

template <>
void elu_impl<bf16>(const bf16* input, bf16* output, size_t size, bf16 alpha, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  elu_bf16_scalar_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, output, size, alpha);
}

template <>
void elu_gradient_impl<bf16>(const bf16* input, const bf16* grad_output, bf16* grad_input,
                             size_t size, bf16 alpha, cudaStream_t stream) {
  const int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  elu_gradient_bf16_scalar_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, grad_output,
                                                                        grad_input, size, alpha);
}

void elu(DType_t dtype, const void* input, void* output, size_t size, double alpha,
         cudaStream_t stream) {
  DISPATCH_DTYPE(dtype, T,
                 elu_impl<T>(static_cast<const T*>(input), static_cast<T*>(output), size,
                             static_cast<T>(alpha), stream));
}

void elu_gradient(DType_t dtype, const void* input, const void* grad_output, void* grad_input,
                  size_t size, double alpha, cudaStream_t stream) {
  DISPATCH_DTYPE(
      dtype, T,
      elu_gradient_impl<T>(static_cast<const T*>(input), static_cast<const T*>(grad_output),
                           static_cast<T*>(grad_input), size, static_cast<T>(alpha), stream));
}

}  // namespace cuda
}  // namespace tunx

#endif
