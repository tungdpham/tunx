#include <cublas_v2.h>

#include "cuda/error_handler.cuh"
#include "math/cuda/gemm.hpp"
#include "type/type.hpp"

namespace tunx {
namespace cuda {

cublasHandle_t get_cublas_handle() {
  static cublasHandle_t handle = nullptr;
  if (!handle) {
    cublasCreate(&handle);
  }
  return handle;
}

template <typename T>
struct CudaType;
template <>
struct CudaType<int8> {
  static constexpr cudaDataType_t type = CUDA_R_8I;
};
template <>
struct CudaType<fp16> {
  static constexpr cudaDataType_t type = CUDA_R_16F;
};
template <>
struct CudaType<bf16> {
  static constexpr cudaDataType_t type = CUDA_R_16BF;
};
template <>
struct CudaType<float> {
  static constexpr cudaDataType_t type = CUDA_R_32F;
};
template <>
struct CudaType<double> {
  static constexpr cudaDataType_t type = CUDA_R_64F;
};
template <>
struct CudaType<int> {
  static constexpr cudaDataType_t type = CUDA_R_32I;
};

template <typename T>
struct CublasComputeType;
template <>
struct CublasComputeType<int8> {
  static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_32F;
};
template <>
struct CublasComputeType<fp16> {
  static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_16F;
};
template <>
struct CublasComputeType<bf16> {
  static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_32F;
};
template <>
struct CublasComputeType<float> {
  static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_32F;
};
template <>
struct CublasComputeType<double> {
  static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_64F;
};
template <>
struct CublasComputeType<int> {
  static constexpr cublasComputeType_t type = CUBLAS_COMPUTE_32I;
};

template <typename A_T, typename B_T, typename C_T, typename Compute_T>
void legacy_gemm_ex(const A_T* A, const B_T* B, C_T* C, size_t M, size_t N, size_t K, const bool transA,
             const bool transB, const Compute_T alpha, const Compute_T beta, size_t lda, size_t ldb,
             size_t ldc, cudaStream_t stream) {
  cublasHandle_t handle = get_cublas_handle();
  cublasSetStream(handle, stream);

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  cublasStatus_t status = cublasGemmEx(
      handle, opB, opA, N, M, K, &alpha, B, CudaType<B_T>::type, ldb, A, CudaType<A_T>::type, lda,
      &beta, C, CudaType<C_T>::type, ldc, CublasComputeType<Compute_T>::type, CUBLAS_GEMM_DEFAULT);
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cublasGemmEx failed with status: " + std::to_string(status));
  }
  cuda::checkCudaError(cudaGetLastError(), "gemm_ex", __FILE__, __LINE__);
}

template <typename A_T, typename B_T, typename C_T, typename Compute_T>
void legacy_gemm_strided_batched_ex(const A_T* A, const B_T* B, C_T* C, size_t M, size_t N, size_t K,
                             const bool transA, const bool transB, const Compute_T alpha,
                             const Compute_T beta, size_t lda, size_t ldb, size_t ldc,
                             size_t strideA, size_t strideB, size_t strideC, size_t batch_count,
                             cudaStream_t stream) {
  cublasHandle_t handle = get_cublas_handle();
  cublasSetStream(handle, stream);

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  cublasStatus_t status = cublasGemmStridedBatchedEx(
      handle, opB, opA, N, M, K, &alpha, B, CudaType<B_T>::type, ldb, strideB, A,
      CudaType<A_T>::type, lda, strideA, &beta, C, CudaType<C_T>::type, ldc, strideC, batch_count,
      CublasComputeType<Compute_T>::type, CUBLAS_GEMM_DEFAULT);
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cublasGemmStridedBatchedEx failed with status: " +
                             std::to_string(status));
  }
  cuda::checkCudaError(cudaGetLastError(), "gemm_strided_batched_ex", __FILE__, __LINE__);
}

static cudaDataType_t get_cuda_data_type(DType_t dtype) {
  switch (dtype) {
    case DType_t::INT8: return CUDA_R_8I;
    case DType_t::FP16: return CUDA_R_16F;
    case DType_t::BF16: return CUDA_R_16BF;
    case DType_t::FP32: return CUDA_R_32F;
    case DType_t::FP64: return CUDA_R_64F;
    case DType_t::INT32: return CUDA_R_32I;
    default: throw std::runtime_error("Unsupported CUDA data type in gemm_ex");
  }
}

static cublasComputeType_t get_cublas_compute_type(DType_t dtype) {
  switch (dtype) {
    case DType_t::FP16: return CUBLAS_COMPUTE_16F;
    case DType_t::BF16: return CUBLAS_COMPUTE_32F;
    case DType_t::FP32: return CUBLAS_COMPUTE_32F;
    case DType_t::FP64: return CUBLAS_COMPUTE_64F;
    case DType_t::INT32: return CUBLAS_COMPUTE_32I;
    default: throw std::runtime_error("Unsupported CUDA compute type in gemm_ex");
  }
}

void gemm_ex(const void* A, const void* B, void* C, size_t M, size_t N, size_t K,
             const bool transA, const bool transB, float alpha,
             float beta, size_t lda, size_t ldb, size_t ldc, 
             DType_t a_type, DType_t b_type, DType_t c_type, DType_t compute_type, cudaStream_t stream) {
  cublasHandle_t handle = get_cublas_handle();
  cublasSetStream(handle, stream);

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  cudaDataType_t cuA = get_cuda_data_type(a_type);
  cudaDataType_t cuB = get_cuda_data_type(b_type);
  cudaDataType_t cuC = get_cuda_data_type(c_type);
  cublasComputeType_t cuCompute = get_cublas_compute_type(compute_type);

  DISPATCH_ANY_DTYPE(compute_type, Compute_T, {
    Compute_T alpha_val = static_cast<Compute_T>(alpha);
    Compute_T beta_val = static_cast<Compute_T>(beta);
    cublasStatus_t status = cublasGemmEx(
        handle, opB, opA, N, M, K, &alpha_val, B, cuB, ldb, A, cuA, lda,
        &beta_val, C, cuC, ldc, cuCompute, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cublasGemmEx failed with status: " + std::to_string(status));
    }
  });
  cuda::checkCudaError(cudaGetLastError(), "gemm_ex", __FILE__, __LINE__);
}

void gemm_strided_batched_ex(const void* A, const void* B, void* C, size_t M, size_t N, size_t K,
                             const bool transA, const bool transB, float alpha,
                             float beta, size_t lda, size_t ldb, size_t ldc,
                             size_t strideA, size_t strideB, size_t strideC, size_t batch_count,
                             DType_t a_type, DType_t b_type, DType_t c_type, DType_t compute_type, cudaStream_t stream) {
  cublasHandle_t handle = get_cublas_handle();
  cublasSetStream(handle, stream);

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  cudaDataType_t cuA = get_cuda_data_type(a_type);
  cudaDataType_t cuB = get_cuda_data_type(b_type);
  cudaDataType_t cuC = get_cuda_data_type(c_type);
  cublasComputeType_t cuCompute = get_cublas_compute_type(compute_type);

  DISPATCH_ANY_DTYPE(compute_type, Compute_T, {
    Compute_T alpha_val = static_cast<Compute_T>(alpha);
    Compute_T beta_val = static_cast<Compute_T>(beta);
    cublasStatus_t status = cublasGemmStridedBatchedEx(
        handle, opB, opA, N, M, K, &alpha_val, B, cuB, ldb, strideB, A,
        cuA, lda, strideA, &beta_val, C, cuC, ldc, strideC, batch_count,
        cuCompute, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cublasGemmStridedBatchedEx failed with status: " + std::to_string(status));
    }
  });
  cuda::checkCudaError(cudaGetLastError(), "gemm_strided_batched_ex", __FILE__, __LINE__);
}

}  // namespace cuda
}  // namespace tunx
