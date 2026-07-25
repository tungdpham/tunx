#include "kernel/cuda/kernels.hpp"
#include "type/type.hpp"

#ifdef TUNX_USE_CUDA

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include <cstdint>
#include <type_traits>

#include "cuda/error_handler.cuh"
#include "type/cuda/vectorized_types.cuh"

namespace tunx {
namespace kernel {
namespace cuda {

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

inline int get_num_blocks(size_t size) { return (size + BLOCK_SIZE - 1) / BLOCK_SIZE; }

template <typename T, typename Func>
__global__ void binary_op_kernel(const T* __restrict__ a, const T* __restrict__ b,
                                 T* __restrict__ c, size_t size, Func op) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  using VecT = typename VectoredTraits<T>::type;
  constexpr int vec_size = VectoredTraits<T>::size;
  size_t vec_idx = idx * vec_size;

  if (vec_idx + vec_size <= size) {
    VecT va = reinterpret_cast<const VecT*>(a)[idx];
    VecT vb = reinterpret_cast<const VecT*>(b)[idx];
    VecT vc;
    if constexpr (vec_size == 4) {
      vc.x = op(va.x, vb.x);
      vc.y = op(va.y, vb.y);
      vc.z = op(va.z, vb.z);
      vc.w = op(va.w, vb.w);
    } else if constexpr (vec_size == 2) {
      vc.x = op(va.x, vb.x);
      vc.y = op(va.y, vb.y);
    } else if constexpr (vec_size == 1) {
      vc.x = op(va.x, vb.x);
    }
    reinterpret_cast<VecT*>(c)[idx] = vc;
  }
}

template <typename T, typename Func>
__global__ void binary_op_scalar_kernel(const T* a, const T* b, T* c, size_t size, Func op) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) c[idx] = op(a[idx], b[idx]);
}

template <typename T, typename Func>
__global__ void unary_op_kernel(const T* __restrict__ a, T* __restrict__ c, size_t size, Func op) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  using VecT = typename VectoredTraits<T>::type;
  constexpr size_t vec_size = size_t(VectoredTraits<T>::size);
  if (idx * vec_size + vec_size <= size) {
    VecT va = reinterpret_cast<const VecT*>(a)[idx];
    VecT vc;
    if constexpr (vec_size == 4) {
      vc.x = op(va.x);
      vc.y = op(va.y);
      vc.z = op(va.z);
      vc.w = op(va.w);
    } else if constexpr (vec_size == 2) {
      vc.x = op(va.x);
      vc.y = op(va.y);
    } else if constexpr (vec_size == 1) {
      vc.x = op(va.x);
    }
    reinterpret_cast<VecT*>(c)[idx] = vc;
  }
}

template <typename T, typename Func>
__global__ void unary_op_scalar_kernel(const T* a, T* c, size_t size, Func op) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) c[idx] = op(a[idx]);
}

template <typename T>
__global__ void set_scalar_vector_kernel(T* __restrict__ c, T scalar, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  using VecT = typename VectoredTraits<T>::type;
  constexpr size_t vec_size = size_t(VectoredTraits<T>::size);
  if (idx * vec_size + vec_size <= size) {
    VecT vc;
    if constexpr (vec_size == 4) {
      vc.x = scalar;
      vc.y = scalar;
      vc.z = scalar;
      vc.w = scalar;
    } else if constexpr (vec_size == 2) {
      vc.x = scalar;
      vc.y = scalar;
    } else if constexpr (vec_size == 1) {
      vc.x = scalar;
    }
    reinterpret_cast<VecT*>(c)[idx] = vc;
  }
}

template <typename T>
__global__ void set_scalar_kernel(T* c, T scalar, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) c[idx] = scalar;
}

template <typename T>
__global__ void axpy_kernel(T alpha, const T* x, T* y, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) y[idx] += alpha * x[idx];
}

template <typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

template <typename T, typename AccT, int Mode>
__global__ void reduce_kernel(const T* a, const T* b, AccT scalar, AccT* result, size_t size) {
  AccT sum = AccT(0);
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = idx; i < size; i += stride) {
    AccT val = AccT(0);
    if constexpr (Mode == 0) {
      val = static_cast<AccT>(a[i]);
    } else if constexpr (Mode == 1) {
      val = static_cast<AccT>(a[i]) * static_cast<AccT>(b[i]);
    } else if constexpr (Mode == 2) {
      AccT diff = static_cast<AccT>(a[i]) - scalar;
      val = diff * diff;
    }
    sum += val;
  }

  sum = warp_reduce_sum(sum);

  static __shared__ AccT shared[WARP_SIZE];
  int lane = threadIdx.x % WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;

  if (lane == 0) shared[warp] = sum;
  __syncthreads();

  sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : AccT(0);
  if (warp == 0) sum = warp_reduce_sum(sum);

  if (threadIdx.x == 0) result[blockIdx.x] = sum;
}

template <typename T>
__global__ void fill_random_uniform_kernel(T* data, size_t size, double min_val, double max_val,
                                           unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);
    if constexpr (std::is_same<T, double>::value) {
      data[idx] = min_val + curand_uniform_double(&state) * (max_val - min_val);
    } else {
      float val = curand_uniform(&state);
      if constexpr (std::is_same<T, bf16>::value) {
        data[idx] = __float2bfloat16(__bfloat162float(min_val) +
                                     val * (__bfloat162float(max_val) - __bfloat162float(min_val)));
      } else {
        data[idx] = min_val + val * (max_val - min_val);
      }
    }
  }
}

template <typename T>
__global__ void fill_random_normal_kernel(T* data, size_t size, double mean, double stddev,
                                          unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, 0, &state);

  for (size_t i = idx; i < size; i += blockDim.x * gridDim.x) {
    if constexpr (std::is_same<T, float>::value) {
      float val = curand_normal(&state);
      data[i] = mean + stddev * val;
    } else if constexpr (std::is_same<T, double>::value) {
      double val = curand_normal_double(&state);
      data[i] = mean + stddev * val;
    } else if constexpr (std::is_same<T, fp16>::value) {
      float val = curand_normal(&state);
      data[i] = __float2half(mean + stddev * val);
    } else if constexpr (std::is_same<T, bf16>::value) {
      float val = curand_normal(&state);
      data[i] = __float2bfloat16(mean + stddev * val);
    }
  }
}

template <>
__global__ void fill_random_normal_kernel<float>(float* data, size_t size, double mean,
                                                 double stddev, unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, 0, &state);

  size_t vec_size = size / 4;
  size_t vec_stride = blockDim.x * gridDim.x;

  for (size_t i = idx; i < vec_size; i += vec_stride) {
    float4 r = curand_normal4(&state);
    float4 res;
    res.x = mean + stddev * r.x;
    res.y = mean + stddev * r.y;
    res.z = mean + stddev * r.z;
    res.w = mean + stddev * r.w;
    reinterpret_cast<float4*>(data)[i] = res;
  }

  size_t remainder_start = vec_size * 4;
  if (idx == 0) {
    curandStatePhilox4_32_10_t state_rem;
    curand_init(seed, remainder_start, 0, &state_rem);
    for (size_t i = remainder_start; i < size; ++i) {
      data[i] = mean + stddev * curand_normal(&state_rem);
    }
  }
}

template <typename T, typename Func>
void dispatch_binary(const T* a, const T* b, T* c, size_t size, cudaStream_t stream, Func op) {
  if (size == 0) return;
  constexpr int vec_size = VectoredTraits<T>::size;
  bool is_aligned =
      ((uintptr_t)a % 16 == 0) && ((uintptr_t)b % 16 == 0) && ((uintptr_t)c % 16 == 0);

  if (is_aligned && size % vec_size == 0) {
    int blocks = get_num_blocks(size / vec_size);
    binary_op_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size, op);
  } else {
    int blocks = get_num_blocks(size);
    binary_op_scalar_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size, op);
  }
  tunx::cuda::checkCudaError(cudaGetLastError(), "binary_op", __FILE__, __LINE__);
}

template <typename T, typename Func>
void dispatch_unary(const T* a, T* c, size_t size, cudaStream_t stream, Func op) {
  if (size == 0) return;
  constexpr int vec_size = VectoredTraits<T>::size;
  bool is_aligned = ((uintptr_t)a % 16 == 0) && ((uintptr_t)c % 16 == 0);

  if (is_aligned && size % vec_size == 0) {
    int blocks = get_num_blocks(size / vec_size);
    unary_op_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, c, size, op);
  } else {
    int blocks = get_num_blocks(size);
    unary_op_scalar_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, c, size, op);
  }
  tunx::cuda::checkCudaError(cudaGetLastError(), "unary_op", __FILE__, __LINE__);
}

template <typename T, typename Func>
__global__ void ternary_op_kernel(const T* a, const T* b, T* c, size_t size, Func op) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) c[idx] = op(a[idx], b[idx], c[idx]);
}

template <typename T, typename Func>
void dispatch_ternary(const T* a, const T* b, T* c, size_t size, cudaStream_t stream, Func op) {
  if (size == 0) return;
  int blocks = get_num_blocks(size);
  ternary_op_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, size, op);
  tunx::cuda::checkCudaError(cudaGetLastError(), "ternary_op", __FILE__, __LINE__);
}

template <typename T, int Mode>
T dispatch_reduce(const T* a, const T* b, T scalar, size_t size, cudaStream_t stream) {
  if (size == 0) return (T)0;
  int blocks = std::min(get_num_blocks(size), 1024);

  using AccT = std::conditional_t<std::is_same<T, fp16>::value, float, T>;

  AccT* d_partial;
  cudaMalloc(&d_partial, blocks * sizeof(AccT));
  reduce_kernel<T, AccT, Mode>
      <<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, static_cast<AccT>(scalar), d_partial, size);

  AccT* h_partial = new AccT[blocks];
  cudaMemcpyAsync(h_partial, d_partial, blocks * sizeof(AccT), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // Use double precision for accumulation to avoid overflow/underflow with fp16
  double result = 0.0;
  for (int i = 0; i < blocks; ++i) result += static_cast<double>(h_partial[i]);

  delete[] h_partial;
  cudaFree(d_partial);
  tunx::cuda::checkCudaError(cudaGetLastError(), "reduction", __FILE__, __LINE__);

  // For fp16, clamp the result to avoid overflow when converting back
  if constexpr (std::is_same<T, fp16>::value) {
    // fp16 max value is approximately 65504
    const double fp16_max = 65504.0;
    if (result > fp16_max) result = fp16_max;
    if (result < -fp16_max) result = -fp16_max;
  }

  return static_cast<T>(result);
}

void add(DType_t dtype, const void* a, const void* b, void* c, size_t size, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    const T* b_cast = static_cast<const T*>(b);
    T* c_cast = static_cast<T*>(c);
    dispatch_binary(a_cast, b_cast, c_cast, size, stream, functors::Add<T>());
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void sub(DType_t dtype, const void* a, const void* b, void* c, size_t size, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    const T* b_cast = static_cast<const T*>(b);
    T* c_cast = static_cast<T*>(c);
    dispatch_binary(a_cast, b_cast, c_cast, size, stream, functors::Sub<T>());
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void mul(DType_t dtype, const void* a, const void* b, void* c, size_t size, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    const T* b_cast = static_cast<const T*>(b);
    T* c_cast = static_cast<T*>(c);
    dispatch_binary(a_cast, b_cast, c_cast, size, stream, functors::Mul<T>());
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void div(DType_t dtype, const void* a, const void* b, void* c, size_t size, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    const T* b_cast = static_cast<const T*>(b);
    T* c_cast = static_cast<T*>(c);
    dispatch_binary(a_cast, b_cast, c_cast, size, stream, functors::Div<T>());
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void min(DType_t dtype, const void* a, const void* b, void* c, size_t size, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    const T* b_cast = static_cast<const T*>(b);
    T* c_cast = static_cast<T*>(c);
    dispatch_binary(a_cast, b_cast, c_cast, size, stream, functors::Min<T>());
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void max(DType_t dtype, const void* a, const void* b, void* c, size_t size, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    const T* b_cast = static_cast<const T*>(b);
    T* c_cast = static_cast<T*>(c);
    dispatch_binary(a_cast, b_cast, c_cast, size, stream, functors::Max<T>());
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void equal(DType_t dtype, const void* a, const void* b, void* c, size_t size, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    const T* b_cast = static_cast<const T*>(b);
    T* c_cast = static_cast<T*>(c);
    dispatch_binary(a_cast, b_cast, c_cast, size, stream, functors::Equal<T>());
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void greater(DType_t dtype, const void* a, const void* b, void* c, size_t size,
             cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    const T* b_cast = static_cast<const T*>(b);
    T* c_cast = static_cast<T*>(c);
    dispatch_binary(a_cast, b_cast, c_cast, size, stream, functors::Greater<T>());
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void add_scalar(DType_t dtype, const void* a, double scalar, void* c, size_t size,
                cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    T* c_cast = static_cast<T*>(c);
    T s_cast = static_cast<T>(scalar);
    dispatch_unary(a_cast, c_cast, size, stream, functors::AddScalar<T>{s_cast});
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void sub_scalar(DType_t dtype, const void* a, double scalar, void* c, size_t size,
                cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    T* c_cast = static_cast<T*>(c);
    T s_cast = static_cast<T>(scalar);
    dispatch_unary(a_cast, c_cast, size, stream, functors::SubScalar<T>{s_cast});
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void mul_scalar(DType_t dtype, const void* a, double scalar, void* c, size_t size,
                cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    T* c_cast = static_cast<T*>(c);
    T s_cast = static_cast<T>(scalar);
    dispatch_unary(a_cast, c_cast, size, stream, functors::MulScalar<T>{s_cast});
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void div_scalar(DType_t dtype, const void* a, double scalar, void* c, size_t size,
                cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    T* c_cast = static_cast<T*>(c);
    T s_cast = static_cast<T>(scalar);
    dispatch_unary(a_cast, c_cast, size, stream, functors::DivScalar<T>{s_cast});
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void scalar_max(DType_t dtype, const void* a, double scalar, void* c, size_t size,
                cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    T* c_cast = static_cast<T*>(c);
    T s_cast = static_cast<T>(scalar);
    dispatch_unary(a_cast, c_cast, size, stream, functors::ScalarMax<T>{s_cast});
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void clamp(DType_t dtype, const void* a, double min_val, double max_val, void* c, size_t size,
           cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    T* c_cast = static_cast<T*>(c);
    T min_cast = static_cast<T>(min_val);
    T max_cast = static_cast<T>(max_val);
    dispatch_unary(a_cast, c_cast, size, stream, functors::Clamp<T>{min_cast, max_cast});
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void sub_mul_scalar(DType_t dtype, const void* a, double sub_scalar, double mul_scalar, void* c,
                    size_t size, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    T* c_cast = static_cast<T*>(c);
    T sub_cast = static_cast<T>(sub_scalar);
    T mul_cast = static_cast<T>(mul_scalar);
    dispatch_unary(a_cast, c_cast, size, stream, functors::SubMulScalar<T>{sub_cast, mul_cast});
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void mul_add_scalar(DType_t dtype, const void* a, double mul_scalar, double add_scalar, void* c,
                    size_t size, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    T* c_cast = static_cast<T*>(c);
    T mul_cast = static_cast<T>(mul_scalar);
    T add_cast = static_cast<T>(add_scalar);
    dispatch_unary(a_cast, c_cast, size, stream, functors::MulAddScalar<T>{mul_cast, add_cast});
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void axpy(DType_t dtype, double alpha, const void* x, void* y, size_t size, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* x_cast = static_cast<const T*>(x);
    T* y_cast = static_cast<T*>(y);
    T alpha_cast = static_cast<T>(alpha);
    int blocks = get_num_blocks(size);
    axpy_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(alpha_cast, x_cast, y_cast, size);
    tunx::cuda::checkCudaError(cudaGetLastError(), "axpy", __FILE__, __LINE__);
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void sqrt(DType_t dtype, const void* a, void* c, size_t size, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    T* c_cast = static_cast<T*>(c);
    dispatch_unary(a_cast, c_cast, size, stream, functors::Sqrt<T>());
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void abs(DType_t dtype, const void* a, void* c, size_t size, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    T* c_cast = static_cast<T*>(c);
    dispatch_unary(a_cast, c_cast, size, stream, functors::Abs<T>());
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void rsqrt(DType_t dtype, const void* a, void* c, size_t size, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    T* c_cast = static_cast<T*>(c);
    dispatch_unary(a_cast, c_cast, size, stream, functors::Rsqrt<T>());
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void rcp(DType_t dtype, const void* a, void* c, size_t size, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const float* a_cast = static_cast<const float*>(a);
    float* c_cast = static_cast<float*>(c);
    dispatch_unary(a_cast, c_cast, size, stream, functors::Rcp<T>());
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void fmadd(DType_t dtype, const void* a, const void* b, void* c, size_t size, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                  std::is_same_v<T, fp16> || std::is_same_v<T, bf16> || std::is_same_v<T, int8_t>) {
      const T* a_cast = static_cast<const T*>(a);
      const T* b_cast = static_cast<const T*>(b);
      T* c_cast = static_cast<T*>(c);
      dispatch_ternary(a_cast, b_cast, c_cast, size, stream, functors::FMAdd<T>());
    } else {
      throw std::runtime_error("fmadd not supported for this type");
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void fmsub(DType_t dtype, const void* a, const void* b, void* c, size_t size, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                  std::is_same_v<T, fp16> || std::is_same_v<T, bf16> || std::is_same_v<T, int8_t>) {
      const T* a_cast = static_cast<const T*>(a);
      const T* b_cast = static_cast<const T*>(b);
      T* c_cast = static_cast<T*>(c);
      dispatch_ternary(a_cast, b_cast, c_cast, size, stream, functors::FMSub<T>());
    } else {
      throw std::runtime_error("fmsub not supported for this type");
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void fnmadd(DType_t dtype, const void* a, const void* b, void* c, size_t size,
            cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                  std::is_same_v<T, fp16> || std::is_same_v<T, bf16> || std::is_same_v<T, int8_t>) {
      const T* a_cast = static_cast<const T*>(a);
      const T* b_cast = static_cast<const T*>(b);
      T* c_cast = static_cast<T*>(c);
      dispatch_ternary(a_cast, b_cast, c_cast, size, stream, functors::FNMAdd<T>());
    } else {
      throw std::runtime_error("fnmadd not supported for this type");
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void copy(DType_t dtype, const void* a, void* c, size_t size, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    T* c_cast = static_cast<T*>(c);
    if (size == 0) return;
    cudaMemcpyAsync(c_cast, a_cast, size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    tunx::cuda::checkCudaError(cudaGetLastError(), "copy", __FILE__, __LINE__);
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void h2d_copy(DType_t dtype, const void* a, void* c, size_t size, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    T* c_cast = static_cast<T*>(c);
    if (size == 0) return;
    cudaMemcpy(c_cast, a_cast, size * sizeof(T), cudaMemcpyHostToDevice);
    tunx::cuda::checkCudaError(cudaGetLastError(), "h2d_copy", __FILE__, __LINE__);
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void d2h_copy(DType_t dtype, const void* a, void* c, size_t size, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    T* c_cast = static_cast<T*>(c);
    if (size == 0) return;
    cudaMemcpy(c_cast, a_cast, size * sizeof(T), cudaMemcpyDeviceToHost);
    tunx::cuda::checkCudaError(cudaGetLastError(), "d2h_copy", __FILE__, __LINE__);
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void fill(DType_t dtype, void* c, double scalar, size_t size, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    T* c_cast = static_cast<T*>(c);
    T scalar_cast = static_cast<T>(scalar);
    if (size == 0) return;
    if (scalar_cast == T(0)) {
      zero(dtype, c_cast, size, stream);
      return;
    }
    constexpr int vec_size = VectoredTraits<T>::size;
    bool is_aligned = ((uintptr_t)c_cast % 16 == 0);

    if (is_aligned && size % vec_size == 0) {
      int blocks = get_num_blocks(size / vec_size);
      set_scalar_vector_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(c_cast, scalar_cast, size);
    } else {
      int blocks = get_num_blocks(size);
      set_scalar_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(c_cast, scalar_cast, size);
    }
    tunx::cuda::checkCudaError(cudaGetLastError(), "fill", __FILE__, __LINE__);
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void zero(DType_t dtype, void* c, size_t size, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    T* c_cast = static_cast<T*>(c);
    if (size == 0) return;
    cudaMemsetAsync(c_cast, 0, size * sizeof(T), stream);
    tunx::cuda::checkCudaError(cudaGetLastError(), "zero", __FILE__, __LINE__);
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void fill_uniform(DType_t dtype, void* data, size_t size, double min_val, double max_val,
                  unsigned long long seed, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    T* data_cast = static_cast<T*>(data);
    if (size == 0) return;
    int blocks = get_num_blocks(size);
    fill_random_uniform_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(data_cast, size, min_val, max_val,
                                                                  seed);
    tunx::cuda::checkCudaError(cudaGetLastError(), "fill_uniform", __FILE__, __LINE__);
  };
  DISPATCH_DTYPE(dtype, T, func(T{}));
}

void fill_normal(DType_t dtype, void* data, size_t size, double mean, double stddev,
                 unsigned long long seed, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    T* data_cast = static_cast<T*>(data);
    if (size == 0) return;
    int blocks = get_num_blocks(size);
    fill_random_normal_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(data_cast, size, mean, stddev,
                                                                 seed);
    tunx::cuda::checkCudaError(cudaGetLastError(), "fill_normal", __FILE__, __LINE__);
  };
  DISPATCH_DTYPE(dtype, T, func(T{}));
}

double sum(DType_t dtype, const void* a, size_t size, cudaStream_t stream) {
  double _ret = 0;
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    return dispatch_reduce<T, 0>(a_cast, nullptr, (T)0, size, stream);
  };
  DISPATCH_ANY_DTYPE(dtype, T, _ret = static_cast<double>(func(T{})));
  return _ret;
}

double dot_product(DType_t dtype, const void* a, const void* b, size_t size, cudaStream_t stream) {
  double _ret = 0;
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    const T* b_cast = static_cast<const T*>(b);
    return dispatch_reduce<T, 1>(a_cast, b_cast, (T)0, size, stream);
  };
  DISPATCH_ANY_DTYPE(dtype, T, _ret = static_cast<double>(func(T{})));
  return _ret;
}

double norm_squared(DType_t dtype, const void* a, size_t size, cudaStream_t stream) {
  double _ret = 0;
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    return dot_product(dtype, a_cast, a_cast, size, stream);
  };
  DISPATCH_ANY_DTYPE(dtype, T, _ret = static_cast<double>(func(T{})));
  return _ret;
}

double sum_squared_diff(DType_t dtype, const void* a, double mean, size_t size,
                        cudaStream_t stream) {
  double _ret = 0;
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    T mean_cast = static_cast<T>(mean);
    return dispatch_reduce<T, 2>(a_cast, nullptr, mean_cast, size, stream);
  };
  DISPATCH_ANY_DTYPE(dtype, T, _ret = static_cast<double>(func(T{})));
  return _ret;
}

// Bswap kernel implementation
template <typename T>
__device__ __forceinline__ T device_bswap(T val) {
  if constexpr (sizeof(T) == 1) {
    return val;
  } else if constexpr (sizeof(T) == 2) {
    uint32_t u = 0;
    memcpy(&u, &val, 2);
    u = __byte_perm(u, 0, 0x0001);
    memcpy(&val, &u, 2);
    return val;
  } else if constexpr (sizeof(T) == 4) {
    uint32_t u;
    memcpy(&u, &val, 4);
    u = __byte_perm(u, 0, 0x0123);
    memcpy(&val, &u, 4);
    return val;
  } else if constexpr (sizeof(T) == 8) {
    uint32_t lo, hi;
    memcpy(&lo, (const char*)&val + 0, 4);
    memcpy(&hi, (const char*)&val + 4, 4);
    uint32_t new_lo = __byte_perm(hi, 0, 0x0123);
    uint32_t new_hi = __byte_perm(lo, 0, 0x0123);
    memcpy((char*)&val + 0, &new_lo, 4);
    memcpy((char*)&val + 4, &new_hi, 4);
    return val;
  } else {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                  "bswap: unsupported type size");
    return val;
  }
}

template <typename T>
__global__ void bswap_kernel(const T* __restrict__ a, T* __restrict__ c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = device_bswap(a[idx]);
  }
}

void bswap(DType_t dtype, const void* a, void* c, size_t size, cudaStream_t stream) {
  auto func = [&]<typename T>(T type_dummy) {
    const T* a_cast = static_cast<const T*>(a);
    T* c_cast = static_cast<T*>(c);
    if (size == 0) return;
    int blocks = get_num_blocks(size);
    bswap_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a_cast, c_cast, size);
    tunx::cuda::checkCudaError(cudaGetLastError(), "bswap", __FILE__, __LINE__);
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

// Cast kernel implementation
template <typename A_T, typename B_T>
__global__ void cast_kernel(const A_T* __restrict__ a, B_T* __restrict__ b, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    b[idx] = static_cast<B_T>(a[idx]);
  }
}

void cast(DType_t a_dtype, DType_t b_dtype, const void* a, void* b, size_t size,
          cudaStream_t stream) {
  auto func = [&]<typename A_T, typename B_T>(A_T dummy_a, B_T dummy_b) {
    const A_T* a_cast = static_cast<const A_T*>(a);
    B_T* b_cast = static_cast<B_T*>(b);
    if (size == 0) return;
    int blocks = get_num_blocks(size);
    cast_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a_cast, b_cast, size);
    tunx::cuda::checkCudaError(cudaGetLastError(), "cast", __FILE__, __LINE__);
  };
  DISPATCH_ANY_DTYPE2(a_dtype, b_dtype, A_T, B_T, func(A_T{}, B_T{}));
}

template <typename T>
__global__ void check_equals_kernel(const T* a, const T* b, size_t size, double eps,
                                    int* d_error_flag) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    if (fabs((double)a[idx] - (double)b[idx]) > eps) {
      atomicExch(d_error_flag, 1);
    }
  }
}

template <typename T>
void check_equals(const T* a, const T* b, size_t size, bool& result, double eps,
                  cudaStream_t stream) {
  if (size == 0) {
    result = true;
    return;
  }
  int* d_error_flag = nullptr;
  cudaMalloc(&d_error_flag, sizeof(int));
  cudaMemsetAsync(d_error_flag, 0, sizeof(int), stream);
  int blocks = get_num_blocks(size);
  check_equals_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, size, eps, d_error_flag);
  tunx::cuda::checkCudaError(cudaGetLastError(), "check_equals", __FILE__, __LINE__);
  int h_error_flag = 0;
  cudaMemcpyAsync(&h_error_flag, d_error_flag, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  result = (h_error_flag == 0);
  cudaFree(d_error_flag);
}

}  // namespace cuda
}  // namespace kernel
}  // namespace tunx

#endif
