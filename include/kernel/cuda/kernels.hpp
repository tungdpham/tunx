#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "type/type.hpp"

namespace tunx {
namespace kernel {
namespace cuda {

void add(DType_t dtype, const void *a, const void *b, void *c, size_t size, cudaStream_t stream);

void sub(DType_t dtype, const void *a, const void *b, void *c, size_t size, cudaStream_t stream);

void mul(DType_t dtype, const void *a, const void *b, void *c, size_t size, cudaStream_t stream);

void div(DType_t dtype, const void *a, const void *b, void *c, size_t size, cudaStream_t stream);

void fmadd(DType_t dtype, const void *a, const void *b, void *c, size_t size, cudaStream_t stream);

void fmsub(DType_t dtype, const void *a, const void *b, void *c, size_t size, cudaStream_t stream);

void fnmadd(DType_t dtype, const void *a, const void *b, void *c, size_t size, cudaStream_t stream);

void add_scalar(DType_t dtype, const void *a, double scalar, void *c, size_t size,
                cudaStream_t stream);

void sub_scalar(DType_t dtype, const void *a, double scalar, void *c, size_t size,
                cudaStream_t stream);

void mul_scalar(DType_t dtype, const void *a, double scalar, void *c, size_t size,
                cudaStream_t stream);

void div_scalar(DType_t dtype, const void *a, double scalar, void *c, size_t size,
                cudaStream_t stream);

void fill(DType_t dtype, void *c, double scalar, size_t size, cudaStream_t stream);

void axpy(DType_t dtype, double alpha, const void *x, void *y, size_t size, cudaStream_t stream);

void sqrt(DType_t dtype, const void *a, void *c, size_t size, cudaStream_t stream);

void rsqrt(DType_t dtype, const void *a, void *c, size_t size, cudaStream_t stream);

void rcp(DType_t dtype, const void *a, void *c, size_t size, cudaStream_t stream);

void abs(DType_t dtype, const void *a, void *c, size_t size, cudaStream_t stream);

void min(DType_t dtype, const void *a, const void *b, void *c, size_t size, cudaStream_t stream);

void max(DType_t dtype, const void *a, const void *b, void *c, size_t size, cudaStream_t stream);

void scalar_max(DType_t dtype, const void *a, double scalar, void *c, size_t size,
                cudaStream_t stream);

void clamp(DType_t dtype, const void *a, double min_val, double max_val, void *c, size_t size,
           cudaStream_t stream);

void equal(DType_t dtype, const void *a, const void *b, void *c, size_t size, cudaStream_t stream);

void greater(DType_t dtype, const void *a, const void *b, void *c, size_t size,
             cudaStream_t stream);

void copy(DType_t dtype, const void *a, void *c, size_t size, cudaStream_t stream);

void h2d_copy(DType_t dtype, const void *a, void *c, size_t size, cudaStream_t stream);

void d2h_copy(DType_t dtype, const void *a, void *c, size_t size, cudaStream_t stream);

void zero(DType_t dtype, void *c, size_t size, cudaStream_t stream);

double sum(DType_t dtype, const void *a, size_t size, cudaStream_t stream);

double dot_product(DType_t dtype, const void *a, const void *b, size_t size, cudaStream_t stream);

double norm_squared(DType_t dtype, const void *a, size_t size, cudaStream_t stream);

double sum_squared_diff(DType_t dtype, const void *a, double mean, size_t size,
                        cudaStream_t stream);

void sub_mul_scalar(DType_t dtype, const void *a, double sub_scalar, double mul_scalar, void *c,
                    size_t size, cudaStream_t stream);

void mul_add_scalar(DType_t dtype, const void *a, double mul_scalar, double add_scalar, void *c,
                    size_t size, cudaStream_t stream);

void fill_uniform(DType_t dtype, void *data, size_t size, double min_val, double max_val,
                  unsigned long long seed, cudaStream_t stream);

void fill_normal(DType_t dtype, void *data, size_t size, double mean, double stddev,
                 unsigned long long seed, cudaStream_t stream);

void cast(DType_t a_dtype, DType_t b_dtype, const void *a, void *b, size_t size,
          cudaStream_t stream);

void bswap(DType_t dtype, const void *a, void *c, size_t size, cudaStream_t stream);

void check_equals(DType_t dtype, const void *a, const void *b, size_t size, bool &result,
                  double eps, cudaStream_t stream);

}  // namespace cuda
}  // namespace kernel
}  // namespace tunx