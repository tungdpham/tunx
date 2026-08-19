#pragma once

#include <cmath>
#include <cstddef>
#include <cstring>

#include "type/type.hpp"

namespace tunx {
namespace kernel {
namespace cpu {

void add(DType_t dtype, const void *a_ptr, const void *b_ptr, void *c_ptr, size_t size);

void sub(DType_t dtype, const void *a_ptr, const void *b_ptr, void *c_ptr, size_t size);

void mul(DType_t dtype, const void *a_ptr, const void *b_ptr, void *c_ptr, size_t size);

void div(DType_t dtype, const void *a_ptr, const void *b_ptr, void *c_ptr, size_t size);

void fmadd(DType_t dtype, const void *a_ptr, const void *b_ptr, void *c_ptr, size_t size);

void fmsub(DType_t dtype, const void *a_ptr, const void *b_ptr, void *c_ptr, size_t size);

void fnmadd(DType_t dtype, const void *a_ptr, const void *b_ptr, void *c_ptr, size_t size);

void add_scalar(DType_t dtype, const void *a_ptr, double scalar, void *c_ptr, size_t size);

void sub_scalar(DType_t dtype, const void *a_ptr, double scalar, void *c_ptr, size_t size);

void mul_scalar(DType_t dtype, const void *a_ptr, double scalar, void *c_ptr, size_t size);

void div_scalar(DType_t dtype, const void *a_ptr, double scalar, void *c_ptr, size_t size);

void fill(DType_t dtype, void *c_ptr, double scalar, size_t size);

void axpy(DType_t dtype, double alpha, const void *x_ptr, void *y_ptr, size_t size);

void sqrt(DType_t dtype, const void *a_ptr, void *c_ptr, size_t size);

void abs(DType_t dtype, const void *a_ptr, void *c_ptr, size_t size);

void rsqrt(DType_t dtype, const void *a_ptr, void *c_ptr, size_t size);

void rcp(DType_t dtype, const void *a_ptr, void *c_ptr, size_t size);

void min(DType_t dtype, const void *a_ptr, const void *b_ptr, void *c_ptr, size_t size);

void max(DType_t dtype, const void *a_ptr, const void *b_ptr, void *c_ptr, size_t size);

void scalar_max(DType_t dtype, const void *a_ptr, double scalar, void *c_ptr, size_t size);

void clamp(DType_t dtype, const void *a_ptr, double min_val, double max_val, void *c_ptr,
           size_t size);

void equal(DType_t dtype, const void *a_ptr, const void *b_ptr, void *c_ptr, size_t size);

void greater(DType_t dtype, const void *a_ptr, const void *b_ptr, void *c_ptr, size_t size);

void copy(DType_t dtype, const void *a_ptr, void *c_ptr, size_t size);

void zero(DType_t dtype, void *c_ptr, size_t size);

void sub_mul_scalar(DType_t dtype, const void *a_ptr, double sub_scalar, double mul_scalar,
                    void *c_ptr, size_t size);

void mul_add_scalar(DType_t dtype, const void *a_ptr, double mul_scalar, double add_scalar,
                    void *c_ptr, size_t size);

double sum(DType_t dtype, const void *a_ptr, size_t size);

double dot_product(DType_t dtype, const void *a_ptr, const void *b_ptr, size_t size);

double sum_squared_diff(DType_t dtype, const void *a_ptr, double mean, size_t size);

double norm_squared(DType_t dtype, const void *a_ptr, size_t size);

void fill_uniform(DType_t dtype, void *data_ptr, size_t size, double min_val, double max_val,
                  unsigned long long seed);

void fill_normal(DType_t dtype, void *data_ptr, size_t size, double mean, double stddev,
                 unsigned long long seed);

void cast(DType_t a_dtype, DType_t b_dtype, const void *a_ptr, void *b_ptr, size_t size);

void bswap(DType_t dtype, const void *a_ptr, void *c_ptr, size_t size);

void check_equals(DType_t dtype, const void *a_ptr, const void *b_ptr, size_t size, bool &result,
                  double eps);

}  // namespace cpu
}  // namespace kernel
}  // namespace tunx