#pragma once

#include "kernel/kernel.hpp"
#include "tensor.hpp"

namespace tunx {
inline Tensor &operator+=(Tensor &lhs, const Tensor &rhs) {
  if (lhs.shape() != rhs.shape())
    throw std::invalid_argument("Tensor shapes must match for addition");
  if (lhs.dtype() != rhs.dtype()) throw std::runtime_error("DType mismatch in Tensor addition");

  DISPATCH_ANY_DTYPE(
      lhs.dtype(), T,
      kernel::add(dtype_of<T>(), lhs.data_ptr(), rhs.data_ptr(), lhs.data_ptr(), lhs.size()));
  return lhs;
}

inline Tensor &operator-=(Tensor &lhs, const Tensor &rhs) {
  if (lhs.shape() != rhs.shape())
    throw std::invalid_argument("Tensor shapes must match for subtraction");
  if (lhs.dtype() != rhs.dtype()) throw std::runtime_error("DType mismatch in Tensor subtraction");

  DISPATCH_ANY_DTYPE(
      lhs.dtype(), T,
      kernel::sub(dtype_of<T>(), lhs.data_ptr(), rhs.data_ptr(), lhs.data_ptr(), lhs.size()));
  return lhs;
}

inline Tensor &operator*=(Tensor &lhs, const Tensor &rhs) {
  if (lhs.shape() != rhs.shape())
    throw std::invalid_argument("Tensor shapes must match for multiplication");
  if (lhs.dtype() != rhs.dtype())
    throw std::runtime_error("DType mismatch in Tensor multiplication");

  DISPATCH_ANY_DTYPE(
      lhs.dtype(), T,
      kernel::mul(dtype_of<T>(), lhs.data_ptr(), rhs.data_ptr(), lhs.data_ptr(), lhs.size()));
  return lhs;
}

inline Tensor &operator/=(Tensor &lhs, const Tensor &rhs) {
  if (lhs.shape() != rhs.shape())
    throw std::invalid_argument("Tensor shapes must match for division");
  if (lhs.dtype() != rhs.dtype()) throw std::runtime_error("DType mismatch in Tensor division");

  DISPATCH_ANY_DTYPE(
      lhs.dtype(), T,
      kernel::div(dtype_of<T>(), lhs.data_ptr(), rhs.data_ptr(), lhs.data_ptr(), lhs.size()));
  return lhs;
}

inline Tensor operator+(const Tensor &lhs, const Tensor &rhs) {
  Tensor result = lhs;
  result += rhs;
  return result;
}

inline Tensor operator-(const Tensor &lhs, const Tensor &rhs) {
  Tensor result = lhs;
  result -= rhs;
  return result;
}

inline Tensor operator*(const Tensor &lhs, const Tensor &rhs) {
  Tensor result = lhs;
  result *= rhs;
  return result;
}

inline Tensor operator/(const Tensor &lhs, const Tensor &rhs) {
  Tensor result = lhs;
  result /= rhs;
  return result;
}

inline Tensor &operator+=(Tensor &lhs, double scalar) {
  DISPATCH_ANY_DTYPE(lhs.dtype(), T,
                     kernel::add_scalar(dtype_of<T>(), lhs.data_ptr(), static_cast<T>(scalar),
                                        lhs.data_ptr(), lhs.size()));
  return lhs;
}

inline Tensor &operator-=(Tensor &lhs, double scalar) {
  DISPATCH_ANY_DTYPE(lhs.dtype(), T,
                     kernel::sub_scalar(dtype_of<T>(), lhs.data_ptr(), static_cast<T>(scalar),
                                        lhs.data_ptr(), lhs.size()));
  return lhs;
}

inline Tensor &operator*=(Tensor &lhs, double scalar) {
  DISPATCH_ANY_DTYPE(lhs.dtype(), T,
                     kernel::mul_scalar(dtype_of<T>(), lhs.data_ptr(), static_cast<T>(scalar),
                                        lhs.data_ptr(), lhs.size()));
  return lhs;
}

inline Tensor &operator/=(Tensor &lhs, double scalar) {
  if (scalar == 0.0) throw std::invalid_argument("Division by zero");
  DISPATCH_ANY_DTYPE(lhs.dtype(), T,
                     kernel::div_scalar(dtype_of<T>(), lhs.data_ptr(), static_cast<T>(scalar),
                                        lhs.data_ptr(), lhs.size()));
  return lhs;
}

inline Tensor operator+(const Tensor &lhs, double scalar) {
  Tensor result = lhs;
  result += scalar;
  return result;
}

inline Tensor operator-(const Tensor &lhs, double scalar) {
  Tensor result = lhs;
  result -= scalar;
  return result;
}

inline Tensor operator*(const Tensor &lhs, double scalar) {
  Tensor result = lhs;
  result *= scalar;
  return result;
}

inline Tensor operator/(const Tensor &lhs, double scalar) {
  Tensor result = lhs;
  result /= scalar;
  return result;
}

inline Tensor operator+(double scalar, const Tensor &rhs) {
  Tensor result = rhs;
  result += scalar;
  return result;
}

inline Tensor operator*(double scalar, const Tensor &rhs) {
  Tensor result = rhs;
  result *= scalar;
  return result;
}
}  // namespace tunx