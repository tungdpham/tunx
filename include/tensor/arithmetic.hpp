#pragma once

#include "tensor.hpp"
#include "tensor/ops.hpp"

namespace tunx {
inline Tensor &operator+=(Tensor &lhs, const Tensor &rhs) {
  if (lhs.shape() != rhs.shape())
    throw std::invalid_argument("Tensor shapes must match for addition");
  if (lhs.dtype() != rhs.dtype()) throw std::runtime_error("DType mismatch in Tensor addition");

  add(lhs, rhs, lhs);
  return lhs;
}

inline Tensor &operator-=(Tensor &lhs, const Tensor &rhs) {
  if (lhs.shape() != rhs.shape())
    throw std::invalid_argument("Tensor shapes must match for subtraction");
  if (lhs.dtype() != rhs.dtype()) throw std::runtime_error("DType mismatch in Tensor subtraction");

  sub(lhs, rhs, lhs);
  return lhs;
}

inline Tensor &operator*=(Tensor &lhs, const Tensor &rhs) {
  if (lhs.shape() != rhs.shape())
    throw std::invalid_argument("Tensor shapes must match for multiplication");
  if (lhs.dtype() != rhs.dtype())
    throw std::runtime_error("DType mismatch in Tensor multiplication");

  mul(lhs, rhs, lhs);
  return lhs;
}

inline Tensor &operator/=(Tensor &lhs, const Tensor &rhs) {
  if (lhs.shape() != rhs.shape())
    throw std::invalid_argument("Tensor shapes must match for division");
  if (lhs.dtype() != rhs.dtype()) throw std::runtime_error("DType mismatch in Tensor division");

  div(lhs, rhs, lhs);
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
  add_scalar(lhs, scalar, lhs);
  return lhs;
}

inline Tensor &operator-=(Tensor &lhs, double scalar) {
  sub_scalar(lhs, scalar, lhs);
  return lhs;
}

inline Tensor &operator*=(Tensor &lhs, double scalar) {
  mul_scalar(lhs, scalar, lhs);
  return lhs;
}

inline Tensor &operator/=(Tensor &lhs, double scalar) {
  if (scalar == 0.0) throw std::invalid_argument("Division by zero");
  div_scalar(lhs, scalar, lhs);
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