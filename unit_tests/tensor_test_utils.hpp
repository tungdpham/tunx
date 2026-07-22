#pragma once

#include <fmt/core.h>
#include <gtest/gtest.h>

#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace tunx {
template <typename OutputType, typename ExpectedType>
void compare_array_t(const OutputType* output, const ExpectedType* expected, size_t size,
                     double eps = 1e-3) {
  size_t mismatch_count = 0;
  for (size_t i = 0; i < size; i++) {
    float out = static_cast<float>(output[i]);
    float exp = static_cast<float>(expected[i]);
    double diff = std::abs(out - exp);
    if (diff > eps && mismatch_count < 100) {
      mismatch_count++;
      fmt::print("Mismatch at index: {}, output: {}, expected: {}, diff: {}\n", i, out, exp, diff);
    }
  }
  EXPECT_EQ(mismatch_count, 0) << fmt::format("Mismatch count: {}", mismatch_count);
}

inline void compare_tensor(const Tensor& output, const Tensor& expected, double eps = 1e-3) {
  Tensor host_output = output.to_host();
  Tensor host_expected = expected.to_host();
  EXPECT_EQ(host_output.size(), expected.size());

  DType_t output_dtype = host_output.dtype();
  DType_t expected_dtype = host_expected.dtype();

  DISPATCH_ANY_DTYPE2(
      output_dtype, expected_dtype, OutputType, ExpectedType,
      compare_array_t(host_output.data_as<OutputType>(), host_expected.data_as<ExpectedType>(),
                      host_output.size(), eps));
}

}  // namespace tunx