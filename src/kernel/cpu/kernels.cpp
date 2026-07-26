#include "kernel/cpu/kernels.hpp"

#include <random>

#include "kernel/cpu/dkernels.hpp"
#include "kernel/cpu/skernels.hpp"

namespace tunx {
namespace kernel {
namespace cpu {

void add(DType_t dtype, const void *a_ptr, const void *b_ptr, void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    const T *b = static_cast<const T *>(b_ptr);
    T *c = static_cast<T *>(c_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::add(a, b, c, size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::add(a, b, c, size);
    } else {
      for (size_t i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void sub(DType_t dtype, const void *a_ptr, const void *b_ptr, void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    const T *b = static_cast<const T *>(b_ptr);
    T *c = static_cast<T *>(c_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::sub(a, b, c, size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::sub(a, b, c, size);
    } else {
      for (size_t i = 0; i < size; ++i) {
        c[i] = a[i] - b[i];
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void mul(DType_t dtype, const void *a_ptr, const void *b_ptr, void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    const T *b = static_cast<const T *>(b_ptr);
    T *c = static_cast<T *>(c_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::mul(a, b, c, size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::mul(a, b, c, size);
    } else {
      for (size_t i = 0; i < size; ++i) {
        c[i] = a[i] * b[i];
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void div(DType_t dtype, const void *a_ptr, const void *b_ptr, void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    const T *b = static_cast<const T *>(b_ptr);
    T *c = static_cast<T *>(c_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::div(a, b, c, size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::div(a, b, c, size);
    } else {
      for (size_t i = 0; i < size; ++i) {
        c[i] = a[i] / b[i];
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void fmadd(DType_t dtype, const void *a_ptr, const void *b_ptr, void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    const T *b = static_cast<const T *>(b_ptr);
    T *c = static_cast<T *>(c_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::fmadd(a, b, c, size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::fmadd(a, b, c, size);
    } else {
      for (size_t i = 0; i < size; ++i) {
        c[i] = fma(a[i], b[i], c[i]);
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void fmsub(DType_t dtype, const void *a_ptr, const void *b_ptr, void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    const T *b = static_cast<const T *>(b_ptr);
    T *c = static_cast<T *>(c_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::fmsub(a, b, c, size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::fmsub(a, b, c, size);
    } else {
      for (size_t i = 0; i < size; ++i) {
        c[i] = (a[i] * b[i]) - c[i];
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void fnmadd(DType_t dtype, const void *a_ptr, const void *b_ptr, void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    const T *b = static_cast<const T *>(b_ptr);
    T *c = static_cast<T *>(c_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::fnmadd(a, b, c, size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::fnmadd(a, b, c, size);
    } else {
      for (size_t i = 0; i < size; ++i) {
        c[i] = -(a[i] * b[i]) + c[i];
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

// Scalar Operations
void add_scalar(DType_t dtype, const void *a_ptr, double scalar, void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    T *c = static_cast<T *>(c_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::add_scalar(a, scalar, c, size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::add_scalar(a, scalar, c, size);
    } else {
      T s = static_cast<T>(scalar);
      for (size_t i = 0; i < size; ++i) {
        c[i] = a[i] + s;
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void sub_scalar(DType_t dtype, const void *a_ptr, double scalar, void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    T *c = static_cast<T *>(c_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::sub_scalar(a, scalar, c, size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::sub_scalar(a, scalar, c, size);
    } else {
      T s = static_cast<T>(scalar);
      for (size_t i = 0; i < size; ++i) {
        c[i] = a[i] - s;
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void mul_scalar(DType_t dtype, const void *a_ptr, double scalar, void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    T *c = static_cast<T *>(c_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::mul_scalar(a, scalar, c, size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::mul_scalar(a, scalar, c, size);
    } else {
      T s = static_cast<T>(scalar);
      for (size_t i = 0; i < size; ++i) {
        c[i] = a[i] * s;
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void div_scalar(DType_t dtype, const void *a_ptr, double scalar, void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    T *c = static_cast<T *>(c_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::div_scalar(a, scalar, c, size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::div_scalar(a, scalar, c, size);
    } else {
      T s = static_cast<T>(scalar);
      for (size_t i = 0; i < size; ++i) {
        c[i] = a[i] / s;
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void fill(DType_t dtype, void *c_ptr, double scalar, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    T *c = static_cast<T *>(c_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::fill(c, scalar, size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::fill(c, scalar, size);
    } else {
      T s = static_cast<T>(scalar);
      for (size_t i = 0; i < size; ++i) {
        c[i] = s;
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

// BLAS-like Operations
void axpy(DType_t dtype, double alpha, const void *x_ptr, void *y_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *x = static_cast<const T *>(x_ptr);
    T *y = static_cast<T *>(y_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::axpy(alpha, x, y, size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::axpy(alpha, x, y, size);
    } else {
      T a = static_cast<T>(alpha);
      for (size_t i = 0; i < size; ++i) {
        y[i] += a * x[i];
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

// Element-wise Functions
void sqrt(DType_t dtype, const void *a_ptr, void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    T *c = static_cast<T *>(c_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::sqrt(a, c, size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::sqrt(a, c, size);
    } else {
      for (size_t i = 0; i < size; ++i) {
        c[i] = static_cast<T>(std::sqrt(static_cast<float>(a[i])));
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void abs(DType_t dtype, const void *a_ptr, void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    T *c = static_cast<T *>(c_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::abs(a, c, size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::abs(a, c, size);
    } else {
      for (size_t i = 0; i < size; ++i) {
        c[i] = static_cast<T>(std::abs(static_cast<float>(a[i])));
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void rsqrt(DType_t dtype, const void *a_ptr, void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    T *c = static_cast<T *>(c_ptr);

    // c[i] = 1.0 / sqrt(a[i])
    if constexpr (std::is_same_v<T, float>) {
      fp::rsqrt(a, c, size);
    } else {
      for (size_t i = 0; i < size; ++i) {
        c[i] = static_cast<T>(1.0 / std::sqrt(static_cast<float>(a[i])));
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void rcp(DType_t dtype, const void *a_ptr, void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    T *c = static_cast<T *>(c_ptr);

    // c[i] = 1.0 / a[i]
    if constexpr (std::is_same_v<T, float>) {
      fp::rcp(a, c, size);
    } else {
      T one = static_cast<T>(1.0);
      for (size_t i = 0; i < size; ++i) {
        c[i] = one / a[i];
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

// Comparison and Clamping
void min(DType_t dtype, const void *a_ptr, const void *b_ptr, void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    const T *b = static_cast<const T *>(b_ptr);
    T *c = static_cast<T *>(c_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::min(a, b, c, size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::min(a, b, c, size);
    } else {
      for (size_t i = 0; i < size; ++i) {
        c[i] = a[i] < b[i] ? a[i] : b[i];  // equivalent to std::min
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void max(DType_t dtype, const void *a_ptr, const void *b_ptr, void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    const T *b = static_cast<const T *>(b_ptr);
    T *c = static_cast<T *>(c_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::max(a, b, c, size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::max(a, b, c, size);
    } else {
      for (size_t i = 0; i < size; ++i) {
        c[i] = a[i] > b[i] ? a[i] : b[i];  // equivalent to std::max
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void scalar_max(DType_t dtype, const void *a_ptr, double scalar, void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    T *c = static_cast<T *>(c_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::scalar_max(a, scalar, c, size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::scalar_max(a, scalar, c, size);
    } else {
      T s = static_cast<T>(scalar);
      for (size_t i = 0; i < size; ++i) {
        c[i] = a[i] > s ? a[i] : s;
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void clamp(DType_t dtype, const void *a_ptr, double min_val, double max_val, void *c_ptr,
           size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    T *c = static_cast<T *>(c_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::clamp(a, min_val, max_val, c, size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::clamp(a, min_val, max_val, c, size);
    } else {
      T min_t = static_cast<T>(min_val);
      T max_t = static_cast<T>(max_val);
      for (size_t i = 0; i < size; ++i) {
        c[i] = a[i] < min_t ? min_t : (a[i] > max_t ? max_t : a[i]);
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void equal(DType_t dtype, const void *a_ptr, const void *b_ptr, void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    const T *b = static_cast<const T *>(b_ptr);
    T *c = static_cast<T *>(c_ptr);

    // c[i] = (a[i] == b[i]) ? 1 : 0
    if constexpr (std::is_same_v<T, float>) {
      fp::equal(a, b, c, size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::equal(a, b, c, size);
    } else {
      for (size_t i = 0; i < size; ++i) {
        c[i] = (a[i] == b[i]) ? static_cast<T>(1.0) : static_cast<T>(0.0);
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void greater(DType_t dtype, const void *a_ptr, const void *b_ptr, void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    const T *b = static_cast<const T *>(b_ptr);
    T *c = static_cast<T *>(c_ptr);

    // c[i] = (a[i] > b[i]) ? 1 : 0
    if constexpr (std::is_same_v<T, float>) {
      fp::greater(a, b, c, size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::greater(a, b, c, size);
    } else {
      for (size_t i = 0; i < size; ++i) {
        c[i] = (a[i] > b[i]) ? static_cast<T>(1.0) : static_cast<T>(0.0);
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

// Memory Operations
void copy(DType_t dtype, const void *a_ptr, void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    T *c = static_cast<T *>(c_ptr);

    std::memcpy(c, a, size * sizeof(T));
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void zero(DType_t dtype, void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    T *c = static_cast<T *>(c_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::zero(c, size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::zero(c, size);
    } else {
      for (size_t i = 0; i < size; ++i) {
        c[i] = static_cast<T>(0.0);
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

// Specialized BatchNorm Operations
void sub_mul_scalar(DType_t dtype, const void *a_ptr, double sub_scalar, double mul_scalar,
                    void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    T *c = static_cast<T *>(c_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::sub_mul_scalar(a, static_cast<float>(sub_scalar), static_cast<float>(mul_scalar), c,
                         size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::sub_mul_scalar(a, sub_scalar, mul_scalar, c, size);
    } else {
      T sub_val = static_cast<T>(sub_scalar);
      T mul_val = static_cast<T>(mul_scalar);
      for (size_t i = 0; i < size; ++i) {
        c[i] = (a[i] - sub_val) * mul_val;
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void mul_add_scalar(DType_t dtype, const void *a_ptr, double mul_scalar, double add_scalar,
                    void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    T *c = static_cast<T *>(c_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::mul_add_scalar(a, static_cast<float>(mul_scalar), static_cast<float>(add_scalar), c,
                         size);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::mul_add_scalar(a, mul_scalar, add_scalar, c, size);
    } else {
      T mul_val = static_cast<T>(mul_scalar);
      T add_val = static_cast<T>(add_scalar);
      for (size_t i = 0; i < size; ++i) {
        c[i] = (a[i] * mul_val) + add_val;
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

// Reduction Functions
double sum(DType_t dtype, const void *a_ptr, size_t size) {
  double _ret = 0;
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);

    if constexpr (std::is_same_v<T, float>) {
      return static_cast<double>(fp::sum(a, size));
    } else if constexpr (std::is_same_v<T, double>) {
      return static_cast<double>(dp::sum(a, size));
    } else {
      double result = 0.0;
      for (size_t i = 0; i < size; ++i) {
        result += static_cast<double>(a[i]);
      }
      return result;
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, _ret = static_cast<double>(func(T{})));
  return static_cast<double>(_ret);
}

double dot_product(DType_t dtype, const void *a_ptr, const void *b_ptr, size_t size) {
  double _ret = 0;
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    const T *b = static_cast<const T *>(b_ptr);

    if constexpr (std::is_same_v<T, float>) {
      return static_cast<double>(fp::dot_product(a, b, size));
    } else if constexpr (std::is_same_v<T, double>) {
      return static_cast<double>(dp::dot_product(a, b, size));
    } else {
      double result = 0.0;
      for (size_t i = 0; i < size; ++i) {
        result += static_cast<double>(a[i]) * static_cast<double>(b[i]);
      }
      return result;
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, _ret = static_cast<double>(func(T{})));
  return static_cast<double>(_ret);
}

double sum_squared_diff(DType_t dtype, const void *a_ptr, double mean, size_t size) {
  double _ret = 0;
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);

    // sum((a[i] - mean)^2)
    if constexpr (std::is_same_v<T, float>) {
      return static_cast<double>(fp::sum_squared_diff(a, mean, size));
    } else if constexpr (std::is_same_v<T, double>) {
      return static_cast<double>(dp::sum_squared_diff(a, mean, size));
    } else {
      double result = 0.0;
      for (size_t i = 0; i < size; ++i) {
        double diff = static_cast<double>(a[i]) - mean;
        result += diff * diff;
      }
      return result;
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, _ret = static_cast<double>(func(T{})));
  return static_cast<double>(_ret);
}

double norm_squared(DType_t dtype, const void *a_ptr, size_t size) {
  double _ret = 0;
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);

    // sum(a[i]^2)
    double result = 0.0;
    for (size_t i = 0; i < size; ++i) {
      double val = static_cast<double>(a[i]);
      result += val * val;
    }
    return result;
  };
  DISPATCH_ANY_DTYPE(dtype, T, _ret = static_cast<double>(func(T{})));
  return static_cast<double>(_ret);
}

void fill_uniform(DType_t dtype, void *data_ptr, size_t size, double min_val, double max_val,
                  unsigned long long seed) {
  auto func = [&]<typename T>(T type_dummy) {
    T *data = static_cast<T *>(data_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::fill_uniform(data, size, min_val, max_val, seed);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::fill_uniform(data, size, min_val, max_val, seed);
    } else {
      std::mt19937_64 rng(seed);
      if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(min_val, max_val);
        for (size_t i = 0; i < size; ++i) {
          data[i] = dist(rng);
        }
      } else {
        std::uniform_real_distribution<float> dist(static_cast<float>(min_val),
                                                   static_cast<float>(max_val));
        for (size_t i = 0; i < size; ++i) {
          data[i] = static_cast<T>(dist(rng));
        }
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void fill_normal(DType_t dtype, void *data_ptr, size_t size, double mean, double stddev,
                 unsigned long long seed) {
  auto func = [&]<typename T>(T type_dummy) {
    T *data = static_cast<T *>(data_ptr);

    if constexpr (std::is_same_v<T, float>) {
      fp::fill_normal(data, size, mean, stddev, seed);
    } else if constexpr (std::is_same_v<T, double>) {
      dp::fill_normal(data, size, mean, stddev, seed);
    } else {
      std::mt19937_64 rng(seed);
      if constexpr (std::is_floating_point_v<T>) {
        std::normal_distribution<T> dist(mean, stddev);
        for (size_t i = 0; i < size; ++i) {
          data[i] = dist(rng);
        }
      } else {
        // For types like fp16 that are not standard floating point
        std::normal_distribution<float> dist(static_cast<float>(mean), static_cast<float>(stddev));
        for (size_t i = 0; i < size; ++i) {
          data[i] = static_cast<T>(dist(rng));
        }
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void cast(DType_t a_dtype, DType_t b_dtype, const void *a_ptr, void *b_ptr, size_t size) {
  auto func = [&]<typename A_T, typename B_T>(A_T dummy_a, B_T dummy_b) {
    const A_T *a = static_cast<const A_T *>(a_ptr);
    B_T *b = static_cast<B_T *>(b_ptr);

    for (size_t i = 0; i < size; ++i) {
      b[i] = static_cast<B_T>(a[i]);
    }
  };
  DISPATCH_ANY_DTYPE2(a_dtype, b_dtype, A_T, B_T, func(A_T{}, B_T{}));
}

void bswap(DType_t dtype, const void *a_ptr, void *c_ptr, size_t size) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    T *c = static_cast<T *>(c_ptr);

    for (size_t i = 0; i < size; ++i) {
      c[i] = a[i];
      if constexpr (sizeof(T) > 1) {
        auto *bytes = reinterpret_cast<uint8_t *>(&c[i]);
        for (size_t j = 0; j < sizeof(T) / 2; ++j) {
          uint8_t tmp = bytes[j];
          bytes[j] = bytes[sizeof(T) - 1 - j];
          bytes[sizeof(T) - 1 - j] = tmp;
        }
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

void check_equals(DType_t dtype, const void *a_ptr, const void *b_ptr, size_t size, bool &result,
                  double eps) {
  auto func = [&]<typename T>(T type_dummy) {
    const T *a = static_cast<const T *>(a_ptr);
    const T *b = static_cast<const T *>(b_ptr);

    result = true;
    for (size_t i = 0; i < size; ++i) {
      double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
      if (diff < 0) diff = -diff;
      if (diff > eps) {
        result = false;
        break;
      }
    }
  };
  DISPATCH_ANY_DTYPE(dtype, T, func(T{}));
}

}  // namespace cpu
}  // namespace kernel
}  // namespace tunx