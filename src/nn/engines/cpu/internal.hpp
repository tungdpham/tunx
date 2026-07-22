#pragma once

#include <cmath>

namespace tunx {

#define CHECK_HOMOGENEOUS_DTYPE(type_desc)                                           \
  if ((type_desc).io_dtype != (type_desc).param_dtype ||                             \
      (type_desc).param_dtype != (type_desc).compute_dtype) {                        \
    throw std::invalid_argument(                                                     \
        "Homogenous functions require identical dtypes for io, param, and compute"); \
  }

}  // namespace tunx
