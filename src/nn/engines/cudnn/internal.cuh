#ifndef TUNX_ENGINES_CUDNN_INTERNAL_H
#define TUNX_ENGINES_CUDNN_INTERNAL_H
#ifdef TUNX_USE_CUDNN

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn_frontend.h>
#include <cudnn_graph.h>
#include <fmt/core.h>

#include <stdexcept>

#include "cudnn_frontend_utils.h"
#include "type/type.hpp"

namespace tunx {

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

namespace fe = cudnn_frontend;

[[maybe_unused]] static void ensure_ok(fe::error_t status, std::string stage) {
  if (status.is_bad()) {
    throw std::runtime_error("cuDNN frontend error at " + stage + ": " + status.get_message());
  }
}

[[maybe_unused]] static std::string to_string(fe::DataType_t data_type) {
  switch (data_type) {
    case fe::DataType_t::HALF:
      return "HALF";
    case fe::DataType_t::FLOAT:
      return "FLOAT";
    case fe::DataType_t::DOUBLE:
      return "DOUBLE";
    case fe::DataType_t::BFLOAT16:
      return "BFLOAT16";
    case fe::DataType_t::INT8:
      return "INT8";
    case fe::DataType_t::INT32:
      return "INT32";
    case fe::DataType_t::INT64:
      return "INT64";
    case fe::DataType_t::UINT8:
      return "UINT8";
    default:
      return "UNKNOWN";
  }
}

[[maybe_unused]] static fe::DataType_t to_fe_data_type(DType_t data_type) {
  switch (data_type) {
    case DType_t::FP16:
      return fe::DataType_t::HALF;
    case DType_t::FP32:
      return fe::DataType_t::FLOAT;
    case DType_t::FP64:
      return fe::DataType_t::DOUBLE;
    case DType_t::BF16:
      return fe::DataType_t::BFLOAT16;
    case DType_t::INT8:
      return fe::DataType_t::INT8;
    default:
      throw std::runtime_error("Unsupported cuDNN data type for GEMM");
  }
}

[[maybe_unused]] static fe::DataType_t to_fe_compute_type(DType_t data_type) {
  if (data_type == DType_t::FP16 || data_type == DType_t::BF16) {
    return fe::DataType_t::FLOAT;
  }
  return to_fe_data_type(data_type);
}

}  // namespace tunx

#endif
#endif