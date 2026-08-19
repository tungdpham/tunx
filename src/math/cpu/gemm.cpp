#include "math/cpu/gemm.hpp"
#include "type/type.hpp"

namespace tunx {
namespace cpu {

void gemm(const void *A, const void *B, void *C, size_t M, size_t N, size_t K, const bool trans_A,
          const bool trans_B, float alpha, float beta, DTypeDesc type_desc) {
  if (type_desc.io_dtype != type_desc.param_dtype || type_desc.io_dtype != type_desc.compute_dtype) {
    throw std::runtime_error("CPU GEMM currently only supports homogeneous data types.");
  }

  DISPATCH_ANY_DTYPE(type_desc.io_dtype, T, {
    legacy_gemm(static_cast<const T*>(A), static_cast<const T*>(B), static_cast<T*>(C), M, N, K,
                trans_A, trans_B, static_cast<T>(alpha), static_cast<T>(beta));
  });
}

}  // namespace cpu
}  // namespace tunx
