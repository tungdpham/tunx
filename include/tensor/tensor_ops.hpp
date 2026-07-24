#pragma once

#include <fmt/core.h>

#include <istream>
#include <ostream>

#include "cpu/tensor_ops.hpp"
#include "device/stream.hpp"
#include "device/task.hpp"
#include "kernel/kernel.hpp"
#include "type/type.hpp"
#ifdef TUNX_USE_CUDA
#include "cuda/tensor_ops.hpp"
#endif
#include "tensor.hpp"

namespace tunx {

inline void save(const Tensor &tensor, std::ostream &out) {
  if (!out) {
    throw std::runtime_error("Stream is not ready for writing");
  }

  // write dims, shape
  size_t dims = tensor.dims();
  DType_t dtype = tensor.dtype();
  out.write(reinterpret_cast<const char *>(&dtype), sizeof(DType_t));
  out.write(reinterpret_cast<const char *>(&dims), sizeof(size_t));
  out.write(reinterpret_cast<const char *>(tensor.shape().data()),
            tensor.shape().size() * sizeof(size_t));

  if (tensor.device_type() == DeviceType::CPU) {
    out.write(reinterpret_cast<const char *>(tensor.data_as<uchar>()),
              tensor.size() * get_dtype_size(dtype));
  } else {
    auto host_tensor = tensor.to_host();
    out.write(reinterpret_cast<const char *>(host_tensor.data_as<uchar>()),
              tensor.size() * get_dtype_size(dtype));
  }
}

inline void load(Tensor &tensor, std::istream &in) {
  if (!in) {
    throw std::runtime_error("Stream is not ready for reading");
  }

  // read dims, shape
  DType_t dtype;
  size_t dims;
  in.read(reinterpret_cast<char *>(&dtype), sizeof(DType_t));
  in.read(reinterpret_cast<char *>(&dims), sizeof(size_t));
  Vec<size_t> shape(dims);
  in.read(reinterpret_cast<char *>(shape.data()), dims * sizeof(size_t));

  tensor = Tensor(shape, dtype, tensor.allocator());

  if (tensor.device_type() == DeviceType::CPU) {
    in.read(reinterpret_cast<char *>(tensor.data_as<uchar>()),
            tensor.size() * get_dtype_size(dtype));
  } else {
    Tensor host_tensor(shape, dtype, DeviceAllocator::instance(getHost()));
    in.read(reinterpret_cast<char *>(host_tensor.data_as<uchar>()),
            tensor.size() * get_dtype_size(dtype));
    host_tensor.copy_to(tensor);
  }
}

inline void fill(Tensor &tensor, double value, stream stream = nullptr) {
  kernel::set_scalar(tensor.dtype(), tensor.data_ptr(), value, tensor.size(), stream);
}

inline void fill_normal(Tensor &tensor, double mean, double stddev, unsigned long long seed,
                        stream stream = nullptr) {
  kernel::fill_random_normal(tensor.dtype(), tensor.data_ptr(), tensor.size(), mean, stddev, seed,
                             stream);
}

inline void fill_normal(Tensor &tensor, double mean, double stddev, stream stream = nullptr) {
  unsigned long long seed = static_cast<unsigned long long>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
      reinterpret_cast<uintptr_t>(tensor.data_as<void>()));

  kernel::fill_random_normal(tensor.dtype(), tensor.data_ptr(), tensor.size(), mean, stddev, seed,
                             stream);
}

inline void fill_uniform(Tensor &tensor, double low, double high, unsigned long long seed,
                         stream stream = nullptr) {
  kernel::fill_random_uniform(tensor.dtype(), tensor.data_ptr(), tensor.size(), low, high, seed,
                              stream);
}

inline void fill_uniform(Tensor &tensor, double low, double high, stream stream = nullptr) {
  unsigned long long seed = static_cast<unsigned long long>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
      reinterpret_cast<uintptr_t>(tensor.data_as<void>()));

  kernel::fill_random_uniform(tensor.dtype(), tensor.data_ptr(), tensor.size(), low, high, seed,
                              stream);
}

inline void im2col(const Tensor &input, Tensor &col_data, size_t kernel_h, size_t kernel_w,
                   size_t stride_h = 1, size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0,
                   stream stream = nullptr) {
  if (col_data.device() != input.device()) {
    throw std::runtime_error("im2col: Mismatched device types between col_data and input");
  }

  auto &device = input.device();

  if (input.dtype() != col_data.dtype()) {
    throw std::runtime_error("im2col: Mismatched data types between col_data and input");
  }

  const auto &shape = input.shape();
  if (shape.size() != 4) {
    throw std::invalid_argument("im2col: Input tensor must be 4-dimensional (NCHW)");
  }

  DType_t dtype = input.dtype();

  size_t batch_size = shape[0];
  size_t channels = shape[1];
  size_t height = shape[2];
  size_t width = shape[3];

  size_t padded_h = height + 2 * pad_h;
  size_t padded_w = width + 2 * pad_w;
  size_t output_h = (padded_h - kernel_h) / stride_h + 1;
  size_t output_w = (padded_w - kernel_w) / stride_w + 1;

  if (input.device_type() == DeviceType::CPU) {
    return create_cpu_task(device, stream, tunx::cpu::im2col, dtype, input.data_as(),
                           col_data.data_as(), batch_size, channels, height, width, kernel_h,
                           kernel_w, stride_h, stride_w, pad_h, pad_w, output_h, output_w);
  }
#ifdef TUNX_USE_CUDA
  else if (input.device_type() == DeviceType::CUDA) {
    return create_cuda_task(device, stream, tunx::cuda::im2col, dtype, input.data_as(),
                            col_data.data_as(), batch_size, channels, height, width, kernel_h,
                            kernel_w, stride_h, stride_w, pad_h, pad_w, output_h, output_w);
  }
#endif
  else {
    throw std::runtime_error("im2col: Unsupported device type");
  }
}

inline void col2im(const Tensor &col_data, Tensor &result_data, size_t batch_size, size_t channels,
                   size_t height, size_t width, size_t kernel_h, size_t kernel_w, size_t stride_h,
                   size_t stride_w, size_t pad_h, size_t pad_w, stream stream = nullptr) {
  if (col_data.device_type() != result_data.device_type()) {
    throw std::runtime_error("col2im: Mismatched device types between col_data and result_data");
  }

  if (col_data.dtype() != result_data.dtype()) {
    throw std::runtime_error("col2im: Mismatched data types between col_data and result_data");
  }

  DType_t dtype = col_data.dtype();
  size_t padded_h = height + 2 * pad_h;
  size_t padded_w = width + 2 * pad_w;
  size_t output_h = (padded_h - kernel_h) / stride_h + 1;
  size_t output_w = (padded_w - kernel_w) / stride_w + 1;

  auto &device = col_data.device();
  if (col_data.device_type() == DeviceType::CPU) {
    return create_cpu_task(device, stream, tunx::cpu::col2im, dtype, col_data.data_as<void>(),
                           result_data.data_as<void>(), batch_size, channels, height, width,
                           kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, output_h,
                           output_w);
  }
#ifdef TUNX_USE_CUDA
  else if (col_data.device_type() == DeviceType::CUDA) {
    return create_cuda_task(device, stream, tunx::cuda::col2im, dtype, col_data.data_as(),
                            result_data.data_as(), batch_size, channels, height, width, kernel_h,
                            kernel_w, stride_h, stride_w, pad_h, pad_w, output_h, output_w);
  }
#endif
  else {
    throw std::runtime_error("col2im: Unsupported device type");
  }
}

inline void pad(const Tensor &input, Tensor &result, size_t pad_h, size_t pad_w, double value = 0.0,
                stream stream = nullptr) {
  if (input.device_type() != result.device_type()) {
    throw std::runtime_error("pad: Mismatched device types between input and result");
  }

  if (input.dtype() != result.dtype()) {
    throw std::runtime_error("pad: Mismatched data types between input and result");
  }

  const auto &shape = input.shape();
  if (shape.size() != 4) {
    throw std::invalid_argument("pad: Input tensor must be 4-dimensional (NCHW)");
  }

  DType_t dtype = input.dtype();
  size_t batch_size = shape[0];
  size_t channels = shape[1];
  size_t height = shape[2];
  size_t width = shape[3];

  auto &device = input.device();
  if (input.device_type() == DeviceType::CPU) {
    return create_cpu_task(device, stream, tunx::cpu::pad, dtype, input.data_as(), result.data_as(),
                           batch_size, channels, height, width, pad_h, pad_w, value);
  }
#ifdef TUNX_USE_CUDA
  else if (input.device_type() == DeviceType::CUDA) {
    return create_cuda_task(device, stream, tunx::cuda::pad, dtype, input.data_as(),
                            result.data_as(), batch_size, channels, height, width, pad_h, pad_w,
                            value);
  }
#endif
  else {
    throw std::runtime_error("pad: Unsupported device type");
  }
}

inline void unpad(const Tensor &input, Tensor &result, size_t pad_h, size_t pad_w,
                  stream stream = nullptr) {
  if (input.device_type() != result.device_type()) {
    throw std::runtime_error("unpad: Mismatched device types between input and result");
  }

  if (input.dtype() != result.dtype()) {
    throw std::runtime_error("unpad: Mismatched data types between input and result");
  }

  const auto &shape = input.shape();
  if (shape.size() != 4) {
    throw std::invalid_argument("unpad: Input tensor must be 4-dimensional (NCHW)");
  }

  size_t padded_height = shape[2];
  size_t padded_width = shape[3];

  if (padded_height <= 2 * pad_h || padded_width <= 2 * pad_w) {
    throw std::invalid_argument("Padding size too large for unpadding");
  }

  DType_t dtype = input.dtype();
  size_t batch_size = shape[0];
  size_t channels = shape[1];
  size_t height = padded_height - 2 * pad_h;
  size_t width = padded_width - 2 * pad_w;

  auto &device = input.device();
  if (input.device_type() == DeviceType::CPU) {
    return create_cpu_task(device, stream, tunx::cpu::unpad, dtype, input.data_as(),
                           result.data_as(), batch_size, channels, height, width, pad_h, pad_w);
  }
#ifdef TUNX_USE_CUDA
  else if (input.device_type() == DeviceType::CUDA) {
    return create_cuda_task(device, stream, tunx::cuda::unpad, dtype, input.data_as(),
                            result.data_as(), batch_size, channels, height, width, pad_h, pad_w);
  }
#endif
  else {
    throw std::runtime_error("unpad: Unsupported device type");
  }
}

inline void crop(const Tensor &input, Tensor &result, size_t start_h, size_t start_w, size_t end_h,
                 size_t end_w, stream stream = nullptr) {
  if (input.device_type() != result.device_type()) {
    throw std::runtime_error("crop: Mismatched device types between input and result");
  }

  if (input.dtype() != result.dtype()) {
    throw std::runtime_error("crop: Mismatched data types between input and result");
  }

  const auto &shape = input.shape();
  if (shape.size() != 4) {
    throw std::invalid_argument("crop: Input tensor must be 4-dimensional (NCHW)");
  }

  size_t height = shape[2];
  size_t width = shape[3];

  if (end_h >= height || end_w >= width || start_h > end_h || start_w > end_w) {
    throw std::invalid_argument("Invalid crop dimensions");
  }

  DType_t dtype = input.dtype();
  size_t batch_size = shape[0];
  size_t channels = shape[1];
  size_t new_height = end_h - start_h + 1;
  size_t new_width = end_w - start_w + 1;

  auto &device = input.device();
  if (input.device_type() == DeviceType::CPU) {
    return create_cpu_task(device, stream, tunx::cpu::crop, dtype, input.data_as(),
                           result.data_as(), batch_size, channels, height, width, start_h, start_w,
                           new_height, new_width);
  }
#ifdef TUNX_USE_CUDA
  else if (input.device_type() == DeviceType::CUDA) {
    return create_cuda_task(device, stream, tunx::cuda::crop, dtype, input.data_as(),
                            result.data_as(), batch_size, channels, height, width, start_h, start_w,
                            new_height, new_width);
  }
#endif
  else {
    throw std::runtime_error("crop: Unsupported device type");
  }
}

inline void slice_batch(const Tensor &input, Tensor &result, size_t start_batch, size_t end_batch,
                        stream stream = nullptr) {
  if (input.device_type() != result.device_type()) {
    throw std::runtime_error("slice_batch: Mismatched device types between input and result");
  }

  if (input.dtype() != result.dtype()) {
    throw std::runtime_error("slice_batch: Mismatched data types between input and result");
  }

  const auto &shape = input.shape();
  size_t batch_size = shape[0];

  if (end_batch > batch_size || start_batch > end_batch) {
    throw std::invalid_argument("Invalid batch slice range");
  }

  size_t batch_stride = 1;
  for (size_t i = 1; i < shape.size(); ++i) {
    batch_stride *= shape[i];
  }

  Vec<size_t> result_shape = shape;
  result_shape[0] = end_batch - start_batch;
  result = Tensor(result_shape, input.dtype(), input.allocator());

  size_t copy_size = (end_batch - start_batch) * batch_stride;
  DType_t dtype = input.dtype();

  auto &device = input.device();
  if (input.device_type() == DeviceType::CPU) {
    return create_cpu_task(
        device, stream, kernel::cpu::copy, dtype,
        input.data_as<char>() + start_batch * batch_stride * get_dtype_size(dtype),
        result.data_as(), copy_size);
  }
#ifdef TUNX_USE_CUDA
  else if (input.device_type() == DeviceType::CUDA) {
    return create_cuda_task(
        device, stream, kernel::cuda::copy, dtype,
        input.data_as<char>() + start_batch * batch_stride * get_dtype_size(dtype),
        result.data_as(), copy_size);
  }
#endif
  else {
    throw std::runtime_error("slice_batch: Unsupported device type");
  }
}

inline void split(const Tensor &input, Vec<Tensor> &results, size_t num_splits,
                  stream stream = nullptr) {
  const auto &shape = input.shape();
  size_t batch_size = shape[0];

  if (num_splits == 0 || num_splits > batch_size) {
    throw std::invalid_argument("Invalid number of splits");
  }

  DType_t dtype = input.dtype();

  results.clear();
  results.reserve(num_splits);
  size_t split_size = batch_size / num_splits;

  size_t batch_stride = 1;
  for (size_t j = 1; j < shape.size(); ++j) {
    batch_stride *= shape[j];
  }

  for (size_t i = 0; i < num_splits; ++i) {
    size_t start = i * split_size;
    size_t end = (i == num_splits - 1) ? batch_size : start + split_size;

    Vec<size_t> split_shape = shape;
    split_shape[0] = end - start;

    Tensor split_tensor = Tensor(split_shape, input.dtype(), input.allocator());

    size_t copy_size = (end - start) * batch_stride;

    auto &device = input.device();
    if (input.device_type() == DeviceType::CPU) {
      create_cpu_task(device, stream, kernel::cpu::copy, dtype,
                      input.data_as<char>() + start * batch_stride * get_dtype_size(dtype),
                      split_tensor.data_as(), copy_size);
    }
#ifdef TUNX_USE_CUDA
    else if (input.device_type() == DeviceType::CUDA) {
      create_cuda_task(device, stream, kernel::cuda::copy, dtype,
                       input.data_as<char>() + start * batch_stride * get_dtype_size(dtype),
                       split_tensor.data_as(), copy_size);
    }
#endif
    else {
      throw std::runtime_error("split: Unsupported device type");
    }
    results.push_back(split_tensor);
  }
}

inline void transpose_2d(const Tensor &input, Tensor &output, size_t rows, size_t cols,
                         stream stream = nullptr) {
  if (output.device() != input.device()) {
    throw std::runtime_error("transpose_2d: Input and output must be on the same device");
  }

  if (input.dtype() != output.dtype()) {
    throw std::runtime_error("transpose_2d: Mismatched data types between input and output");
  }

  DType_t dtype = input.dtype();

  auto &device = input.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, tunx::cpu::transpose_2d, dtype, input.data_as(),
                           output.data_as(), rows, cols);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, tunx::cuda::transpose_2d, dtype, input.data_as(),
                            output.data_as(), rows, cols);
  }
#endif
  else {
    throw std::runtime_error("transpose_2d: Unsupported device type");
  }
}

inline void nchw_to_cnhw(const Tensor &input, Tensor &output, size_t n, size_t c, size_t h,
                         size_t w, stream stream = nullptr) {
  if (output.device() != input.device()) {
    throw std::runtime_error("nchw_to_cnhw: Input and output must be on the same device");
  }

  if (input.dtype() != output.dtype()) {
    throw std::runtime_error("nchw_to_cnhw: Mismatched data types between input and output");
  }

  DType_t dtype = input.dtype();

  auto &device = input.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, tunx::cpu::nchw_to_cnhw, dtype, input.data_as(),
                           output.data_as(), n, c, h, w);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, tunx::cuda::nchw_to_cnhw, dtype, input.data_as(),
                            output.data_as(), n, c, h, w);
  }
#endif
  else {
    throw std::runtime_error("nchw_to_cnhw: Unsupported device type");
  }
}

inline void cnhw_to_nchw(const Tensor &input, Tensor &output, size_t n, size_t c, size_t h,
                         size_t w, stream stream = nullptr) {
  if (output.device() != input.device()) {
    throw std::runtime_error("cnhw_to_nchw: Input and output must be on the same device");
  }

  if (input.dtype() != output.dtype()) {
    throw std::runtime_error("cnhw_to_nchw: Mismatched data types between input and output");
  }

  DType_t dtype = input.dtype();

  auto &device = input.device();
  auto device_type = device.device_type();

  if (device_type == DeviceType::CPU) {
    return create_cpu_task(device, stream, tunx::cpu::cnhw_to_nchw, dtype, input.data_as(),
                           output.data_as(), n, c, h, w);
  }
#ifdef TUNX_USE_CUDA
  else if (device_type == DeviceType::CUDA) {
    return create_cuda_task(device, stream, tunx::cuda::cnhw_to_nchw, dtype, input.data_as(),
                            output.data_as(), n, c, h, w);
  }
#endif
  else {
    throw std::runtime_error("cnhw_to_nchw: Unsupported device type");
  }
}

inline void check_equals(const Tensor &t1, const Tensor &t2, bool &result, double eps = 1e-5,
                         stream stream = nullptr) {
  if (t1.shape() != t2.shape()) {
    throw std::runtime_error("check_equals: Shapes mismatch");
  }

  if (t1.dtype() != t2.dtype()) {
    throw std::runtime_error("check_equals: Data types mismatch");
  }

  if (t1.device() != t2.device()) {
    throw std::runtime_error("check_equals: Devices mismatch");
  }

  DType_t dtype = t1.dtype();

  DISPATCH_ANY_DTYPE(
      dtype, T,
      return kernel::check_equals(dtype_of<T>(), t1.data_ptr(), t2.data_ptr(), result, eps));
}

inline void print_tensor(const Tensor &tensor, size_t num_elements_, std::string_view label) {
  auto host_tensor = tensor.to_host();
  fmt::print("{}: ", label);
  for (size_t i = 0; i < num_elements_; i++) {
    DISPATCH_ANY_DTYPE(tensor.dtype(), T,
                       { fmt::print("{:.3f} ", static_cast<float>(host_tensor.data_as<T>()[i])); })
  }
  fmt::print("\n");
}

}  // namespace tunx