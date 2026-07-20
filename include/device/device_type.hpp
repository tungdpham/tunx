/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#pragma once

#include <string>
namespace tunx {
// all supported device types
enum class DeviceType { CPU, CUDA, UNKNOWN };

inline std::string device_type_to_string(DeviceType dev_type) {
  switch (dev_type) {
    case DeviceType::CPU:
      return "CPU";
    case DeviceType::CUDA:
      return "CUDA";
    case DeviceType::UNKNOWN:
      return "UNKNOWN";
    default:
      return "IMPOSSIBLE";
  }
}

inline DeviceType device_type_from_string(std::string_view str) {
  if (str == "CPU") {
    return DeviceType::CPU;
  } else if (str == "CUDA") {
    return DeviceType::CUDA;
  } else {
    return DeviceType::UNKNOWN;
  }
}

}  // namespace tunx