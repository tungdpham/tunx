/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <ostream>

#include "device.hpp"
#include "device_type.hpp"
#include "type/type.hpp"

namespace tunx {

inline std::vector<std::string> split(const std::string &str, char delim) {
  std::vector<std::string> res;
  auto left = str.begin();

  for (auto it = str.begin(); it != str.end(); ++it) {
    if (*it == delim) {
      res.emplace_back(left, it);
      left = it + 1;
    }
  }
  res.emplace_back(left, str.end());

  return res;
}

struct DeviceID {
  DeviceType type;
  int id;

  bool operator<(const DeviceID &other) const {
    if (type != other.type) {
      return static_cast<int>(type) < static_cast<int>(other.type);
    }
    return id < other.id;
  }

  static DeviceID from_string(const std::string &str) {
    Vec<std::string> parts = split(str, ':');
    std::cout << "Parts: ";
    for (auto s : parts) {
      std::cout << s << " ";
    }
    std::cout << std::endl;
    int id = std::stoi(parts[1]);
    return DeviceID{device_type_from_string(parts[0]), id};
  }
};

class DeviceManager {
public:
  static DeviceManager &instance();

private:
  static DeviceManager instance_;

public:
  DeviceManager();
  ~DeviceManager();

  /**
   * Discover all available devices on the system.
   */
  void discover();

  /**
   * Add a device to the manager.
   * @param device The device to add.
   */
  void add(std::unique_ptr<Device> device);

  /**
   * Remove a device from the manager.
   * @param id The ID of the device to remove.
   */
  void remove(DeviceType type, int id);
  void remove(DeviceID device_id);

  /**
   * Clear all devices from the manager.
   */
  void clear();

  /**
   * Get a device by its ID.
   * @param id The ID of the device to retrieve.
   * @return The device with the specified ID.
   */
  Device &get(DeviceType type, int id) const;
  Device &get(DeviceID device_id) const;

  /**
   * Get all available device ids.
   */
  Vec<DeviceID> get_all() const;

  /**
   * Check if device manager has a device with an id.
   */
  bool has(DeviceType type, int id) const;
  bool has(DeviceID device_id) const;

private:
  std::map<DeviceID, std::unique_ptr<Device>> devices_;
};

void initializeDefaultDevices();
Device &getGPU(size_t gpu_index = 0);
Device &getHost();

inline std::ostream &operator<<(std::ostream &os, const tunx::DeviceID &device_id) {
  os << tunx::device_type_to_string(device_id.type) << ":" << device_id.id;
  return os;
}

}  // namespace tunx
