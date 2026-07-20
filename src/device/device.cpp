#include "device/device.hpp"

namespace tunx {

Device::Device(int id)
    : id_(id) {}

Device::~Device() = default;

Device::Device(Device &&other) noexcept
    : id_(other.id_) {
  other.id_ = -1;
}

Device &Device::operator=(Device &&other) noexcept {
  if (this != &other) {
    id_ = other.id_;
    other.id_ = -1;
  }
  return *this;
}

bool Device::operator==(const Device &other) const { return this == &other; }

int Device::get_id() const { return id_; }

}  // namespace tunx
