/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <variant>

#include "tensor/tensor.hpp"

namespace tunx {

class ResidualObject {
private:
  struct Impl {
    using ResidualValue =
        std::variant<std::monostate, std::map<std::string, ResidualObject>, Tensor, Vec<size_t>>;

    ResidualValue data;
  };
  std::shared_ptr<Impl> impl_;

public:
  ResidualObject()
      : impl_(std::make_shared<Impl>()) {}
  ~ResidualObject() = default;
  ResidualObject(const ResidualObject &) = default;
  ResidualObject &operator=(const ResidualObject &) = default;
  ResidualObject(ResidualObject &&) = default;
  ResidualObject &operator=(ResidualObject &&) = default;

  ResidualObject &operator[](const std::string &key) {
    if (impl_->data.index() == 0) {
      impl_->data = std::map<std::string, ResidualObject>{};
    } else if (impl_->data.index() == 1) {
      // already a map, do nothing
    } else if (impl_->data.index() == 2) {
      throw std::runtime_error("ResidualObject: Attempting to index into a leaf node");
    }

    return std::get<1>(impl_->data)[key];
  }

  ResidualObject &operator=(const Tensor &tensor) {
    if (impl_->data.index() == 0) {
      impl_->data = tensor;
    } else if (impl_->data.index() == 1) {
      throw std::runtime_error("ResidualObject: Attempting to assign a Tensor to a non-leaf node");
    }
    return *this;
  }

  ResidualObject &operator=(const Vec<size_t> &vec) {
    if (impl_->data.index() == 0) {
      impl_->data = vec;
    } else if (impl_->data.index() == 1) {
      throw std::runtime_error(
          "ResidualObject: Attempting to assign a Vec<size_t> to a non-leaf node");
    }
    return *this;
  }

  operator Tensor &() {
    if (impl_->data.index() != 2) {
      throw std::runtime_error("ResidualObject: Attempting to convert a non-leaf node to Tensor");
    }
    return std::get<2>(impl_->data);
  }

  operator Vec<size_t> &() {
    if (impl_->data.index() != 3) {
      throw std::runtime_error(
          "ResidualObject: Attempting to convert a non-leaf node to Vec<size_t>");
    }
    return std::get<3>(impl_->data);
  }

  void print(std::ostream &os, int indent = 0) const {
    std::string indent_str(indent * 2, ' ');
    if (!impl_) {
      os << indent_str << "<null impl_>\n";
      return;
    }

    std::visit(
        [&os, indent, &indent_str](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;

          if constexpr (std::is_same_v<T, std::monostate>) {
            os << "<empty>\n";
          } else if constexpr (std::is_same_v<T, std::map<std::string, ResidualObject>>) {
            if (arg.empty()) {
              os << "{}\n";
              return;
            }
            os << "{\n";
            for (const auto &[key, value] : arg) {
              os << indent_str << "  " << key << ": ";
              value.print(os, indent + 1);
            }
            os << indent_str << "}\n";
          } else if constexpr (std::is_same_v<T, Tensor>) {
            // For leaf nodes, output the Tensor
            os << "Tensor(shape=[";
            const auto &shape = arg.shape();
            for (size_t i = 0; i < shape.size(); ++i) {
              os << shape[i];
              if (i < shape.size() - 1) os << ", ";
            }
            os << "])\n";
          } else if constexpr (std::is_same_v<T, Vec<size_t>>) {
            os << "Vec<size_t>[";
            for (size_t i = 0; i < arg.size(); ++i) {
              os << arg[i];
              if (i < arg.size() - 1) os << ", ";
            }
            os << "]\n";
          }
        },
        impl_->data);
  }

  size_t num_bytes() const {
    if (!impl_) return 0;
    return std::visit(
        [](auto &&arg) -> size_t {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, std::monostate>) {
            return 0;
          } else if constexpr (std::is_same_v<T, std::map<std::string, ResidualObject>>) {
            size_t total = 0;
            for (const auto &[key, value] : arg) {
              total += value.num_bytes();
            }
            return total;
          } else if constexpr (std::is_same_v<T, Tensor>) {
            return arg.num_bytes();
          } else if constexpr (std::is_same_v<T, Vec<size_t>>) {
            return arg.size() * sizeof(size_t);
          }
          return 0;
        },
        impl_->data);
  }

  friend std::ostream &operator<<(std::ostream &os, const ResidualObject &obj) {
    obj.print(os, 0);
    return os;
  }

  void apply_tensors(const std::function<void(Tensor &)> &func) {
    if (!impl_) return;
    std::visit(
        [&func](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, std::map<std::string, ResidualObject>>) {
            for (auto &[key, value] : arg) {
              value.apply_tensors(func);
            }
          } else if constexpr (std::is_same_v<T, Tensor>) {
            func(arg);
          }
        },
        impl_->data);
  }

  std::map<std::string, Tensor> tensors() const {
    std::map<std::string, Tensor> res;
    std::visit(
        [&res](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, std::monostate>) {
          } else if constexpr (std::is_same_v<T, std::map<std::string, ResidualObject>>) {
            for (const auto &[key, value] : arg) {
              auto inner_tensors = value.tensors();
              for (auto &[inner_key, inner_value] : inner_tensors) {
                res[key + "." + inner_key] = inner_value;
              }
            }
          } else if constexpr (std::is_same_v<T, Tensor>) {
            res[""] = arg;
          } else if constexpr (std::is_same_v<T, Vec<size_t>>) {
          }
        },
        impl_->data);
    return res;
  }
};

using Residuals = ResidualObject;

}  // namespace tunx
