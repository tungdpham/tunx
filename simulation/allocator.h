#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>

class Allocator {
private:
  size_t allocated_;
  std::unordered_map<std::string, std::function<void(size_t)>> listeners_;

public:
  Allocator()
      : allocated_(0) {}

  bool subscribe(std::string name, std::function<void(size_t)> listener) {
    auto [_, ok] = listeners_.insert({name, listener});
    return ok;
  }

  void unsubscribe(std::string name) {
    auto it = listeners_.find(name);
    if (it != listeners_.end()) {
      listeners_.erase(it);
    }
  }

  void allocate(size_t size) {
    allocated_ += size;
    for (auto& [_, listener] : listeners_) {
      listener(allocated_);
    }
  }

  void free(size_t size) {
    allocated_ -= size;
    for (auto& [_, listener] : listeners_) {
      listener(allocated_);
    }
  }

  size_t allocated() const { return allocated_; }
};
