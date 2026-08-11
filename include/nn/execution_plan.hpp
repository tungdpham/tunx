#pragma once

#include <vector>

#include "nn/edge.hpp"
namespace tunx {

struct ExecutionPlan {
  std::vector<Edge> order;
};

}  // namespace tunx