#pragma once

#include <vector>
#include <string>

#include "nn/edge.hpp"
namespace tunx {

struct ExecutionPlan {
  std::vector<Edge> order;
};

struct EdgeMemStats {
  std::string layer_name;
  size_t allocated_mem;
  size_t reserved_mem;
  size_t unused_mem;
};

struct ExecutionPlanStats {
  size_t peak_mem;
  std::vector<EdgeMemStats> edge_stats;
};

}  // namespace tunx