#pragma once

#include <string>
#include <vector>

#include "nn/edge.hpp"
namespace tunx {

struct ExecutionPlan {
  std::vector<Edge> order;
};

struct EdgeMemStats {
  std::string layer_name;
  size_t allocated_mem;
  size_t peak_mem;
};

struct ExecutionPlanStats {
  size_t peak_mem;  // should just be the last peak mem in edge stats.
  std::vector<EdgeMemStats> edge_stats;
};

}  // namespace tunx