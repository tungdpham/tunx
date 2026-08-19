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
  int64_t allocated_mem;
  int64_t reserved_mem;
  int64_t peak_mem;
  int64_t cached_mem;
  int64_t fragmented_mem;
  int64_t host_mem;
  int64_t gradients_mem;
  int64_t activations_mem;
  int64_t workspaces_mem;
};

struct ExecutionPlanStats {
  size_t peak_mem;  // should just be the last peak mem in edge stats.
  std::vector<EdgeMemStats> edge_stats;
};

}  // namespace tunx