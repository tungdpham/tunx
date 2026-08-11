#pragma once

#include <cstddef>
#include <ostream>
#include <vector>

#include "nn/edge_profile.hpp"
#include "nn/execution_plan.hpp"
#include "nn/graph.hpp"

namespace tunx {
class MacroSolver {
private:
  Graph &graph_;
  std::ostream *log_stream_;

public:
  MacroSolver(Graph &graph, std::ostream *log_stream = nullptr)
      : graph_(graph),
        log_stream_(log_stream) {}

  ExecutionPlan find_order(const std::map<Edge, EdgeProfile> &edge_profiles);
};
}  // namespace tunx