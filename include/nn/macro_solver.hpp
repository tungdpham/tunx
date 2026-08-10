#pragma once

#include <cstddef>
#include <ostream>
#include <vector>

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

  std::vector<size_t> find_order(TensorBundle &input_map);
};
}  // namespace tunx