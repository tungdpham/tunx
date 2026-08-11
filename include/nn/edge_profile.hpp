#pragma once

#include <cstdint>

namespace tunx {
struct EdgeProfile {
  double exec_time;
  int64_t total_mem;      // memory to execute (including workspace and output memory)
  int64_t net_mem;        // delta between after - before.
  int64_t secondary_mem;  // memory for secondary stats (e.g. attention stats, batch mean, var)
  int64_t workspace_mem;  // memory for temporal workspace
  int64_t input_mem;      // memory for input tensors
  int64_t output_mem;     // memory for output tensors
};
}  // namespace tunx