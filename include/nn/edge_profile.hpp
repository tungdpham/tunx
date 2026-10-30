#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <optional>

namespace tunx {
enum class BufferRole {
    Output,
    SavedInputAlias,
    SavedOutputAlias,
    SecondaryResidual,
    Workspace,
    GradientOutput,
    GradientContribution
};

struct BufferProfile {
    std::string local_id;
    BufferRole role;
    size_t bytes;
    size_t alignment;
    std::optional<std::string> alias_node_uid;
};

struct EdgeProfile {
  double exec_time;
  int64_t total_mem;      // memory to execute (including workspace and output memory)
  int64_t net_mem;        // delta between after - before.
  int64_t secondary_mem;  // memory for secondary stats (e.g. attention stats, batch mean, var)
  int64_t workspace_mem;  // memory for temporal workspace
  int64_t input_mem;      // memory for input tensors
  int64_t output_mem;     // memory for output tensors
  std::vector<std::string> cached_nodes;  // uid of nodes that are cached in residuals
  std::vector<BufferProfile> forward_buffers;
  std::vector<BufferProfile> backward_buffers;
};
}  // namespace tunx