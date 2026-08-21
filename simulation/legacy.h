#pragma once

#include <algorithm>
#include <bitset>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "graph.h"
#include "graph_executor.h"

inline std::vector<std::string> find_legacy_minimum_memory_execution_order(Graph& graph) {
  std::vector<std::string> op_ids;
  std::map<std::string, int> op_index;
  for (auto& [uuid, node] : graph.op_nodes()) {
    op_index[node.uuid()] = static_cast<int>(op_ids.size());
    op_ids.push_back(node.uuid());
  }
  const int n = static_cast<int>(op_ids.size());
  if (n > 256) {
    throw std::runtime_error("Too many ops for bitmask memoization (max 256).");
  }

  auto out_deg = get_out_deg(graph);
  auto [deps, dependents] = get_op_dependencies(graph);

  std::map<std::string, int> in_degree;
  for (auto& id : op_ids) {
    in_degree[id] = static_cast<int>(deps[id].size());
  }

  auto get_score = [&](const std::string& id) -> long long {
    auto& node = graph.get_op(id);
    long long score = static_cast<long long>(node.workspace_req());
    for (auto* t : node.inputs()) {
      // Note: out_deg here is a static estimation to guide heuristic,
      // not a dynamic tracking value, which is fine for scoring.
      if (out_deg[t->uuid()] == 1) {
        score -= static_cast<long long>(t->size());
      }
    }
    for (auto* t : node.outputs()) {
      score += static_cast<long long>(t->size());
    }
    return score;
  };

  std::unordered_map<std::bitset<256>, size_t> memo;
  size_t best_peak = std::numeric_limits<size_t>::max();
  std::vector<std::string> best_order;
  [[maybe_unused]] int num_orderings = 0;
  [[maybe_unused]] int num_pruned = 0;

  Allocator allocator;
  GraphExecutor executor(graph);

  // --- Seed best_peak with the greedy heuristic solution ---
  {
    std::vector<std::string> heuristic_order;
    if (!heuristic_order.empty()) {
      size_t heuristic_peak = 0;
      allocator.subscribe("heuristic", [&](size_t new_mem) {
        if (new_mem > heuristic_peak) heuristic_peak = new_mem;
      });
      executor.init_boundaries(&allocator);
      for (const auto& op_id : heuristic_order) {
        OperationNode* op_node = nullptr;
        for (auto& [uuid, node] : graph.op_nodes()) {
          if (node.uuid() == op_id) {
            op_node = &node;
            break;
          }
        }
        executor.run_op_node(op_node, &allocator);
      }
      allocator.unsubscribe("heuristic");
      best_peak = heuristic_peak;
      best_order = heuristic_order;
      num_orderings = 1;

      for (auto it = heuristic_order.rbegin(); it != heuristic_order.rend(); ++it) {
        OperationNode* op_node = nullptr;
        for (auto& [uuid, node] : graph.op_nodes()) {
          if (node.uuid() == *it) {
            op_node = &node;
            break;
          }
        }
        executor.undo_run_op_node(op_node, &allocator);
      }
      // we will re-init inside dfs
    }
  }

  struct Frame {
    std::vector<std::string> candidates;
    int candidate_idx;
    std::string chosen_op;
    std::bitset<256> executed_mask;
    size_t local_peak;
  };

  std::vector<std::string> current_order;
  std::set<std::string> executed;
  std::bitset<256> executed_mask;

  // Allocate graph input tensors before starting the search.
  Allocator dfs_allocator;
  GraphExecutor dfs_executor(graph);
  size_t initial_peak = 0;
  dfs_allocator.subscribe("init", [&](size_t new_mem) {
    if (new_mem > initial_peak) initial_peak = new_mem;
  });
  dfs_executor.init_boundaries(&dfs_allocator);
  dfs_allocator.unsubscribe("init");

  auto get_sorted_candidates = [&]() -> std::vector<std::string> {
    std::vector<std::pair<long long, std::string>> scored;
    for (auto& id : op_ids) {
      if (executed.count(id)) continue;
      if (in_degree[id] != 0) continue;
      scored.push_back({get_score(id), id});
    }
    std::sort(scored.begin(), scored.end());
    std::vector<std::string> result;
    result.reserve(scored.size());
    for (auto& [s, id] : scored) {
      result.push_back(std::move(id));
    }
    return result;
  };

  std::vector<Frame> stack;
  stack.push_back({get_sorted_candidates(), 0, "", std::bitset<256>(), initial_peak});

  while (!stack.empty()) {
    auto& frame = stack.back();

    if (!frame.chosen_op.empty()) {
      OperationNode* op_node = nullptr;
      for (auto& [uuid, node] : graph.op_nodes()) {
        if (node.uuid() == frame.chosen_op) {
          op_node = &node;
          break;
        }
      }
      dfs_executor.undo_run_op_node(op_node, &dfs_allocator);

      current_order.pop_back();
      executed.erase(frame.chosen_op);
      executed_mask.reset(op_index[frame.chosen_op]);
      for (auto& dep : dependents[frame.chosen_op]) {
        in_degree[dep]++;
      }
      frame.chosen_op.clear();
      frame.candidate_idx++;
    }

    bool pushed_child = false;
    while (frame.candidate_idx < static_cast<int>(frame.candidates.size())) {
      const auto& id = frame.candidates[frame.candidate_idx];

      executed.insert(id);
      executed_mask.set(op_index[id]);
      current_order.push_back(id);
      for (auto& dep : dependents[id]) {
        in_degree[dep]--;
      }

      frame.chosen_op = id;
      OperationNode* op_node = nullptr;
      for (auto& [uuid, node] : graph.op_nodes()) {
        if (node.uuid() == id) {
          op_node = &node;
          break;
        }
      }
      size_t op_peak = dfs_allocator.allocated();
      dfs_allocator.subscribe("dfs", [&](size_t new_mem) {
        if (new_mem > op_peak) op_peak = new_mem;
      });
      dfs_executor.run_op_node(op_node, &dfs_allocator);
      dfs_allocator.unsubscribe("dfs");

      size_t new_path_peak = std::max(frame.local_peak, op_peak);

      if (new_path_peak > best_peak) {
        num_pruned++;
        dfs_executor.undo_run_op_node(op_node, &dfs_allocator);
        current_order.pop_back();
        executed.erase(id);
        executed_mask.reset(op_index[id]);
        for (auto& dep : dependents[id]) {
          in_degree[dep]++;
        }
        frame.chosen_op.clear();
        frame.candidate_idx++;
        continue;
      }

      auto memo_it = memo.find(executed_mask);
      if (memo_it != memo.end() && memo_it->second <= new_path_peak) {
        num_pruned++;
        dfs_executor.undo_run_op_node(op_node, &dfs_allocator);
        current_order.pop_back();
        executed.erase(id);
        executed_mask.reset(op_index[id]);
        for (auto& dep : dependents[id]) {
          in_degree[dep]++;
        }
        frame.chosen_op.clear();
        frame.candidate_idx++;
        continue;
      }
      memo[executed_mask] = new_path_peak;

      if (static_cast<int>(current_order.size()) == n) {
        num_orderings++;
        if (new_path_peak < best_peak) {
          best_peak = new_path_peak;
          best_order = current_order;
        }
        break;
      }

      stack.push_back({get_sorted_candidates(), 0, "", executed_mask, new_path_peak});
      pushed_child = true;
      break;
    }

    if (!pushed_child && (frame.chosen_op.empty() ||
                          frame.candidate_idx >= static_cast<int>(frame.candidates.size()) - 1)) {
      if (!frame.chosen_op.empty()) {
        OperationNode* op_node = nullptr;
        for (auto& [uuid, node] : graph.op_nodes()) {
          if (node.uuid() == frame.chosen_op) {
            op_node = &node;
            break;
          }
        }
        dfs_executor.undo_run_op_node(op_node, &dfs_allocator);
        current_order.pop_back();
        executed.erase(frame.chosen_op);
        executed_mask.reset(op_index[frame.chosen_op]);
        for (auto& dep : dependents[frame.chosen_op]) {
          in_degree[dep]++;
        }
      }
      stack.pop_back();
    }
  }

  return best_order;
}