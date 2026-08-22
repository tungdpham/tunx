#pragma once

#include <algorithm>
#include <bitset>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "allocator.h"
#include "graph.h"
#include "graph_executor.h"
#include "macro_solver.h"

inline std::vector<std::string> find_fw_fork_join_execution_order(Graph& graph, size_t max_states,
                                                                  bool* oracle_complete) {
  std::vector<std::string> op_ids;
  std::map<std::string, int> op_index;
  for (const auto& [uuid, op] : graph.op_nodes()) {
    if (op_ids.size() == 256) return find_fw_macro_candidate_execution_order(graph);
    op_index[op.uuid()] = static_cast<int>(op_ids.size());
    op_ids.push_back(op.uuid());
  }

  std::vector<std::string> baseline_order = find_fw_macro_candidate_execution_order(graph);
  size_t baseline_peak = 0;
  {
    Allocator temp_allocator;
    GraphExecutor temp_executor(graph);
    temp_allocator.subscribe(
        "baseline", [&](size_t memory) { baseline_peak = std::max(baseline_peak, memory); });
    temp_executor.init_boundaries(&temp_allocator);
    for (const auto& op_id : baseline_order) {
      temp_executor.run_op_node(&graph.get_op(op_id), &temp_allocator);
    }
    temp_allocator.unsubscribe("baseline");
  }

  auto [dependencies, dependents] = get_op_dependencies(graph);
  struct Schedule {
    size_t peak;
    std::vector<std::string> order;
  };

  Allocator allocator;
  GraphExecutor executor(graph);
  size_t initial_peak = 0;
  allocator.subscribe("fork_join_dp_init",
                      [&](size_t memory) { initial_peak = std::max(initial_peak, memory); });
  executor.init_boundaries(&allocator);
  allocator.unsubscribe("fork_join_dp_init");

  std::unordered_map<std::bitset<256>, Schedule> memo;
  bool state_limit_reached = false;
  if (oracle_complete) *oracle_complete = true;
  std::function<Schedule(std::bitset<256>&)> solve = [&](std::bitset<256>& completed) {
    if (const auto it = memo.find(completed); it != memo.end()) return it->second;
    if (memo.size() >= max_states) {
      state_limit_reached = true;
      if (oracle_complete) *oracle_complete = false;
      return Schedule{std::numeric_limits<size_t>::max(), {}};
    }
    if (allocator.allocated() >= baseline_peak) {
      return memo.emplace(completed, Schedule{std::numeric_limits<size_t>::max(), {}})
          .first->second;
    }

    if (completed.count() == op_ids.size()) {
      return memo.emplace(completed, Schedule{allocator.allocated(), {}}).first->second;
    }

    Schedule best{std::numeric_limits<size_t>::max(), {}};
    for (const auto& op_id : op_ids) {
      const int index = op_index[op_id];
      if (completed.test(index)) continue;

      bool ready = true;
      for (const auto& dependency : dependencies[op_id]) {
        if (!completed.test(op_index[dependency])) {
          ready = false;
          break;
        }
      }
      if (!ready) continue;

      const auto& op = graph.get_op(op_id);
      size_t op_peak = allocator.allocated();
      allocator.subscribe("fork_join_dp",
                          [&](size_t memory) { op_peak = std::max(op_peak, memory); });
      executor.run_op_node(&op, &allocator);
      allocator.unsubscribe("fork_join_dp");

      if (op_peak >= baseline_peak || op_peak >= best.peak) {
        executor.undo_run_op_node(&op, &allocator);
        continue;
      }

      completed.set(index);

      Schedule suffix = solve(completed);
      const size_t candidate_peak = std::max(op_peak, suffix.peak);
      if (candidate_peak < best.peak) {
        best.peak = candidate_peak;
        best.order = {op_id};
        best.order.insert(best.order.end(), suffix.order.begin(), suffix.order.end());
      }

      completed.reset(index);
      executor.undo_run_op_node(&op, &allocator);
    }
    return memo.emplace(completed, std::move(best)).first->second;
  };

  std::bitset<256> completed;
  Schedule result = solve(completed);
  if (result.peak >= baseline_peak) {
    return baseline_order;
  }
  return result.order;
}

inline std::vector<std::string> find_fw_naive_dfs_execution_order(Graph& graph) {
  auto [dependencies, dependents] = get_op_dependencies(graph);
  std::set<std::string> visited;
  std::vector<std::string> order;

  std::vector<std::string> op_ids;
  for (const auto& [uuid, op] : graph.op_nodes()) op_ids.push_back(uuid);
  std::sort(op_ids.begin(), op_ids.end());

  std::function<void(const std::string&)> dfs = [&](const std::string& op_id) {
    if (visited.count(op_id)) return;
    visited.insert(op_id);

    std::vector<std::string> deps(dependencies[op_id].begin(), dependencies[op_id].end());
    std::sort(deps.begin(), deps.end());
    for (const auto& dep : deps) dfs(dep);

    order.push_back(op_id);
  };

  for (const auto& uuid : op_ids) dfs(uuid);

  return order;
}

inline std::vector<std::string> find_bw_fork_join_execution_order(Graph& graph,
                                                                  size_t max_states = 1000000) {
  std::vector<std::string> op_ids;
  std::map<std::string, int> op_index;
  for (const auto& [uuid, op] : graph.op_nodes()) {
    if (op_ids.size() == 256) {
      MacroSolver solver(graph);
      return solver.find_backward_order();
    }
    op_index[op.uuid()] = static_cast<int>(op_ids.size());
    op_ids.push_back(op.uuid());
  }

  std::vector<std::string> baseline_order;
  {
    MacroSolver solver(graph);
    baseline_order = solver.find_backward_order();
  }
  size_t baseline_peak = 0;
  {
    Allocator temp_allocator;
    GraphExecutor temp_executor(graph);
    temp_executor.init_boundaries(&temp_allocator);
    std::vector<std::string> fw_order = find_fw_naive_dfs_execution_order(graph);
    for (const auto& op_id : fw_order) {
      temp_executor.run_op_node(&graph.get_op(op_id), &temp_allocator);
    }
    temp_allocator.subscribe(
        "baseline", [&](size_t memory) { baseline_peak = std::max(baseline_peak, memory); });
    temp_executor.transition_to_backward(&temp_allocator);
    for (const auto& op_id : baseline_order) {
      temp_executor.run_backward_op_node(&graph.get_op(op_id), &temp_allocator);
    }
    temp_allocator.unsubscribe("baseline");
  }

  auto deps_and_dependents = get_op_dependencies(graph);
  auto dependencies = deps_and_dependents.second;

  struct Schedule {
    size_t peak;
    std::vector<std::string> order;
  };

  Allocator allocator;
  GraphExecutor executor(graph);

  executor.init_boundaries(&allocator);
  std::vector<std::string> fw_order = find_fw_naive_dfs_execution_order(graph);
  for (const auto& op_id : fw_order) {
    OperationNode* op_node = nullptr;
    for (auto& [uuid, node] : graph.op_nodes()) {
      if (node.uuid() == op_id) {
        op_node = &node;
        break;
      }
    }
    if (op_node) executor.run_op_node(op_node, &allocator);
  }

  size_t initial_peak = 0;
  allocator.subscribe("fork_join_dp_init",
                      [&](size_t memory) { initial_peak = std::max(initial_peak, memory); });
  executor.transition_to_backward(&allocator);
  allocator.unsubscribe("fork_join_dp_init");

  std::unordered_map<std::bitset<256>, Schedule> memo;
  bool state_limit_reached = false;
  std::function<Schedule(std::bitset<256>&)> solve = [&](std::bitset<256>& completed) {
    if (const auto it = memo.find(completed); it != memo.end()) return it->second;
    if (memo.size() >= max_states) {
      state_limit_reached = true;
      return Schedule{std::numeric_limits<size_t>::max(), {}};
    }
    if (allocator.allocated() >= baseline_peak) {
      return memo.emplace(completed, Schedule{std::numeric_limits<size_t>::max(), {}})
          .first->second;
    }

    if (completed.count() == op_ids.size()) {
      return memo.emplace(completed, Schedule{allocator.allocated(), {}}).first->second;
    }

    Schedule best{std::numeric_limits<size_t>::max(), {}};
    for (const auto& op_id : op_ids) {
      const int index = op_index[op_id];
      if (completed.test(index)) continue;

      bool ready = true;
      for (const auto& dependency : dependencies[op_id]) {
        if (!completed.test(op_index[dependency])) {
          ready = false;
          break;
        }
      }
      if (!ready) continue;

      const auto& op = graph.get_op(op_id);
      size_t op_peak = allocator.allocated();
      allocator.subscribe("fork_join_dp",
                          [&](size_t memory) { op_peak = std::max(op_peak, memory); });
      executor.run_backward_op_node(&op, &allocator);
      allocator.unsubscribe("fork_join_dp");

      if (op_peak >= baseline_peak || op_peak >= best.peak) {
        executor.undo_run_backward_op_node(&op, &allocator);
        continue;
      }

      completed.set(index);

      Schedule suffix = solve(completed);
      const size_t candidate_peak = std::max(op_peak, suffix.peak);
      if (candidate_peak < best.peak) {
        best.peak = candidate_peak;
        best.order = {op_id};
        best.order.insert(best.order.end(), suffix.order.begin(), suffix.order.end());
      }

      completed.reset(index);
      executor.undo_run_backward_op_node(&op, &allocator);
    }
    return memo.emplace(completed, std::move(best)).first->second;
  };

  std::bitset<256> completed;
  Schedule result = solve(completed);
  if (result.peak >= baseline_peak) {
    return baseline_order;
  }
  return result.order;
}

inline std::vector<std::string> find_bw_naive_dfs_execution_order(Graph& graph) {
  auto deps_and_dependents = get_op_dependencies(graph);
  auto dependencies = deps_and_dependents.second;
  std::set<std::string> visited;
  std::vector<std::string> order;

  std::vector<std::string> op_ids;
  for (const auto& [uuid, op] : graph.op_nodes()) op_ids.push_back(uuid);
  std::sort(op_ids.begin(), op_ids.end());

  std::function<void(const std::string&)> dfs = [&](const std::string& op_id) {
    if (visited.count(op_id)) return;
    visited.insert(op_id);

    std::vector<std::string> deps(dependencies[op_id].begin(), dependencies[op_id].end());
    std::sort(deps.begin(), deps.end());
    for (const auto& dep : deps) dfs(dep);

    order.push_back(op_id);
  };

  for (const auto& uuid : op_ids) dfs(uuid);

  return order;
}