#include <algorithm>
#include <bitset>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "allocator.h"
#include "graph.h"
#include "graph_executor.h"
#include "graph_generator.h"
#include "macro_solver.h"
#include "macro_solver_v2.h"
#include "memory_packer.h"

std::vector<std::string> find_fw_macro_candidate_execution_order(Graph& graph,
                                                                 std::ostream* os = nullptr) {
  MacroSolver solver(graph, os);
  return solver.find_forward_order();
}

std::vector<std::string> find_bw_macro_candidate_execution_order(Graph& graph,
                                                                 std::ostream* os = nullptr) {
  MacroSolver solver(graph, os);
  return solver.find_backward_order();
}

std::vector<std::string> find_fw_macro_v2_candidate_execution_order(Graph& graph,
                                                                    std::ostream* os = nullptr) {
  MacroSolverV2 solver(graph, os);
  return solver.find_forward_order();
}

std::vector<std::string> find_bw_macro_v2_candidate_execution_order(Graph& graph,
                                                                    std::ostream* os = nullptr) {
  MacroSolverV2 solver(graph, os);
  return solver.find_backward_order();
}

std::vector<std::string> find_fw_fork_join_execution_order(Graph& graph,
                                                           size_t max_states = 1000000) {
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

std::vector<std::string> find_fw_naive_dfs_execution_order(Graph& graph) {
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

std::vector<std::string> find_bw_fork_join_execution_order(Graph& graph,
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

std::vector<std::string> find_bw_naive_dfs_execution_order(Graph& graph) {
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

inline bool near(double a, double b) { return std::abs(a - b) < 1e-15; }

std::map<std::string, double> rank_execution_orders(
    Graph& g, const std::map<std::string, std::vector<std::string>>& orders) {
  std::map<std::string, size_t> peaks;
  size_t best_peak = std::numeric_limits<size_t>::max();

  for (const auto& [name, order] : orders) {
    if (order.empty()) {
      peaks[name] = std::numeric_limits<size_t>::max();
      continue;
    }
    Allocator allocator;
    GraphExecutor executor(g);
    size_t peak = 0;
    allocator.subscribe("rank", [&](size_t new_mem) {
      if (new_mem > peak) peak = new_mem;
    });
    executor.init_boundaries(&allocator);
    for (const auto& op_id : order) {
      OperationNode* op_node = nullptr;
      for (auto& [uuid, node] : g.op_nodes()) {
        if (node.uuid() == op_id) {
          op_node = &node;
          break;
        }
      }
      if (op_node) executor.run_op_node(op_node, &allocator);
    }
    allocator.unsubscribe("rank");
    peaks[name] = peak;
    if (peaks[name] < best_peak) {
      best_peak = peaks[name];
    }
  }

  std::map<std::string, double> efficiencies;
  for (const auto& [name, peak] : peaks) {
    if (peak == std::numeric_limits<size_t>::max()) {
      efficiencies[name] = 0.0;
    } else {
      efficiencies[name] = (static_cast<double>(best_peak) / static_cast<double>(peak)) * 100.0;
    }
  }
  return efficiencies;
}

std::map<std::string, double> rank_backward_execution_orders(
    Graph& g, const std::map<std::string, std::vector<std::string>>& orders) {
  std::map<std::string, size_t> peaks;
  size_t best_peak = std::numeric_limits<size_t>::max();

  for (const auto& [name, order] : orders) {
    if (order.empty()) {
      peaks[name] = std::numeric_limits<size_t>::max();
      continue;
    }
    Allocator allocator;
    GraphExecutor executor(g);

    executor.init_boundaries(&allocator);
    std::vector<std::string> fw_order = find_fw_naive_dfs_execution_order(g);
    for (const auto& op_id : fw_order) {
      OperationNode* op_node = nullptr;
      for (auto& [uuid, node] : g.op_nodes()) {
        if (node.uuid() == op_id) {
          op_node = &node;
          break;
        }
      }
      if (op_node) executor.run_op_node(op_node, &allocator);
    }

    size_t peak = 0;
    allocator.subscribe("rank", [&](size_t new_mem) {
      if (new_mem > peak) peak = new_mem;
    });
    executor.transition_to_backward(&allocator);
    for (const auto& op_id : order) {
      OperationNode* op_node = nullptr;
      for (auto& [uuid, node] : g.op_nodes()) {
        if (node.uuid() == op_id) {
          op_node = &node;
          break;
        }
      }
      if (op_node) executor.run_backward_op_node(op_node, &allocator);
    }
    allocator.unsubscribe("rank");
    peaks[name] = peak;
    if (peaks[name] < best_peak) {
      best_peak = peaks[name];
    }
  }

  std::map<std::string, double> efficiencies;
  for (const auto& [name, peak] : peaks) {
    if (peak == std::numeric_limits<size_t>::max()) {
      efficiencies[name] = 0.0;
    } else {
      efficiencies[name] = (static_cast<double>(best_peak) / static_cast<double>(peak)) * 100.0;
    }
  }
  return efficiencies;
}

std::map<std::string, double> rank_packing_efficiencies(
    Graph& g, const std::map<std::string, std::vector<std::string>>& orders, bool training) {
  std::map<std::string, double> efficiencies;

  for (const auto& [name, order] : orders) {
    if (order.empty()) {
      efficiencies[name] = 0.0;
      continue;
    }
    size_t packed_peak = compute_ffd_peak_memory(g, order, training);

    Allocator allocator;
    GraphExecutor executor(g);
    size_t idealized_peak = 0;
    allocator.subscribe("rank", [&](size_t new_mem) {
      if (new_mem > idealized_peak) idealized_peak = new_mem;
    });
    if (training) {
      executor.init_boundaries(&allocator);
      std::vector<std::string> fw_order = order;
      std::reverse(fw_order.begin(), fw_order.end());
      for (const auto& op_id : fw_order) {
        OperationNode* op_node = nullptr;
        for (auto& [uuid, node] : g.op_nodes()) {
          if (node.uuid() == op_id) {
            op_node = &node;
            break;
          }
        }
        executor.run_op_node(op_node, &allocator);
      }
      executor.transition_to_backward(&allocator);
    } else {
      executor.init_boundaries(&allocator);
    }
    for (const auto& op_id : order) {
      OperationNode* op_node = nullptr;
      for (auto& [uuid, node] : g.op_nodes()) {
        if (node.uuid() == op_id) {
          op_node = &node;
          break;
        }
      }
      if (op_node) {
        if (training)
          executor.run_backward_op_node(op_node, &allocator);
        else
          executor.run_op_node(op_node, &allocator, false);
      }
    }
    allocator.unsubscribe("rank");

    if (packed_peak == 0 || packed_peak == std::numeric_limits<size_t>::max()) {
      efficiencies[name] = 0.0;
    } else {
      efficiencies[name] =
          (static_cast<double>(idealized_peak) / static_cast<double>(packed_peak)) * 100.0;
    }
  }
  return efficiencies;
}

template <typename T>
bool contains(const std::vector<T>& vec, const T& val) {
  return std::find(vec.begin(), vec.end(), val) != vec.end();
}

void print_stats(std::map<std::string, std::vector<double>>& effs_by_name) {
  for (auto& [name, effs] : effs_by_name) {
    if (effs.empty()) return;
    std::sort(effs.begin(), effs.end());
    double sum = 0;
    for (double e : effs) sum += e;
    double avg = sum / effs.size();
    double worst = effs.front();
    double best = effs.back();
    double p25 = effs[effs.size() * 25 / 100];
    double p50 = effs[effs.size() * 50 / 100];
    double p75 = effs[effs.size() * 75 / 100];

    std::cout << "Order: " << name << "\n";
    std::cout << "  Average: " << std::fixed << std::setprecision(2) << avg << "%\n";
    std::cout << "  Worst:   " << std::fixed << std::setprecision(2) << worst << "%\n";
    std::cout << "  p25:     " << std::fixed << std::setprecision(2) << p25 << "%\n";
    std::cout << "  Median:  " << std::fixed << std::setprecision(2) << p50 << "%\n";
    std::cout << "  p75:     " << std::fixed << std::setprecision(2) << p75 << "%\n";
    std::cout << "  Best:    " << std::fixed << std::setprecision(2) << best << "%\n";
  }
}

void run_simulation_trials(int original_trials, const std::string& title,
                           const std::string& log_prefix, std::function<Graph()> graph_generator,
                           const std::vector<std::string>& to_checks) {
  std::map<std::string, std::vector<double>> fw_efficiencies;
  std::map<std::string, std::vector<double>> bw_efficiencies;
  std::map<std::string, std::vector<double>> fw_packing_efficiencies;
  std::map<std::string, std::vector<double>> fp_packing_efficiencies;

  int trials = original_trials;
  while (trials--) {
    Graph g = graph_generator();

    auto fw_best_order = find_fw_fork_join_execution_order(g);
    auto fw_macro_order = find_fw_macro_candidate_execution_order(g);
    auto fw_macro_v2_order = find_fw_macro_v2_candidate_execution_order(g);
    auto fw_dfs_order = find_fw_naive_dfs_execution_order(g);

    std::map<std::string, std::vector<std::string>> fw_orders = {{"BEST", fw_best_order},
                                                                 {"MACRO", fw_macro_order},
                                                                 {"MACRO_V2", fw_macro_v2_order},
                                                                 {"DFS", fw_dfs_order}};

    auto fw_effs = rank_execution_orders(g, fw_orders);

    auto bw_best_order = find_bw_fork_join_execution_order(g);
    auto bw_macro_order = find_bw_macro_candidate_execution_order(g);
    auto bw_macro_v2_order = find_bw_macro_v2_candidate_execution_order(g);
    auto bw_dfs_order = find_bw_naive_dfs_execution_order(g);
    std::map<std::string, std::vector<std::string>> bw_orders = {{"BEST", bw_best_order},
                                                                 {"MACRO", bw_macro_order},
                                                                 {"MACRO_V2", bw_macro_v2_order},
                                                                 {"DFS", bw_dfs_order}};
    auto bw_effs = rank_backward_execution_orders(g, bw_orders);
    auto fw_packing_effs = rank_packing_efficiencies(g, fw_orders, false);
    auto bw_packing_effs = rank_packing_efficiencies(g, bw_orders, true);

    for (const auto& [name, eff] : fw_effs) {
      fw_efficiencies[name].push_back(eff);
      fw_packing_efficiencies[name].push_back(fw_packing_effs[name]);
      if (contains(to_checks, name) && !near(eff, 100.0)) {
        save_graph_to_dot(g, "./logs/" + log_prefix + "_fw_graph.dot");
        std::ofstream log("./logs/" + log_prefix + "_fw_bad_macro.log", std::ios_base::app);
        log << "--- Trial Failure ---\n";
        log << name << " Efficiency: " << eff << "%\n";
        auto print_path = [&](const std::string& p_name, const std::vector<std::string>& path) {
          log << p_name << " Path: ";
          for (const auto& op : path) log << op << " ";
          log << "\n";
        };
        print_path("BEST", fw_best_order);
        print_path(name, fw_orders[name]);
        if (name == "MACRO_V2") {
          find_fw_macro_v2_candidate_execution_order(g, &log);
        } else {
          find_fw_macro_candidate_execution_order(g, &log);
        }
        log << "\n";
      }
    }

    for (const auto& [name, eff] : bw_effs) {
      bw_efficiencies[name].push_back(eff);
      fp_packing_efficiencies[name].push_back(bw_packing_effs[name]);
      if (contains(to_checks, name) && !near(eff, 100.0)) {
        save_graph_to_dot(g, "./logs/" + log_prefix + "_bw_graph.dot", true);
        std::ofstream log("./logs/" + log_prefix + "_bw_bad_macro.log", std::ios_base::app);
        log << "--- Trial Failure ---\n";
        log << name << " Efficiency: " << eff << "%\n";
        log << "BEST Efficiency: " << bw_effs["BEST"] << "%\n";
        log << "DFS Efficiency: " << bw_effs["DFS"] << "%\n";
        auto print_path = [&](const std::string& p_name, const std::vector<std::string>& path) {
          log << p_name << " Path: ";
          for (const auto& op : path) log << op << " ";
          log << "\n";
        };
        print_path("BEST", bw_best_order);
        print_path("DFS", bw_dfs_order);
        print_path(name, bw_orders[name]);
        if (name == "MACRO_V2") {
          find_bw_macro_v2_candidate_execution_order(g, &log);
        } else {
          find_bw_macro_candidate_execution_order(g, &log);
        }
        log << "\n";
      }
    }
  }

  std::cout << "=== " << title << " Forward Efficiency Overview (" << original_trials
            << " trials) ===\n";
  print_stats(fw_efficiencies);
  std::cout << "=== " << title << " Forward Packing Efficiency Overview (" << original_trials
            << " trials) ===\n";
  print_stats(fw_packing_efficiencies);

  std::cout << "=== " << title << " Backward Pass Efficiency Overview (" << original_trials
            << " trials) ===\n";
  print_stats(bw_efficiencies);
  std::cout << "=== " << title << " Full Pass Packing Efficiency Overview (" << original_trials
            << " trials) ===\n";
  print_stats(fp_packing_efficiencies);
}

int main() {
  srand(static_cast<unsigned int>(time(nullptr)));
  std::vector<std::string> to_checks = {"MACRO_V2"};

  run_simulation_trials(
      1, "Sample Failure", "sample_failure", []() { return sample_failure_graph(); }, to_checks);
  run_simulation_trials(
      1, "Sample Failure 2", "sample_failure_2", []() { return sample_failure_graph_2(); }, to_checks);

  int trials;
  std::cin >> trials;

  // run_simulation_trials(
  //     trials, "Independent Operators", "independent_operators",
  //     []() { return random_m_sequences_graph(10, 1); }, to_checks);
  // run_simulation_trials(
  //     trials, "Parallel Sequences", "parallel_sequences",
  //     []() { return random_m_sequences_graph(4, 4); }, to_checks);
  // run_simulation_trials(
  //     trials, "Join", "join", []() { return random_joining_graph(4); }, to_checks);
  // run_simulation_trials(
  //     trials, "Order-Invariant Branch", "order_invariant_branch",
  //     []() { return random_order_invariant_branching_graph(3); }, to_checks);
  // run_simulation_trials(
  //     trials, "Order-Invariant Fork Join", "order_invariant_fork_join",
  //     []() { return random_order_invariant_fork_join_graph(4); }, to_checks);
  run_simulation_trials(
      trials, "Order-Dependent Branch", "branch", []() { return random_branching_graph(3); },
      to_checks);
  // run_simulation_trials(
  //     trials, "Order-Dependent Fork Join", "fork_join", []() { return random_fork_join_graph(4);
  //     }, to_checks);

  return 0;
}
