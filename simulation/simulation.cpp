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

std::vector<std::string> find_macro_candidate_execution_order(Graph& graph,
                                                              std::ostream* os = nullptr) {
  MacroSolver solver(graph, os);
  return solver.find_order();
}

std::vector<std::string> find_fork_join_optimal_execution_order(Graph& graph);

std::vector<std::string> find_diamond_execution_order(Graph& graph, size_t max_states = 1000000);

std::vector<std::string> find_fork_join_optimal_execution_order(Graph& graph) {
  if (graph.inputs().size() != 1 || graph.outputs().size() != 1) return {};

  std::map<std::string, std::vector<std::string>> consumers;
  for (const auto& [op_id, op] : graph.op_nodes()) {
    for (auto* input : op.inputs()) consumers[input->uuid()].push_back(op_id);
  }

  const auto* shared_input = graph.inputs().front();
  const auto& root_ops = consumers[shared_input->uuid()];
  if (root_ops.size() < 2) return {};

  std::vector<std::vector<std::string>> branches;
  std::string join_op_id;
  std::set<std::string> branch_ops;

  for (const auto& root_op_id : root_ops) {
    const auto* current_op = &graph.get_op(root_op_id);
    if (current_op->inputs().size() != 1 || current_op->inputs().front() != shared_input ||
        current_op->outputs().size() != 1) {
      return {};
    }

    std::vector<std::string> branch;
    while (true) {
      if (!branch_ops.insert(current_op->uuid()).second) return {};
      branch.push_back(current_op->uuid());

      const auto& next_ops = consumers[current_op->outputs().front()->uuid()];
      if (next_ops.size() != 1) return {};
      const auto* next_op = &graph.get_op(next_ops.front());
      if (next_op->inputs().size() > 1) {
        if (join_op_id.empty()) join_op_id = next_op->uuid();
        if (join_op_id != next_op->uuid()) return {};
        break;
      }
      if (next_op->inputs().size() != 1 || next_op->outputs().size() != 1) return {};
      current_op = next_op;
    }
    branches.push_back(std::move(branch));
  }

  if (join_op_id.empty()) return {};
  const auto& join_op = graph.get_op(join_op_id);
  if (join_op.outputs().size() != 1 || join_op.outputs().front() != graph.outputs().front() ||
      join_op.inputs().size() != branches.size() ||
      branch_ops.size() + 1 != graph.op_nodes().size()) {
    return {};
  }

  std::set<std::string> branch_tails;
  for (const auto& branch : branches) {
    const auto& tail = graph.get_op(branch.back());
    branch_tails.insert(tail.outputs().front()->uuid());
  }
  for (auto* input : join_op.inputs()) {
    if (!branch_tails.erase(input->uuid())) return {};
  }
  if (!branch_tails.empty()) return {};

  struct Schedule {
    size_t peak;
    std::vector<std::string> order;
  };

  Allocator allocator;
  GraphExecutor executor(graph);
  executor.init_boundaries(&allocator);
  std::map<std::vector<size_t>, Schedule> memo;

  std::function<Schedule(std::vector<size_t>&)> solve = [&](std::vector<size_t>& completed) {
    if (const auto it = memo.find(completed); it != memo.end()) return it->second;

    bool all_branches_complete = true;
    for (size_t branch_index = 0; branch_index < branches.size(); ++branch_index) {
      if (completed[branch_index] != branches[branch_index].size()) {
        all_branches_complete = false;
        break;
      }
    }

    if (all_branches_complete) {
      size_t peak = allocator.allocated();
      allocator.subscribe("fork_join_dp", [&](size_t memory) { peak = std::max(peak, memory); });
      executor.run_op_node(&join_op, &allocator);
      allocator.unsubscribe("fork_join_dp");
      executor.undo_run_op_node(&join_op, &allocator);
      return memo.emplace(completed, Schedule{peak, {join_op_id}}).first->second;
    }

    Schedule best{std::numeric_limits<size_t>::max(), {}};
    for (size_t branch_index = 0; branch_index < branches.size(); ++branch_index) {
      if (completed[branch_index] == branches[branch_index].size()) continue;

      const auto& op_id = branches[branch_index][completed[branch_index]++];
      const auto& op = graph.get_op(op_id);
      size_t op_peak = allocator.allocated();
      allocator.subscribe("fork_join_dp",
                          [&](size_t memory) { op_peak = std::max(op_peak, memory); });
      executor.run_op_node(&op, &allocator);
      allocator.unsubscribe("fork_join_dp");

      Schedule suffix = solve(completed);
      const size_t candidate_peak = std::max(op_peak, suffix.peak);
      if (candidate_peak < best.peak) {
        best.peak = candidate_peak;
        best.order = {op_id};
        best.order.insert(best.order.end(), suffix.order.begin(), suffix.order.end());
      }

      executor.undo_run_op_node(&op, &allocator);
      completed[branch_index]--;
    }
    return memo.emplace(completed, std::move(best)).first->second;
  };

  std::vector<size_t> completed(branches.size(), 0);
  return solve(completed).order;
}

std::vector<std::string> find_diamond_execution_order(Graph& graph, size_t max_states) {
  std::vector<std::string> op_ids;
  std::map<std::string, int> op_index;
  for (const auto& [uuid, op] : graph.op_nodes()) {
    if (op_ids.size() == 256) return find_macro_candidate_execution_order(graph);
    op_index[op.uuid()] = static_cast<int>(op_ids.size());
    op_ids.push_back(op.uuid());
  }

  auto [dependencies, dependents] = get_dependencies(graph);
  struct Schedule {
    size_t peak;
    std::vector<std::string> order;
  };

  Allocator allocator;
  GraphExecutor executor(graph);
  size_t initial_peak = 0;
  allocator.subscribe("diamond_dp_init",
                      [&](size_t memory) { initial_peak = std::max(initial_peak, memory); });
  executor.init_boundaries(&allocator);
  allocator.unsubscribe("diamond_dp_init");

  std::unordered_map<std::bitset<256>, Schedule> memo;
  bool state_limit_reached = false;
  std::function<Schedule(std::bitset<256>&)> solve = [&](std::bitset<256>& completed) {
    if (const auto it = memo.find(completed); it != memo.end()) return it->second;
    if (memo.size() >= max_states) {
      state_limit_reached = true;
      return Schedule{std::numeric_limits<size_t>::max(), {}};
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
      allocator.subscribe("diamond_dp",
                          [&](size_t memory) { op_peak = std::max(op_peak, memory); });
      executor.run_op_node(&op, &allocator);
      allocator.unsubscribe("diamond_dp");
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
  return result.order;
}

std::vector<std::string> find_minimum_memory_execution_order(Graph& graph) {
  return find_diamond_execution_order(graph);
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
    double p10 = effs[effs.size() * 10 / 100];
    double p50 = effs[effs.size() * 50 / 100];
    double p90 = effs[effs.size() * 90 / 100];

    std::cout << "Order: " << name << "\n";
    std::cout << "  Average: " << std::fixed << std::setprecision(2) << avg << "%\n";
    std::cout << "  Worst:   " << std::fixed << std::setprecision(2) << worst << "%\n";
    std::cout << "  p10:     " << std::fixed << std::setprecision(2) << p10 << "%\n";
    std::cout << "  Median:  " << std::fixed << std::setprecision(2) << p50 << "%\n";
    std::cout << "  p90:     " << std::fixed << std::setprecision(2) << p90 << "%\n";
    std::cout << "  Best:    " << std::fixed << std::setprecision(2) << best << "%\n\n";
  }
}

int main() {
  srand(static_cast<unsigned int>(time(nullptr)));

  {
    Graph g = tunx_v1_graph();
    auto best_order = find_minimum_memory_execution_order(g);
    auto macro_order = find_macro_candidate_execution_order(g);
    auto effs = rank_execution_orders(g, {{"BEST", best_order}, {"MACRO", macro_order}});

    std::cout << "=== Tunx V1 Graph Efficiency ===\n";
    for (const auto& [name, eff] : effs) {
      std::cout << "  " << name << " Efficiency: " << std::fixed << std::setprecision(2) << eff
                << "%\n";
      if (name == "MACRO") {
        save_graph_to_dot(g, "./logs/tunx_v1_graph.dot");
        std::ofstream log("./logs/tunx_v1_bad_macro.log", std::ios_base::app);
        log << "--- Tunx V1 Graph Failure ---\n";
        log << name << " Efficiency: " << eff << "%\n";
        auto print_path = [&](const std::string& p_name, const std::vector<std::string>& path) {
          log << p_name << " Path: ";
          for (const auto& op : path) log << op << " ";
          log << "\n";
        };
        print_path("BEST", best_order);
        print_path(name, macro_order);
        find_macro_candidate_execution_order(g, &log);
        log << "\n";
      }
    }
  }
  int trials;
  std::cin >> trials;
  int original_trials = trials;
  std::map<std::string, std::vector<double>> all_sample_efficiencies;
  std::map<std::string, std::vector<double>> all_branch_efficiencies;
  std::map<std::string, std::vector<double>> all_static_branch_efficiencies;
  std::map<std::string, std::vector<double>> all_join_efficiencies;
  std::map<std::string, std::vector<double>> all_diamond_efficiencies;
  std::map<std::string, std::vector<double>> all_static_diamond_efficiencies;

  std::vector<std::string> to_checks = {"MACRO"};

  trials = original_trials;
  while (trials--) {
    Graph g = sample_branch_graph();

    auto best_order = find_minimum_memory_execution_order(g);
    auto macro_order = find_macro_candidate_execution_order(g);

    auto effs = rank_execution_orders(g, {
                                             {"BEST", best_order},
                                             {"MACRO", macro_order},
                                         });

    for (const auto& [name, eff] : effs) {
      all_sample_efficiencies[name].push_back(eff);
      if (contains(to_checks, name) && !near(eff, 100.0)) {
        save_graph_to_dot(g, "./logs/sample_graph.dot");
        std::ofstream log("./logs/sample_bad_macro.log", std::ios_base::app);
        log << "--- Trial Failure ---\n";
        log << name << " Efficiency: " << eff << "%\n";
        auto print_path = [&](const std::string& p_name, const std::vector<std::string>& path) {
          log << p_name << " Path: ";
          for (const auto& op : path) log << op << " ";
          log << "\n";
        };
        print_path("BEST", best_order);
        print_path(name, macro_order);
        find_macro_candidate_execution_order(g, &log);
        log << "\n";
      }
    }
  }
  std::cout << "=== Sample Efficiency Overview (" << original_trials << " trials) ===\n";
  print_stats(all_sample_efficiencies);

  trials = original_trials;
  while (trials--) {
    Graph g = random_joining_graph(4);

    auto best_order = find_minimum_memory_execution_order(g);
    auto macro_order = find_macro_candidate_execution_order(g);

    auto effs = rank_execution_orders(g, {{"BEST", best_order}, {"MACRO", macro_order}});

    for (const auto& [name, eff] : effs) {
      all_join_efficiencies[name].push_back(eff);
      if (contains(to_checks, name) && !near(eff, 100.0)) {
        save_graph_to_dot(g, "./logs/join_graph.dot");
        std::ofstream log("./logs/join_bad_macro.log", std::ios_base::app);
        log << "--- Trial Failure ---\n";
        log << name << " Efficiency: " << eff << "%\n";
        auto print_path = [&](const std::string& p_name, const std::vector<std::string>& path) {
          log << p_name << " Path: ";
          for (const auto& op : path) log << op << " ";
          log << "\n";
        };
        print_path("BEST", best_order);
        print_path(name, macro_order);
        find_macro_candidate_execution_order(g, &log);
        log << "\n";
      }
    }
  }

  std::cout << "=== Join Efficiency Overview (" << original_trials << " trials) ===\n";
  print_stats(all_join_efficiencies);

  trials = original_trials;
  while (trials--) {
    Graph g = random_branching_graph(3);

    auto best_order = find_minimum_memory_execution_order(g);
    auto macro_order = find_macro_candidate_execution_order(g);

    auto effs = rank_execution_orders(g, {{"BEST", best_order}, {"MACRO", macro_order}});

    for (const auto& [name, eff] : effs) {
      all_branch_efficiencies[name].push_back(eff);
      if (contains(to_checks, name) && !near(eff, 100.0)) {
        save_graph_to_dot(g, "./logs/graph.dot");
        std::ofstream log("./logs/branch_bad_macro.log", std::ios_base::app);
        log << "--- Trial Failure ---\n";
        log << name << " Efficiency: " << eff << "%\n";
        auto print_path = [&](const std::string& p_name, const std::vector<std::string>& path) {
          log << p_name << " Path: ";
          for (const auto& op : path) log << op << " ";
          log << "\n";
        };
        print_path("BEST", best_order);
        print_path(name, macro_order);
        find_macro_candidate_execution_order(g, &log);
        log << "\n";
      }
    }
  }

  std::cout << "=== Branch Efficiency Overview (" << original_trials << " trials) ===\n";
  print_stats(all_branch_efficiencies);

  trials = original_trials;
  while (trials--) {
    Graph g = random_static_branching_graph(3);

    auto best_order = find_minimum_memory_execution_order(g);
    auto macro_order = find_macro_candidate_execution_order(g);

    auto effs = rank_execution_orders(g, {{"BEST", best_order}, {"MACRO", macro_order}});

    for (const auto& [name, eff] : effs) {
      all_static_branch_efficiencies[name].push_back(eff);
      if (contains(to_checks, name) && !near(eff, 100.0)) {
        save_graph_to_dot(g, "./logs/static_branch_graph.dot");
        std::ofstream log("./logs/static_branch_bad_macro.log", std::ios_base::app);
        log << "--- Trial Failure ---\n";
        log << name << " Efficiency: " << eff << "%\n";
        auto print_path = [&](const std::string& p_name, const std::vector<std::string>& path) {
          log << p_name << " Path: ";
          for (const auto& op : path) log << op << " ";
          log << "\n";
        };
        print_path("BEST", best_order);
        print_path(name, macro_order);
        find_macro_candidate_execution_order(g, &log);
        log << "\n";
      }
    }
  }

  std::cout << "=== Static Branch Efficiency Overview (" << original_trials << " trials) ===\n";
  print_stats(all_static_branch_efficiencies);

  trials = original_trials;
  while (trials--) {
    Graph g = random_diamond_graph(4);

    auto best_order = find_minimum_memory_execution_order(g);
    auto macro_order = find_macro_candidate_execution_order(g);

    auto effs = rank_execution_orders(g, {{"MACRO", macro_order}, {"BEST", best_order}});

    for (const auto& [name, eff] : effs) {
      all_diamond_efficiencies[name].push_back(eff);
      if (contains(to_checks, name) && !near(eff, 100.0)) {
        save_graph_to_dot(g, "./logs/diamond_graph.dot");
        std::ofstream log("./logs/diamond_bad_macro.log", std::ios_base::app);
        log << "--- Trial Failure ---\n";
        log << name << " Efficiency: " << eff << "%\n";
        auto print_path = [&](const std::string& p_name, const std::vector<std::string>& path) {
          log << p_name << " Path: ";
          for (const auto& op : path) log << op << " ";
          log << "\n";
        };
        print_path("BEST", best_order);
        print_path(name, macro_order);
        find_macro_candidate_execution_order(g, &log);
        log << "\n";
      }
    }
  }

  std::cout << "=== Diamond Efficiency Overview (" << original_trials << " trials) ===\n";
  print_stats(all_diamond_efficiencies);

  trials = original_trials;
  while (trials--) {
    Graph g = random_static_diamond_graph(4);

    auto best_order = find_minimum_memory_execution_order(g);
    auto macro_order = find_macro_candidate_execution_order(g);

    auto effs = rank_execution_orders(g, {{"BEST", best_order}, {"MACRO", macro_order}});

    for (const auto& [name, eff] : effs) {
      all_static_diamond_efficiencies[name].push_back(eff);
      if (contains(to_checks, name) && !near(eff, 100.0)) {
        save_graph_to_dot(g, "./logs/static_diamond_graph.dot");
        std::ofstream log("./logs/static_diamond_bad_macro.log", std::ios_base::app);
        log << "--- Trial Failure ---\n";
        log << name << " Efficiency: " << eff << "%\n";
        auto print_path = [&](const std::string& p_name, const std::vector<std::string>& path) {
          log << p_name << " Path: ";
          for (const auto& op : path) log << op << " ";
          log << "\n";
        };
        print_path("BEST", best_order);
        print_path(name, macro_order);
        find_macro_candidate_execution_order(g, &log);
        log << "\n";
      }
    }
  }

  std::cout << "=== Static Diamond Efficiency Overview (" << original_trials << " trials) ===\n";
  print_stats(all_static_diamond_efficiencies);

  return 0;
}