#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "allocator.h"
#include "dp_solver.h"
#include "graph.h"
#include "graph_executor.h"
#include "graph_generator.h"
#include "memory_packer.h"
#include "vision_graph_generator.h"

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

void run_forward_trials(int trials, const std::string& title, const std::string& log_prefix,
                        std::function<Graph()> graph_generator,
                        const std::vector<std::string>& to_checks) {
  std::map<std::string, std::vector<double>> fw_efficiencies;
  std::map<std::string, std::vector<double>> fw_packing_efficiencies;

  int original_trials = trials;
  while (trials--) {
    Graph g = graph_generator();

    bool oracle_complete = false;
    auto fw_best_order = find_fw_fork_join_execution_order(g, 1000000, &oracle_complete);
    auto fw_macro_order = find_fw_macro_candidate_execution_order(g);
    auto fw_ranked_order = find_fw_ranked_execution_order(g);
    auto fw_linear_order = find_fw_linear_execution_order(g);
    auto fw_branching_order = find_fw_branching_execution_order(g);
    auto fw_joining_order = find_fw_joining_execution_order(g);
    auto fw_flat_bj_order = find_fw_flat_bj_execution_order(g);
    auto fw_full_order = find_fw_full_execution_order(g);
    auto fw_dfs_order = find_fw_naive_dfs_execution_order(g);

    std::map<std::string, std::vector<std::string>> fw_orders = {
        {"BEST", fw_best_order},
        {"MACRO", fw_macro_order},
        {"BRANCHING", fw_branching_order},
        {"JOINING", fw_joining_order},
        {"LINEAR", fw_linear_order},
        {"RANKED", fw_ranked_order},
        {"FLAT_BJ", fw_flat_bj_order},
        {"FULL", fw_full_order},
        {"DFS", fw_dfs_order},
    };

    auto fw_effs = rank_execution_orders(g, fw_orders);
    auto fw_packing_effs = rank_packing_efficiencies(g, fw_orders, false);

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
        if (name == "MACRO") {
          find_fw_macro_candidate_execution_order(g, &log);
        }
        log << "\n";
      }
    }
  }

  std::cout << "=== " << title << " Forward Efficiency Overview (" << original_trials
            << " trials) ===\n";
  print_stats(fw_efficiencies);
}

void run_backward_trials(int trials, const std::string& title, const std::string& log_prefix,
                         std::function<Graph()> graph_generator,
                         const std::vector<std::string>& to_checks) {
  std::map<std::string, std::vector<double>> bw_efficiencies;
  std::map<std::string, std::vector<double>> fp_packing_efficiencies;

  int original_trials = trials;
  while (trials--) {
    Graph g = graph_generator();

    auto bw_best_order = find_bw_fork_join_execution_order(g);
    auto bw_macro_order = find_bw_macro_candidate_execution_order(g);
    auto bw_ranked_order = find_bw_ranked_execution_order(g);
    auto bw_linear_order = find_bw_linear_execution_order(g);
    auto bw_branching_order = find_bw_branching_execution_order(g);
    auto bw_joining_order = find_bw_joining_execution_order(g);
    auto bw_dfs_order = find_bw_naive_dfs_execution_order(g);
    std::map<std::string, std::vector<std::string>> bw_orders = {
        {"BEST", bw_best_order},
        {"MACRO", bw_macro_order},
        {"BRANCHING", bw_branching_order},
        {"JOINING", bw_joining_order},
        {"LINEAR", bw_linear_order},
        {"RANKED", bw_ranked_order},
        {"DFS", bw_dfs_order},
    };
    auto bw_effs = rank_backward_execution_orders(g, bw_orders);
    auto bw_packing_effs = rank_packing_efficiencies(g, bw_orders, true);

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
        if (name == "MACRO") {
          find_bw_macro_candidate_execution_order(g, &log);
        }
        log << "\n";
      }
    }
  }

  std::cout << "=== " << title << " Backward Pass Efficiency Overview (" << original_trials
            << " trials) ===\n";
  print_stats(bw_efficiencies);
}

int main() {
  srand(static_cast<unsigned int>(time(nullptr)));
  std::vector<std::string> to_checks = {"MACRO"};

  int trials = 1000;

  // Stress tests (Forward)
  run_forward_trials(
      trials, "Independent Operators", "independent_operators",
      []() { return random_m_sequences_graph(10, 1); }, to_checks);
  run_forward_trials(
      trials, "Parallel Sequences", "parallel_sequences",
      []() { return random_m_sequences_graph(4, 4); }, to_checks);
  run_forward_trials(trials, "Join", "join", []() { return random_joining_graph(4); }, to_checks);
  run_forward_trials(
      trials, "Order-Invariant Branch", "order_invariant_branch",
      []() { return random_order_invariant_branching_graph(3); }, to_checks);
  run_forward_trials(
      trials, "Order-Invariant Fork Join", "order_invariant_fork_join",
      []() { return random_order_invariant_fork_join_graph(4); }, to_checks);
  run_forward_trials(
      trials, "Order-Dependent Branch", "branch", []() { return random_branching_graph(3); },
      to_checks);
  run_forward_trials(
      trials, "Order-Dependent Fork Join", "fork_join", []() { return random_fork_join_graph(4); },
      to_checks);

  // Stress tests (Backward)
  run_backward_trials(
      trials, "Independent Operators", "independent_operators",
      []() { return random_m_sequences_graph(10, 1); }, to_checks);
  run_backward_trials(
      trials, "Parallel Sequences", "parallel_sequences",
      []() { return random_m_sequences_graph(4, 4); }, to_checks);
  run_backward_trials(trials, "Join", "join", []() { return random_joining_graph(4); }, to_checks);
  run_backward_trials(
      trials, "Order-Invariant Branch", "order_invariant_branch",
      []() { return random_order_invariant_branching_graph(3); }, to_checks);
  run_backward_trials(
      trials, "Order-Invariant Fork Join", "order_invariant_fork_join",
      []() { return random_order_invariant_fork_join_graph(4); }, to_checks);
  run_backward_trials(
      trials, "Order-Dependent Branch", "branch", []() { return random_branching_graph(3); },
      to_checks);
  run_backward_trials(
      trials, "Order-Dependent Fork Join", "fork_join", []() { return random_fork_join_graph(4); },
      to_checks);

  return 0;
}
