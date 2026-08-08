#include <algorithm>
#include <bitset>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "allocator.h"
#include "graph.h"
#include "graph_executor.h"
#include "graph_generator.h"

std::vector<std::string> find_macro_candidate_execution_order(Graph& graph,
                                                              std::ostream* os = nullptr);

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

struct MacroNode {
  std::string id;
  std::vector<std::string> ops;
  long long a;
  long long b;
};

std::pair<long, long> get_rank(const MacroNode& m) {
  if (m.b < 0) {
    return {0, m.a};
  } else {
    return {1, m.b - m.a};
  }
}

bool compare_macros(const MacroNode& m1, const MacroNode& m2) {
  auto rank1 = get_rank(m1);
  auto rank2 = get_rank(m2);
  if (rank1.first != rank2.first) return rank1.first < rank2.first;
  return rank1.second <= rank2.second;
}

bool operator<=(const MacroNode& m1, const MacroNode& m2) { return compare_macros(m1, m2); }

std::vector<std::string> find_macro_candidate_execution_order(Graph& graph, std::ostream* os) {
  std::vector<std::string> op_ids;
  for (auto& [uuid, node] : graph.op_nodes()) {
    op_ids.push_back(node.uuid());
  }

  auto out_deg = get_out_deg(graph);
  auto [deps, dependents] = get_dependencies(graph);

  std::map<std::string, MacroNode> macros;
  std::map<std::string, std::set<std::string>> macro_deps;
  std::map<std::string, std::set<std::string>> macro_dependents;

  for (auto& id : op_ids) {
    auto& node = graph.get_op(id);
    long long all_outputs = 0;
    for (auto* t : node.outputs()) {
      all_outputs += t->size();
    }

    long long workspace = node.workspace_req();
    long long total_memory_for_execution = all_outputs + workspace;

    long long memory_generate = all_outputs;
    long long memory_consumes = 0;
    for (auto* t : node.inputs()) {
      if (out_deg[t->uuid()] == 1) {
        memory_consumes += t->size();
      }
    }

    MacroNode mn;
    mn.id = id;
    mn.ops = {id};
    mn.a = total_memory_for_execution;
    mn.b = memory_generate - memory_consumes;
    macros[id] = mn;

    macro_deps[id] = deps[id];
    macro_dependents[id] = dependents[id];
  }

  std::vector<std::string> topo_order;
  std::map<std::string, int> in_deg_topo;
  for (auto& id : op_ids) in_deg_topo[id] = macro_deps[id].size();
  std::queue<std::string> q;
  for (auto& id : op_ids)
    if (in_deg_topo[id] == 0) q.push(id);
  while (!q.empty()) {
    std::string u = q.front();
    q.pop();
    topo_order.push_back(u);
    for (auto& v : macro_dependents[u]) {
      if (--in_deg_topo[v] == 0) q.push(v);
    }
  }

  int next_macro_id = 0;
  std::deque<std::string> dq;
  for (const auto& u : topo_order) {
    dq.push_back(u);
  }

  while (!dq.empty()) {
    std::string Y = dq.front();
    dq.pop_front();

    std::string best_X = "";
    for (auto& X : macro_deps[Y]) {
      if (macro_dependents[X].size() != 1) continue;

      // drop if share input tensor with any peers
      bool has_peers = false;
      for (const auto& op_id : macros[X].ops) {
        const auto& op_node = graph.get_op(op_id);
        for (auto* t : op_node.inputs()) {
          if (out_deg[t->uuid()] > 1) {
            int internal_consumers = 0;
            for (const auto& inner_op_id : macros[X].ops) {
              const auto& inner_op = graph.get_op(inner_op_id);
              for (auto* inner_t : inner_op.inputs()) {
                if (inner_t->uuid() == t->uuid()) {
                  internal_consumers++;
                }
              }
            }
            if (internal_consumers < out_deg[t->uuid()]) {
              has_peers = true;
              break;
            }
          }
        }
        if (has_peers) break;
      }
      if (has_peers) {
        continue;
      }

      if (macros[Y] <= macros[X]) {
        if (best_X == "" || macros[best_X] <= macros[X]) {
          best_X = X;
        }
      }
    }

    if (best_X != "") {
      if (os) {
        *os << "Merging forward macro " << macros[best_X].id << " [";
        for (auto op : macros[best_X].ops) {
          *os << op << " ";
        }
        *os << ", a:" << macros[best_X].a << ", b:" << macros[best_X].b << " ] with macro "
            << macros[Y].id << " [";
        for (auto op : macros[Y].ops) {
          *os << op << " ";
        }
        *os << ", a:" << macros[Y].a << ", b:" << macros[Y].b << " ] into macro "
            << "macro_" + std::to_string(next_macro_id) << std::endl;
      }
      std::string XY_id = "macro_" + std::to_string(next_macro_id++);
      MacroNode XY;
      XY.id = XY_id;
      XY.ops = macros[best_X].ops;
      XY.ops.insert(XY.ops.end(), macros[Y].ops.begin(), macros[Y].ops.end());
      XY.a = std::max(macros[best_X].a, macros[best_X].b + macros[Y].a);
      XY.b = macros[best_X].b + macros[Y].b;
      macros[XY_id] = XY;

      macro_deps[XY_id] = macro_deps[best_X];
      for (auto& dep : macro_deps[Y]) {
        if (dep != best_X) macro_deps[XY_id].insert(dep);
      }

      macro_dependents[XY_id] = macro_dependents[best_X];
      macro_dependents[XY_id].erase(Y);
      for (auto& child : macro_dependents[Y]) {
        macro_dependents[XY_id].insert(child);
      }

      for (auto& p : macro_deps[XY_id]) {
        macro_dependents[p].erase(best_X);
        macro_dependents[p].erase(Y);
        macro_dependents[p].insert(XY_id);
      }
      for (auto& child : macro_dependents[XY_id]) {
        macro_deps[child].erase(best_X);
        macro_deps[child].erase(Y);
        macro_deps[child].insert(XY_id);
      }

      macros.erase(best_X);
      macros.erase(Y);
      macro_deps.erase(best_X);
      macro_deps.erase(Y);
      macro_dependents.erase(best_X);
      macro_dependents.erase(Y);

      dq.push_front(XY_id);
    }
  }

  // sort new contracted graph in reverse topological order
  std::map<std::string, int> in_deg;
  for (const auto& [m_id, _] : macros) {
    in_deg[m_id] = macro_deps[m_id].size();
  }

  std::queue<std::string> top_q;
  for (const auto& [m_id, deg] : in_deg) {
    if (deg == 0) top_q.push(m_id);
  }

  std::vector<std::string> reverse_topo;
  while (!top_q.empty()) {
    std::string u = top_q.front();
    top_q.pop();
    reverse_topo.push_back(u);
    for (const auto& v : macro_dependents[u]) {
      if (--in_deg[v] == 0) top_q.push(v);
    }
  }
  std::reverse(reverse_topo.begin(), reverse_topo.end());

  for (auto& macro_id : reverse_topo) {
    dq.push_back(macro_id);
  }

  while (!dq.empty()) {
    std::string Y = dq.front();
    dq.pop_front();

    std::string best_Z = "";
    for (auto& Z : macro_dependents[Y]) {
      if (macro_deps[Z].size() != 1) continue;

      if (macros[Z] <= macros[Y]) {
        if (best_Z == "" || macros[Z] <= macros[best_Z]) {
          best_Z = Z;
        }
      }
    }

    if (best_Z != "") {
      if (os) {
        *os << "Merging backward macro " << macros[Y].id << " [";
        for (auto op : macros[Y].ops) {
          *os << op << " ";
        }
        *os << ", a: " << macros[Y].a << ", b: " << macros[Y].b << "] with macro "
            << macros[best_Z].id << " [";
        for (auto op : macros[best_Z].ops) {
          *os << op << " ";
        }
        *os << ", a: " << macros[best_Z].a << ", b: " << macros[best_Z].b << "] into macro "
            << "macro_" + std::to_string(next_macro_id) << std::endl;
      }
      std::string YZ_id = "macro_" + std::to_string(next_macro_id++);
      MacroNode YZ;
      YZ.id = YZ_id;
      YZ.ops = macros[Y].ops;
      YZ.ops.insert(YZ.ops.end(), macros[best_Z].ops.begin(), macros[best_Z].ops.end());
      YZ.a = std::max(macros[Y].a, macros[Y].b + macros[best_Z].a);
      YZ.b = macros[Y].b + macros[best_Z].b;
      macros[YZ_id] = YZ;

      macro_deps[YZ_id] = macro_deps[Y];
      for (auto& dep : macro_deps[best_Z]) {
        if (dep != Y) macro_deps[YZ_id].insert(dep);
      }

      macro_dependents[YZ_id] = macro_dependents[Y];
      macro_dependents[YZ_id].erase(best_Z);
      for (auto& child : macro_dependents[best_Z]) {
        macro_dependents[YZ_id].insert(child);
      }

      for (auto& p : macro_deps[YZ_id]) {
        macro_dependents[p].erase(Y);
        macro_dependents[p].erase(best_Z);
        macro_dependents[p].insert(YZ_id);
      }
      for (auto& child : macro_dependents[YZ_id]) {
        macro_deps[child].erase(Y);
        macro_deps[child].erase(best_Z);
        macro_deps[child].insert(YZ_id);
      }

      macros.erase(Y);
      macros.erase(best_Z);
      macro_deps.erase(Y);
      macro_deps.erase(best_Z);
      macro_dependents.erase(Y);
      macro_dependents.erase(best_Z);

      dq.push_front(YZ_id);
      continue;
    }
  }

  if (os) {
    *os << "All macros generated" << std::endl;
    for (auto& [id, macro] : macros) {
      *os << id << ": ";
      for (auto& op : macro.ops) {
        *os << op << " ";
      }
      *os << " -> [ " << macro.a << " " << macro.b << " ]" << std::endl;
    }
  }

  std::vector<std::string> final_order;
  std::set<std::string> executed_macros;

  auto pq_compare = [&macros](const std::string& a, const std::string& b) {
    return macros[b] <= macros[a];
  };

  std::priority_queue<std::string, std::vector<std::string>, decltype(pq_compare)> ready_macros(
      pq_compare);

  for (auto& [id, deps_set] : macro_deps) {
    if (deps_set.empty()) {
      ready_macros.push(id);
    }
  }

  while (final_order.size() < op_ids.size()) {
    if (ready_macros.empty()) {
      throw std::runtime_error("Graph has a cycle or unresolved dependencies.");
    }

    std::string best_macro = ready_macros.top();
    ready_macros.pop();

    executed_macros.insert(best_macro);

    for (auto& op : macros[best_macro].ops) {
      final_order.push_back(op);
    }

    for (auto& child : macro_dependents[best_macro]) {
      bool ready = true;
      for (auto& parent : macro_deps[child]) {
        if (!executed_macros.count(parent)) {
          ready = false;
          break;
        }
      }
      if (ready) {
        ready_macros.push(child);
      }
    }
  }

  return final_order;
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

void save_graph_to_dot(Graph& graph, const std::string& filename) {
  std::ofstream out(filename);
  out << "digraph G {\n";
  for (auto& [uuid, act] : graph.act_nodes()) {
    out << "  \"" << act.uuid() << "\" [shape=ellipse, label=\"" << act.uuid() << "\\n("
        << act.size() << ")\"];\n";
  }
  for (auto& [uuid, op] : graph.op_nodes()) {
    out << "  \"" << op.uuid() << "\" [shape=box, label=\"" << op.uuid() << "\\n("
        << op.workspace_req() << ")\"];\n";
    for (auto* in : op.inputs()) {
      out << "  \"" << in->uuid() << "\" -> \"" << op.uuid() << "\";\n";
    }
    for (auto* out_act : op.outputs()) {
      out << "  \"" << op.uuid() << "\" -> \"" << out_act->uuid() << "\";\n";
    }
  }
  out << "}\n";
}

Graph sample_branch_graph() {
  Graph g;
  ActivationNode* input = g.add_act("input", 10240);

  size_t branch_length = 5;
  ActivationNode* prev = input;
  ActivationNode* b1_tail = nullptr;
  // branch 1
  for (size_t i = 0; i < branch_length; i++) {
    b1_tail = g.add_act("b1_act" + std::to_string(i), random_act_size());
    g.add_op("b1_op" + std::to_string(i), random_ws_size(), {prev}, {b1_tail});
    prev = b1_tail;
  }

  prev = input;
  ActivationNode* b2_tail = nullptr;
  // branch 2
  for (size_t i = 0; i < branch_length; i++) {
    b2_tail = g.add_act("b2_act" + std::to_string(i), random_act_size());
    g.add_op("b2_op" + std::to_string(i), random_ws_size(), {prev}, {b2_tail});
    prev = b2_tail;
  }

  prev = input;
  ActivationNode* b3_tail = nullptr;
  // branch 3
  for (size_t i = 0; i < branch_length; i++) {
    b3_tail = g.add_act("b3_act" + std::to_string(i), random_act_size());
    g.add_op("b3_op" + std::to_string(i), random_ws_size(), {prev}, {b3_tail});
    prev = b3_tail;
  }

  // join
  ActivationNode* output = g.add_act("output", 10240);
  g.add_op("join_op", random_ws_size(), {b1_tail, b2_tail, b3_tail}, {output});

  g.set_inputs({input});
  g.set_outputs({output});

  return g;
}

template <typename T>
bool contains(const std::vector<T>& vec, const T& val) {
  return std::find(vec.begin(), vec.end(), val) != vec.end();
}

int main() {
  srand(static_cast<unsigned int>(time(nullptr)));

  int trials;
  std::cin >> trials;
  int original_trials = trials;
  std::map<std::string, std::vector<double>> all_sample_efficiencies;
  std::map<std::string, std::vector<double>> all_branch_efficiencies;
  std::map<std::string, std::vector<double>> all_static_branch_efficiencies;
  std::map<std::string, std::vector<double>> all_join_efficiencies;
  std::map<std::string, std::vector<double>> all_diamond_efficiencies;
  std::map<std::string, std::vector<double>> all_static_diamond_efficiencies;

  auto print_stats = [](std::vector<double>& effs) {
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

    std::cout << "  Average: " << std::fixed << std::setprecision(2) << avg << "%\n";
    std::cout << "  Worst:   " << std::fixed << std::setprecision(2) << worst << "%\n";
    std::cout << "  p10:     " << std::fixed << std::setprecision(2) << p10 << "%\n";
    std::cout << "  Median:  " << std::fixed << std::setprecision(2) << p50 << "%\n";
    std::cout << "  p90:     " << std::fixed << std::setprecision(2) << p90 << "%\n";
    std::cout << "  Best:    " << std::fixed << std::setprecision(2) << best << "%\n\n";
  };

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
  for (auto name : {"BEST", "MACRO", "MACRO_V2"}) {
    std::cout << name << " Order:\n";
    print_stats(all_sample_efficiencies[name]);
  }

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
  for (auto name : {"BEST", "MACRO", "MACRO_V2"}) {
    std::cout << name << " Order:\n";
    print_stats(all_join_efficiencies[name]);
  }

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
  for (auto name : {"BEST", "MACRO", "MACRO_V2"}) {
    std::cout << name << " Order:\n";
    print_stats(all_branch_efficiencies[name]);
  }

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
  for (auto name : {"BEST", "MACRO", "MACRO_V2"}) {
    std::cout << name << " Order:\n";
    print_stats(all_static_branch_efficiencies[name]);
  }

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
  for (auto name : {"BEST", "MACRO", "MACRO_V2"}) {
    std::cout << name << " Order:\n";
    print_stats(all_diamond_efficiencies[name]);
  }

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
  for (auto name : {"BEST", "MACRO", "MACRO_V2"}) {
    std::cout << name << " Order:\n";
    print_stats(all_static_diamond_efficiencies[name]);
  }

  return 0;
}