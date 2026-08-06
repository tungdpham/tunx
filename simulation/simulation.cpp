#include <algorithm>
#include <bitset>
#include <cmath>
#include <fstream>
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

std::pair<std::map<std::string, std::set<std::string>>,
          std::map<std::string, std::set<std::string>>>
get_dependencies(Graph& graph) {
  std::map<std::string, std::string> tensor_producer;
  for (auto& [uuid, node] : graph.op_nodes()) {
    for (auto* t : node.outputs()) {
      tensor_producer[t->uuid()] = node.uuid();
    }
  }

  std::map<std::string, std::set<std::string>> deps;
  std::map<std::string, std::set<std::string>> dependents;
  for (auto& [uuid, node] : graph.op_nodes()) {
    std::string op_id = node.uuid();
    deps[op_id] = {};
    for (auto* t : node.inputs()) {
      auto it = tensor_producer.find(t->uuid());
      if (it != tensor_producer.end()) {
        deps[op_id].insert(it->second);
      }
    }
  }
  for (auto& [op_id, dep_set] : deps) {
    for (auto& dep : dep_set) {
      dependents[dep].insert(op_id);
    }
  }
  return {deps, dependents};
}

std::map<std::string, int> get_out_deg(Graph& graph) {
  std::map<std::string, int> out_deg;
  for (auto& [uuid, node] : graph.op_nodes()) {
    for (auto* t : node.inputs()) {
      out_deg[t->uuid()]++;
    }
  }
  for (auto* t : graph.outputs()) {
    out_deg[t->uuid()]++;
  }
  return out_deg;
}

std::vector<std::string> find_macro_candidate_execution_order(Graph& graph,
                                                              std::ostream* os = nullptr);

std::vector<std::string> find_macro_candidate_execution_order_v2(Graph& graph,
                                                                 std::ostream* os = nullptr);

std::vector<std::string> find_minimum_memory_execution_order(Graph& graph) {
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
  auto [deps, dependents] = get_dependencies(graph);

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
    auto heuristic_order = find_macro_candidate_execution_order(graph);
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

std::vector<std::string> find_macro_candidate_execution_order(Graph& graph, std::ostream* os) {
  std::vector<std::string> op_ids;
  for (auto& [uuid, node] : graph.op_nodes()) {
    op_ids.push_back(node.uuid());
  }

  auto out_deg = get_out_deg(graph);
  auto [deps, dependents] = get_dependencies(graph);

  struct MacroNode {
    std::string id;
    std::vector<std::string> ops;
    long long a;
    long long b;
  };

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

  auto compare = [](const MacroNode& i, const MacroNode& j) {
    long long cost_ij = std::max(i.a, i.b + j.a);
    long long cost_ji = std::max(j.a, j.b + i.a);
    if (cost_ij != cost_ji) return cost_ij < cost_ji;
    if (i.b != j.b) return i.b < j.b;
    return i.a < j.a;
  };

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
      if (macro_dependents[X].size() == 1) {
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
          best_X = "";
          break;
        }

        if (compare(macros[Y], macros[X])) {
          if (best_X == "" || compare(macros[best_X], macros[X])) {
            best_X = X;
          }
        }
      }
    }

    if (best_X != "") {
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

  auto pq_compare = [&macros, &compare](const std::string& a, const std::string& b) {
    return compare(macros[b], macros[a]);
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

std::vector<std::string> find_macro_candidate_execution_order_v2(Graph& graph, std::ostream* os) {
  std::vector<std::string> op_ids;
  for (auto& [uuid, node] : graph.op_nodes()) {
    op_ids.push_back(node.uuid());
  }

  auto out_deg = get_out_deg(graph);
  auto [deps, dependents] = get_dependencies(graph);

  struct MacroNode {
    std::string id;
    std::vector<std::string> ops;
    long long a;
    long long b;
  };

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

  auto compare = [](const MacroNode& i, const MacroNode& j) {
    long long cost_ij = std::max(i.a, i.b + j.a);
    long long cost_ji = std::max(j.a, j.b + i.a);
    if (cost_ij != cost_ji) return cost_ij < cost_ji;
    if (i.b != j.b) return i.b < j.b;
    return i.a < j.a;
  };

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
      if (macro_dependents[X].size() == 1) {
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
          best_X = "";
          break;
        }

        if (compare(macros[Y], macros[X])) {
          if (best_X == "" || compare(macros[best_X], macros[X])) {
            best_X = X;
          }
        }
      }
    }

    if (best_X != "") {
      std::string XY_id = "macro_v2_" + std::to_string(next_macro_id++);
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

  // Phase 2: Sibling Family Contraction
  // Idea: The problem with macro_v1 is that it doesn't account for shared input activations since
  // linear macro contraction was originally based on static cost assumption. However, for reference
  // counted tensors, like in instances where C branch into A, B, D is a successor of A, E is
  // successor of B, contracting A and D into a single macro would be a bad choice since the
  // reference count of executing other nodes could change cost of A. To avoid bad contraction, we
  // add a check if A is shares its inputs with any other macros/operators. And the idea for
  // considering contracting A with either D or B or both is called family grouping. Say A have
  // lower cost than B, we compare macro {A, B} with family interleaved macros like {A, some
  // sequences after A but possibly empty..., B, some sequences after B but possibly empty...}. It
  // is intuitive that executing B first is suboptimal but this remains my intuition for now. I will
  // add a test case later to verify. But it is also possible that family macros are more expensive
  // than {A, some sequences after A, some other nodes from other branches that totally not a
  // successor of C} if freeing the input tensor created by C doesn't yield much prospect. The idea
  // is that after the initial linear contraction, the sequences after A, B is monotonically
  // increasing, so not much lookahead is needed. Also we have to do the family macro in reverse
  // topological order.
  std::vector<std::string> shared_tensors;
  for (auto* t : graph.inputs()) {
    if (out_deg[t->uuid()] > 1) shared_tensors.push_back(t->uuid());
  }
  for (const auto& op_id : topo_order) {
    auto& node = graph.get_op(op_id);
    for (auto* t : node.outputs()) {
      if (out_deg[t->uuid()] > 1) shared_tensors.push_back(t->uuid());
    }
  }

  std::reverse(shared_tensors.begin(), shared_tensors.end());

  auto evaluate_full = [&](const std::vector<std::string>& ops) -> std::pair<long long, long long> {
    long long current_b = 0;
    long long peak_a = 0;
    std::map<std::string, int> local_consumptions;
    for (const auto& op_id : ops) {
      auto& node = graph.get_op(op_id);
      long long all_outputs = 0;
      for (auto* t : node.outputs()) all_outputs += t->size();
      long long workspace = node.workspace_req();

      peak_a = std::max(peak_a, current_b + all_outputs + workspace);
      current_b += all_outputs;

      for (auto* t : node.inputs()) {
        local_consumptions[t->uuid()]++;
        if (local_consumptions[t->uuid()] == out_deg[t->uuid()]) {
          current_b -= t->size();
        }
      }
    }
    return {peak_a, current_b};
  };

  auto evaluate_schedule = [&](const std::vector<std::string>& macro_seq) -> long long {
    std::vector<std::string> ops;
    for (const auto& m_id : macro_seq) {
      for (const auto& op : macros[m_id].ops) ops.push_back(op);
    }
    return evaluate_full(ops).first;
  };

  auto pq_compare = [&macros, &compare](const std::string& a, const std::string& b) {
    return compare(macros[b], macros[a]);
  };

  for (const auto& T_id : shared_tensors) {
    std::set<std::string> F_set;
    for (const auto& [m_id, macro] : macros) {
      for (const auto& op_id : macro.ops) {
        auto& node = graph.get_op(op_id);
        bool consumes_T = false;
        for (auto* t : node.inputs()) {
          if (t->uuid() == T_id) consumes_T = true;
        }
        if (consumes_T) {
          F_set.insert(m_id);
          break;
        }
      }
    }

    if (F_set.size() <= 1) continue;

    bool is_sibling = true;
    for (const auto& start_m : F_set) {
      std::queue<std::string> bfs_q;
      std::set<std::string> visited;
      bfs_q.push(start_m);
      visited.insert(start_m);
      while (!bfs_q.empty()) {
        auto curr = bfs_q.front();
        bfs_q.pop();
        for (const auto& dep : macro_dependents[curr]) {
          if (F_set.count(dep)) {
            is_sibling = false;
            break;
          }
          if (!visited.count(dep)) {
            visited.insert(dep);
            bfs_q.push(dep);
          }
        }
        if (!is_sibling) break;
      }
      if (!is_sibling) break;
    }
    if (!is_sibling) continue;

    std::set<std::string> succ_F_set;
    for (const auto& m : F_set) {
      for (const auto& dep : macro_dependents[m]) {
        succ_F_set.insert(dep);
      }
    }

    std::set<std::string> U = F_set;
    for (const auto& s : succ_F_set) U.insert(s);

    std::vector<std::string> sched1;
    {
      std::vector<std::string> F_vec(F_set.begin(), F_set.end());
      auto sort_compare = [&macros, &compare](const std::string& a, const std::string& b) {
        return compare(macros[a], macros[b]);
      };
      std::sort(F_vec.begin(), F_vec.end(), sort_compare);
      sched1.insert(sched1.end(), F_vec.begin(), F_vec.end());

      std::map<std::string, int> in_deg;
      for (const auto& s : succ_F_set) in_deg[s] = 0;
      for (const auto& s : succ_F_set) {
        for (const auto& dep : macro_dependents[s]) {
          if (succ_F_set.count(dep)) in_deg[dep]++;
        }
      }
      std::priority_queue<std::string, std::vector<std::string>, decltype(pq_compare)> ready(
          pq_compare);
      for (const auto& s : succ_F_set) {
        if (in_deg[s] == 0) ready.push(s);
      }
      while (!ready.empty()) {
        auto curr = ready.top();
        ready.pop();
        sched1.push_back(curr);
        for (const auto& dep : macro_dependents[curr]) {
          if (succ_F_set.count(dep)) {
            if (--in_deg[dep] == 0) ready.push(dep);
          }
        }
      }
    }

    std::vector<std::string> sched2;
    {
      std::map<std::string, int> in_deg;
      for (const auto& u : U) in_deg[u] = 0;
      for (const auto& u : U) {
        for (const auto& dep : macro_dependents[u]) {
          if (U.count(dep)) in_deg[dep]++;
        }
      }
      std::priority_queue<std::string, std::vector<std::string>, decltype(pq_compare)> ready(
          pq_compare);
      for (const auto& u : U) {
        if (in_deg[u] == 0) ready.push(u);
      }
      while (!ready.empty()) {
        auto curr = ready.top();
        ready.pop();
        sched2.push_back(curr);
        for (const auto& dep : macro_dependents[curr]) {
          if (U.count(dep)) {
            if (--in_deg[dep] == 0) ready.push(dep);
          }
        }
      }
    }

    long long peak1 = evaluate_schedule(sched1);
    long long peak2 = evaluate_schedule(sched2);

    auto creates_cycle = [&](const std::set<std::string>& C) {
      std::map<std::string, std::set<std::string>> adj;
      std::map<std::string, int> in_deg;
      in_deg["C_NODE"] = 0;
      for (const auto& [u, deps] : macro_dependents) {
        if (!C.count(u)) in_deg[u] = 0;
        for (const auto& v : deps) {
          if (!C.count(v)) in_deg[v] = 0;
        }
      }
      for (const auto& [u, deps] : macro_dependents) {
        std::string src = C.count(u) ? "C_NODE" : u;
        for (const auto& v : deps) {
          std::string dst = C.count(v) ? "C_NODE" : v;
          if (src != dst) {
            if (adj[src].insert(dst).second) {
              in_deg[dst]++;
            }
          }
        }
      }
      std::queue<std::string> q;
      for (const auto& [u, deg] : in_deg) {
        if (deg == 0) q.push(u);
      }
      int count = 0;
      while (!q.empty()) {
        std::string u = q.front();
        q.pop();
        count++;
        for (const auto& v : adj[u]) {
          if (--in_deg[v] == 0) q.push(v);
        }
      }
      return count < (int)in_deg.size();
    };

    std::vector<std::string> chosen_sched;
    std::set<std::string> contracted_macros;
    if (peak1 < peak2 || creates_cycle(U)) {
      for (const auto& m : sched1) {
        if (F_set.count(m)) chosen_sched.push_back(m);
      }
      contracted_macros = F_set;
    } else {
      chosen_sched = sched2;
      contracted_macros = U;
    }

    std::string new_macro_id = "macro_v2_" + std::to_string(next_macro_id++);
    MacroNode new_macro;
    new_macro.id = new_macro_id;
    for (const auto& m : chosen_sched) {
      for (const auto& op : macros[m].ops) new_macro.ops.push_back(op);
    }

    auto [a_val, b_val] = evaluate_full(new_macro.ops);
    new_macro.a = a_val;
    new_macro.b = b_val;
    macros[new_macro_id] = new_macro;

    macro_deps[new_macro_id] = {};
    macro_dependents[new_macro_id] = {};
    for (const auto& m : contracted_macros) {
      for (const auto& dep : macro_deps[m]) {
        if (!contracted_macros.count(dep)) {
          macro_deps[new_macro_id].insert(dep);
          macro_dependents[dep].erase(m);
          macro_dependents[dep].insert(new_macro_id);
        }
      }
      for (const auto& dep : macro_dependents[m]) {
        if (!contracted_macros.count(dep)) {
          macro_dependents[new_macro_id].insert(dep);
          macro_deps[dep].erase(m);
          macro_deps[dep].insert(new_macro_id);
        }
      }
    }

    for (const auto& m : contracted_macros) {
      macros.erase(m);
      macro_deps.erase(m);
      macro_dependents.erase(m);
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

int main() {
  srand(static_cast<unsigned int>(time(nullptr)));

  int trials;
  std::cin >> trials;
  int original_trials = trials;
  std::map<std::string, std::vector<double>> all_sample_efficiencies;
  std::map<std::string, std::vector<double>> all_branch_efficiencies;
  std::map<std::string, std::vector<double>> all_join_efficiencies;

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

  trials = original_trials;
  while (trials--) {
    Graph g = sample_branch_graph();

    auto best_order = find_minimum_memory_execution_order(g);
    auto macro_order = find_macro_candidate_execution_order(g);
    auto macro_v2_order = find_macro_candidate_execution_order_v2(g);

    auto effs = rank_execution_orders(
        g, {{"BEST", best_order}, {"MACRO", macro_order}, {"MACRO_V2", macro_v2_order}});

    for (const auto& [name, eff] : effs) {
      all_sample_efficiencies[name].push_back(eff);
      if ((name == "MACRO" || name == "MACRO_V2") && !near(eff, 100.0)) {
        save_graph_to_dot(g, "sample_graph.dot");
        std::ofstream log("sample_bad_macro.log", std::ios_base::app);
        log << "--- Trial Failure ---\n";
        log << name << " Efficiency: " << eff << "%\n";
        auto print_path = [&](const std::string& p_name, const std::vector<std::string>& path) {
          log << p_name << " Path: ";
          for (const auto& op : path) log << op << " ";
          log << "\n";
        };
        print_path("BEST", best_order);
        print_path(name, name == "MACRO" ? macro_order : macro_v2_order);
        if (name == "MACRO") {
          find_macro_candidate_execution_order(g, &log);
        } else {
          find_macro_candidate_execution_order_v2(g, &log);
        }
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
    auto macro_v2_order = find_macro_candidate_execution_order_v2(g);

    auto effs = rank_execution_orders(
        g, {{"BEST", best_order}, {"MACRO", macro_order}, {"MACRO_V2", macro_v2_order}});

    for (const auto& [name, eff] : effs) {
      all_join_efficiencies[name].push_back(eff);
      if ((name == "MACRO" || name == "MACRO_V2") && !near(eff, 100.0)) {
        save_graph_to_dot(g, "join_graph.dot");
        std::ofstream log("join_bad_macro.log", std::ios_base::app);
        log << "--- Trial Failure ---\n";
        log << name << " Efficiency: " << eff << "%\n";
        auto print_path = [&](const std::string& p_name, const std::vector<std::string>& path) {
          log << p_name << " Path: ";
          for (const auto& op : path) log << op << " ";
          log << "\n";
        };
        print_path("BEST", best_order);
        print_path(name, name == "MACRO" ? macro_order : macro_v2_order);
        if (name == "MACRO") {
          find_macro_candidate_execution_order(g, &log);
        } else {
          find_macro_candidate_execution_order_v2(g, &log);
        }
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
    auto macro_v2_order = find_macro_candidate_execution_order_v2(g);

    auto effs = rank_execution_orders(
        g, {{"BEST", best_order}, {"MACRO", macro_order}, {"MACRO_V2", macro_v2_order}});

    for (const auto& [name, eff] : effs) {
      all_branch_efficiencies[name].push_back(eff);
      if ((name == "MACRO" || name == "MACRO_V2") && !near(eff, 100.0)) {
        save_graph_to_dot(g, "graph.dot");
        std::ofstream log("branch_bad_macro.log", std::ios_base::app);
        log << "--- Trial Failure ---\n";
        log << name << " Efficiency: " << eff << "%\n";
        auto print_path = [&](const std::string& p_name, const std::vector<std::string>& path) {
          log << p_name << " Path: ";
          for (const auto& op : path) log << op << " ";
          log << "\n";
        };
        print_path("BEST", best_order);
        print_path(name, name == "MACRO" ? macro_order : macro_v2_order);
        if (name == "MACRO") {
          find_macro_candidate_execution_order(g, &log);
        } else {
          find_macro_candidate_execution_order_v2(g, &log);
        }
        log << "\n";
      }
    }
  }

  std::cout << "=== Branch Efficiency Overview (" << original_trials << " trials) ===\n";
  for (auto name : {"BEST", "MACRO", "MACRO_V2"}) {
    std::cout << name << " Order:\n";
    print_stats(all_branch_efficiencies[name]);
  }

  return 0;
}