#pragma once

#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <ostream>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "graph.h"

struct MacroNode {
  std::string id;
  std::vector<std::string> ops;
  long long a;
  long long b;
};

inline std::pair<long, long> get_rank(const MacroNode& m) {
  if (m.b < 0) {
    return {0, m.a};
  } else {
    return {1, m.b - m.a};
  }
}

inline bool compare_macros(const MacroNode& m1, const MacroNode& m2) {
  auto rank1 = get_rank(m1);
  auto rank2 = get_rank(m2);
  if (rank1.first != rank2.first) return rank1.first < rank2.first;
  return rank1.second < rank2.second;
}

inline bool operator<(const MacroNode& m1, const MacroNode& m2) { return compare_macros(m1, m2); }

struct SolverOptions {
  bool enable_linear = true;
  bool enable_branching = true;
  bool enable_joining = true;
};

class MacroSolver {
private:
  // core
  Graph& graph_;
  std::ostream* os_;
  SolverOptions options_;

  // generated
  std::map<std::string, MacroNode> macros_;
  std::map<std::string, std::set<std::string>> macro_deps_;
  std::map<std::string, std::set<std::string>> macro_dependents_;

  bool has_peers_forward(const std::string& macro_id, const std::map<std::string, int>& out_deg,
                         const std::set<std::string>& cached_tensors) const {
    for (const auto& op_id : macros_.at(macro_id).ops) {
      const auto& op_node = graph_.get_op(op_id);
      for (auto* tensor : op_node.inputs()) {
        if (out_deg.at(tensor->uuid()) <= 1) continue;
        if (cached_tensors.count(tensor->uuid())) continue;

        int internal_consumers = 0;
        for (const auto& inner_op_id : macros_.at(macro_id).ops) {
          const auto& inner_op = graph_.get_op(inner_op_id);
          for (auto* inner_tensor : inner_op.inputs()) {
            if (inner_tensor->uuid() == tensor->uuid()) ++internal_consumers;
          }
        }
        if (internal_consumers < out_deg.at(tensor->uuid())) return true;
      }
    }
    return false;
  }

  bool has_peers_backward(const std::string& macro_id, const std::map<std::string, int>& out_deg,
                          const std::set<std::string>& cached_tensors) const {
    for (const auto& op_id : macros_.at(macro_id).ops) {
      const auto& op_node = graph_.get_op(op_id);
      for (auto* tensor : op_node.outputs()) {
        if (out_deg.at(tensor->uuid()) <= 1) continue;
        if (cached_tensors.count(tensor->uuid())) continue;

        int internal_consumers = 0;
        for (const auto& inner_op_id : macros_.at(macro_id).ops) {
          const auto& inner_op = graph_.get_op(inner_op_id);
          for (auto* inner_tensor : inner_op.outputs()) {
            if (inner_tensor->uuid() == tensor->uuid()) ++internal_consumers;
          }
        }
        if (internal_consumers < out_deg.at(tensor->uuid())) return true;
      }
    }
    return false;
  }

  std::string merge_macros(const std::string& parent, const std::string& child, int& next_macro_id,
                           const char* reason) {
    const std::string merged_id = "macro_" + std::to_string(next_macro_id++);
    MacroNode merged;
    merged.id = merged_id;
    merged.ops = macros_.at(parent).ops;
    merged.ops.insert(merged.ops.end(), macros_.at(child).ops.begin(), macros_.at(child).ops.end());
    merged.a = std::max(macros_.at(parent).a, macros_.at(parent).b + macros_.at(child).a);
    merged.b = macros_.at(parent).b + macros_.at(child).b;

    if (os_) {
      auto print_macro = [this](const MacroNode& macro) {
        *os_ << macro.id << " [";
        for (const auto& op : macro.ops) *os_ << op << " ";
        *os_ << ", a: " << macro.a << ", b: " << macro.b << "]";
      };
      *os_ << "Merging " << reason << " ";
      print_macro(macros_.at(parent));
      *os_ << " with ";
      print_macro(macros_.at(child));
      *os_ << " into ";
      print_macro(merged);
      *os_ << std::endl;
    }

    std::set<std::string> merged_deps = macro_deps_.at(parent);
    for (const auto& dependency : macro_deps_.at(child)) {
      if (dependency != parent) merged_deps.insert(dependency);
    }

    std::set<std::string> merged_dependents = macro_dependents_.at(parent);
    merged_dependents.erase(child);
    merged_dependents.insert(macro_dependents_.at(child).begin(),
                             macro_dependents_.at(child).end());

    macros_[merged_id] = std::move(merged);
    macro_deps_[merged_id] = std::move(merged_deps);
    macro_dependents_[merged_id] = std::move(merged_dependents);

    for (const auto& dependency : macro_deps_.at(merged_id)) {
      macro_dependents_.at(dependency).erase(parent);
      macro_dependents_.at(dependency).erase(child);
      macro_dependents_.at(dependency).insert(merged_id);
    }
    for (const auto& dependent : macro_dependents_.at(merged_id)) {
      macro_deps_.at(dependent).erase(parent);
      macro_deps_.at(dependent).erase(child);
      macro_deps_.at(dependent).insert(merged_id);
    }

    macros_.erase(parent);
    macros_.erase(child);
    macro_deps_.erase(parent);
    macro_deps_.erase(child);
    macro_dependents_.erase(parent);
    macro_dependents_.erase(child);
    return merged_id;
  }

  std::map<std::string, int> ancestor_distances(const std::string& macro_id) const {
    std::map<std::string, int> distances;
    distances[macro_id] = 0;
    
    std::map<std::string, int> in_deg;
    std::queue<std::string> q;
    q.push(macro_id);
    std::set<std::string> visited = {macro_id};
    
    while (!q.empty()) {
      std::string u = q.front();
      q.pop();
      for (const auto& parent : macro_deps_.at(u)) {
        in_deg[parent]++;
        if (visited.insert(parent).second) {
          q.push(parent);
        }
      }
    }
    
    q.push(macro_id);
    while (!q.empty()) {
      std::string u = q.front();
      q.pop();
      for (const auto& parent : macro_deps_.at(u)) {
        distances[parent] = std::max(distances[parent], distances[u] + 1);
        if (--in_deg[parent] == 0) {
          q.push(parent);
        }
      }
    }
    return distances;
  }

  std::string nearest_common_branching_ancestor(const std::string& first,
                                                const std::string& second) const {
    const auto first_distances = ancestor_distances(first);
    const auto second_distances = ancestor_distances(second);
    std::string nearest;
    std::pair<int, int> best_distance;
    for (const auto& [ancestor, first_distance] : first_distances) {
      const auto second_it = second_distances.find(ancestor);
      if (second_it == second_distances.end() || macro_dependents_.at(ancestor).size() < 2)
        continue;

      const std::pair<int, int> distance = {std::max(first_distance, second_it->second),
                                            first_distance + second_it->second};
      if (nearest.empty() || distance < best_distance) {
        nearest = ancestor;
        best_distance = distance;
      }
    }
    return nearest;
  }

  std::string merge_upward_recursively(std::string macro_id, int& next_macro_id) {
    while (macro_deps_.at(macro_id).size() == 1) {
      const std::string parent = *macro_deps_.at(macro_id).begin();
      if (macro_dependents_.at(parent).size() != 1 ||
          !(macros_.at(macro_id) < macros_.at(parent))) {
        break;
      }
      macro_id = merge_macros(parent, macro_id, next_macro_id, "upward");
    }
    return macro_id;
  }

  std::string merge_branch_recursively(std::string ancestor, int& next_macro_id) {
    bool merged_branch = false;
    while (true) {
      std::string best_child;
      for (const auto& child : macro_dependents_.at(ancestor)) {
        if (macro_deps_.at(child).size() != 1 || !(macros_.at(child) < macros_.at(ancestor)))
          continue;
        if (best_child.empty() || macros_.at(child) < macros_.at(best_child)) best_child = child;
      }
      if (best_child.empty()) {
        return merged_branch ? merge_upward_recursively(ancestor, next_macro_id) : ancestor;
      }
      ancestor = merge_macros(ancestor, best_child, next_macro_id, "branch");
      merged_branch = true;
    }
  }

  std::string prepare_join_branches(const std::string& join, int& next_macro_id) {
    const std::vector<std::string> parents(macro_deps_.at(join).begin(),
                                           macro_deps_.at(join).end());
    std::set<std::string> ancestors;
    for (size_t first = 0; first < parents.size(); ++first) {
      for (size_t second = first + 1; second < parents.size(); ++second) {
        const std::string ancestor =
            nearest_common_branching_ancestor(parents[first], parents[second]);
        if (!ancestor.empty()) ancestors.insert(ancestor);
      }
    }

    const auto distances_from_join = ancestor_distances(join);
    std::vector<std::string> ordered_ancestors(ancestors.begin(), ancestors.end());
    std::sort(ordered_ancestors.begin(), ordered_ancestors.end(),
              [&distances_from_join](const auto& left, const auto& right) {
                return distances_from_join.at(left) < distances_from_join.at(right);
              });
    for (const auto& ancestor : ordered_ancestors) {
      if (!macros_.contains(ancestor)) continue;
      const std::string merged_ancestor = merge_branch_recursively(ancestor, next_macro_id);
      if (!macros_.contains(join)) return merged_ancestor;
    }
    return join;
  }

  std::string merge_join_parent(const std::string& join, const std::map<std::string, int>& out_deg,
                                const std::set<std::string>& cached_tensors, int& next_macro_id) {
    std::string worst_parent;
    for (const auto& parent : macro_deps_.at(join)) {
      if (macro_dependents_.at(parent).size() != 1 ||
          has_peers_forward(parent, out_deg, cached_tensors) ||
          !(macros_.at(join) < macros_.at(parent))) {
        continue;
      }
      if (worst_parent.empty() || macros_.at(worst_parent) < macros_.at(parent)) {
        worst_parent = parent;
      }
    }
    return worst_parent.empty() ? join : merge_macros(worst_parent, join, next_macro_id, "join");
  }

  std::string merge_join_parent_backward(const std::string& join,
                                         const std::map<std::string, int>& in_deg,
                                         const std::set<std::string>& cached_tensors,
                                         int& next_macro_id) {
    std::string worst_parent;
    for (const auto& parent : macro_deps_.at(join)) {
      if (macro_dependents_.at(parent).size() != 1 ||
          has_peers_backward(parent, in_deg, cached_tensors) ||
          !(macros_.at(join) < macros_.at(parent))) {
        continue;
      }
      if (worst_parent.empty() || macros_.at(worst_parent) < macros_.at(parent)) {
        worst_parent = parent;
      }
    }
    return worst_parent.empty() ? join : merge_macros(worst_parent, join, next_macro_id, "join");
  }

public:
  MacroSolver(Graph& graph, std::ostream* os = nullptr, SolverOptions options = {})
      : graph_(graph),
        os_(os),
        options_(options) {}

  std::vector<std::string> find_forward_order() {
    std::vector<std::string> op_ids;
    for (auto& [uuid, node] : graph_.op_nodes()) {
      op_ids.push_back(node.uuid());
    }

    auto out_deg = get_out_deg(graph_);
    auto deps_and_dependents = get_op_dependencies(graph_);
    auto deps = deps_and_dependents.first;
    auto dependents = deps_and_dependents.second;

    std::set<std::string> cached_tensors;
    for (auto& [uuid, node] : graph_.op_nodes()) {
      for (auto* t : node.cache()) {
        cached_tensors.insert(t->uuid());
      }
    }

    macros_.clear();
    macro_deps_.clear();
    macro_dependents_.clear();
    for (auto& [uuid, node] : graph_.op_nodes()) {
      for (auto* t : node.cache()) {
        cached_tensors.insert(t->uuid());
      }
    }

    for (auto& id : op_ids) {
      auto& node = graph_.get_op(id);
      long long all_outputs = 0;
      for (auto* t : node.outputs()) {
        all_outputs += t->size();
      }

      long long workspace = node.workspace_req();
      long long total_memory_for_execution = all_outputs + workspace + node.residual_mem();

      long long memory_generate = all_outputs + static_cast<long long>(node.residual_mem());
      long long memory_consumes = 0;
      for (auto* t : node.inputs()) {
        if (out_deg[t->uuid()] == 1 &&
            std::find(node.cache().begin(), node.cache().end(), t) == node.cache().end()) {
          memory_consumes += t->size();
        }
      }

      MacroNode mn;
      mn.id = id;
      mn.ops = {id};
      mn.a = total_memory_for_execution;
      mn.b = memory_generate - memory_consumes;
      macros_[id] = mn;

      macro_deps_[id] = deps[id];
      macro_dependents_[id] = dependents[id];
    }

    std::vector<std::string> topo_order;
    std::map<std::string, int> in_deg_topo;
    for (auto& id : op_ids) in_deg_topo[id] = macro_deps_[id].size();
    std::queue<std::string> q;
    for (auto& id : op_ids)
      if (in_deg_topo[id] == 0) q.push(id);
    while (!q.empty()) {
      std::string u = q.front();
      q.pop();
      topo_order.push_back(u);
      for (auto& v : macro_dependents_[u]) {
        if (--in_deg_topo[v] == 0) q.push(v);
      }
    }

    const std::string virtual_join_id = "__virtual_join__";
    std::vector<std::string> terminal_macros;
    for (const auto& [macro_id, macro] : macros_) {
      if (macro_dependents_.at(macro_id).empty()) terminal_macros.push_back(macro_id);
    }
    if (terminal_macros.size() > 1) {
      macros_[virtual_join_id] = {virtual_join_id, {}, 0, std::numeric_limits<long long>::max()};
      macro_deps_[virtual_join_id] = {};
      macro_dependents_[virtual_join_id] = {};
      for (const auto& terminal : terminal_macros) {
        macro_deps_[virtual_join_id].insert(terminal);
        macro_dependents_[terminal].insert(virtual_join_id);
      }
    }

    int next_macro_id = 0;
    std::deque<std::string> pending(topo_order.begin(), topo_order.end());
    while (!pending.empty()) {
      const std::string current = pending.front();
      pending.pop_front();
      if (!macros_.contains(current)) continue;

      if (options_.enable_linear) {
        if (macro_deps_.at(current).size() == 1) {
          const std::string parent = *macro_deps_.at(current).begin();
          if (macro_dependents_.at(parent).size() == 1 &&
              macros_.at(current) < macros_.at(parent)) {
            pending.push_front(merge_macros(parent, current, next_macro_id, "linear"));
            continue;
          }
        }
      }

      if (macro_deps_.at(current).size() > 1) {
        std::string prepared = current;
        if (options_.enable_branching) {
          prepared = prepare_join_branches(current, next_macro_id);
          if (prepared != current) {
            pending.push_front(prepared);
            continue;
          }
        }

        if (options_.enable_joining) {
          const std::string merged =
              merge_join_parent(prepared, out_deg, cached_tensors, next_macro_id);
          if (merged != prepared) pending.push_front(merged);
        }
      }
    }
    if (macros_.contains(virtual_join_id)) {
      if (options_.enable_branching) {
        prepare_join_branches(virtual_join_id, next_macro_id);
      }
      for (const auto& terminal : macro_deps_.at(virtual_join_id)) {
        macro_dependents_.at(terminal).erase(virtual_join_id);
      }
      macros_.erase(virtual_join_id);
      macro_deps_.erase(virtual_join_id);
      macro_dependents_.erase(virtual_join_id);
    }

    if (os_) {
      *os_ << "All macros generated" << std::endl;
      for (auto& [id, macro] : macros_) {
        *os_ << id << ": ";
        for (auto& op : macro.ops) {
          *os_ << op << " ";
        }
        *os_ << " -> [ " << macro.a << " " << macro.b << " ]" << std::endl;
      }
    }

    std::vector<std::string> final_order;
    std::set<std::string> executed_macros;

    auto pq_compare = [this](const std::string& a, const std::string& b) {
      return macros_[b] < macros_[a];
    };

    std::priority_queue<std::string, std::vector<std::string>, decltype(pq_compare)> ready_macros(
        pq_compare);

    for (auto& [id, deps_set] : macro_deps_) {
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

      for (auto& op : macros_[best_macro].ops) {
        final_order.push_back(op);
      }

      for (auto& child : macro_dependents_[best_macro]) {
        bool ready = true;
        for (auto& parent : macro_deps_[child]) {
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

  std::vector<std::string> find_backward_order() {
    std::vector<std::string> op_ids;
    for (auto& [uuid, node] : graph_.op_nodes()) {
      op_ids.push_back(node.uuid());
    }

    std::map<std::string, int> in_deg;
    for (auto& [uuid, node] : graph_.op_nodes()) {
      for (auto* t : node.outputs()) {
        in_deg[t->uuid()]++;
      }
    }
    for (auto* t : graph_.inputs()) {
      in_deg[t->uuid()]++;
    }

    auto deps_and_dependents = get_op_dependencies(graph_);
    auto deps = deps_and_dependents.second;
    auto dependents = deps_and_dependents.first;

    std::set<std::string> cached_tensors;
    std::map<std::string, int> cache_deg;
    for (auto& [uuid, node] : graph_.op_nodes()) {
      for (auto* t : node.cache()) {
        cached_tensors.insert(t->uuid());
        cache_deg[t->uuid()]++;
      }
    }

    macros_.clear();
    macro_deps_.clear();
    macro_dependents_.clear();

    for (auto& id : op_ids) {
      auto& node = graph_.get_op(id);
      long long all_inputs = 0;
      for (auto* t : node.inputs()) {
        all_inputs += t->size();
      }

      long long workspace = node.workspace_req();
      long long total_memory_for_execution = all_inputs + workspace;

      long long memory_generate = all_inputs;
      long long memory_consumes = node.residual_mem();
      for (auto* t : node.outputs()) {
        if (in_deg[t->uuid()] == 1) {
          memory_consumes += t->size();
        }
      }
      for (auto* t : node.cache()) {
        if (cache_deg[t->uuid()] == 1) {
          memory_consumes += t->size();
        }
      }

      MacroNode mn;
      mn.id = id;
      mn.ops = {id};
      mn.a = total_memory_for_execution;
      mn.b = memory_generate - memory_consumes;
      macros_[id] = mn;

      macro_deps_[id] = deps[id];
      macro_dependents_[id] = dependents[id];
    }

    std::vector<std::string> topo_order;
    std::map<std::string, int> in_deg_topo;
    for (auto& id : op_ids) in_deg_topo[id] = macro_deps_[id].size();
    std::queue<std::string> q;
    for (auto& id : op_ids)
      if (in_deg_topo[id] == 0) q.push(id);
    while (!q.empty()) {
      std::string u = q.front();
      q.pop();
      topo_order.push_back(u);
      for (auto& v : macro_dependents_[u]) {
        if (--in_deg_topo[v] == 0) q.push(v);
      }
    }

    const std::string virtual_join_id = "__virtual_join__";
    std::vector<std::string> terminal_macros;
    for (const auto& [macro_id, macro] : macros_) {
      if (macro_dependents_.at(macro_id).empty()) terminal_macros.push_back(macro_id);
    }
    int next_macro_id = 0;
    if (terminal_macros.size() > 1) {
      macros_[virtual_join_id] = {virtual_join_id, {}, 0, std::numeric_limits<long long>::max()};
      macro_deps_[virtual_join_id] = {};
      macro_dependents_[virtual_join_id] = {};
      for (const auto& terminal : terminal_macros) {
        macro_deps_[virtual_join_id].insert(terminal);
        macro_dependents_[terminal].insert(virtual_join_id);
      }
    }

    std::deque<std::string> pending(topo_order.begin(), topo_order.end());
    while (!pending.empty()) {
      const std::string current = pending.front();
      pending.pop_front();
      if (!macros_.contains(current)) continue;

      if (options_.enable_linear && macro_deps_.at(current).size() == 1) {
        const std::string parent = *macro_deps_.at(current).begin();
        if (macro_dependents_.at(parent).size() == 1 && macros_.at(current) < macros_.at(parent)) {
          pending.push_front(merge_macros(parent, current, next_macro_id, "linear"));
          continue;
        }
      }

      if (macro_deps_.at(current).size() > 1) {
        std::string prepared = current;
        if (options_.enable_branching) {
          prepared = prepare_join_branches(current, next_macro_id);
          if (prepared != current) {
            pending.push_front(prepared);
            continue;
          }
        }

        if (options_.enable_joining) {
          const std::string merged =
              merge_join_parent_backward(prepared, in_deg, cached_tensors, next_macro_id);
          if (merged != prepared) pending.push_front(merged);
        }
      }
    }
    if (macros_.contains(virtual_join_id)) {
      if (options_.enable_branching) {
        prepare_join_branches(virtual_join_id, next_macro_id);
      }
      for (const auto& terminal : macro_deps_.at(virtual_join_id)) {
        macro_dependents_.at(terminal).erase(virtual_join_id);
      }
      macros_.erase(virtual_join_id);
      macro_deps_.erase(virtual_join_id);
      macro_dependents_.erase(virtual_join_id);
    }

    if (os_) {
      *os_ << "All backward macros generated" << std::endl;
      for (auto& [id, macro] : macros_) {
        *os_ << id << ": ";
        for (auto& op : macro.ops) {
          *os_ << op << " ";
        }
        *os_ << " -> [ " << macro.a << " " << macro.b << " ]" << std::endl;
      }
    }

    std::vector<std::string> final_order;
    std::set<std::string> executed_macros;

    auto pq_compare = [this](const std::string& a, const std::string& b) {
      return macros_[b] < macros_[a];
    };

    std::priority_queue<std::string, std::vector<std::string>, decltype(pq_compare)> ready_macros(
        pq_compare);

    for (auto& [id, deps_set] : macro_deps_) {
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

      for (auto& op : macros_[best_macro].ops) {
        final_order.push_back(op);
      }

      for (auto& child : macro_dependents_[best_macro]) {
        bool ready = true;
        for (auto& parent : macro_deps_[child]) {
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
};

class RankedSolver : public MacroSolver {
public:
  RankedSolver(Graph& graph, std::ostream* os = nullptr)
      : MacroSolver(graph, os, {false, false, false}) {}
};

class LinearSolver : public MacroSolver {
public:
  LinearSolver(Graph& graph, std::ostream* os = nullptr)
      : MacroSolver(graph, os, {true, false, false}) {}
};

class BranchingSolver : public MacroSolver {
public:
  BranchingSolver(Graph& graph, std::ostream* os = nullptr)
      : MacroSolver(graph, os, {true, true, false}) {}
};

class JoiningSolver : public MacroSolver {
public:
  JoiningSolver(Graph& graph, std::ostream* os = nullptr)
      : MacroSolver(graph, os, {true, false, true}) {}
};

class FullSolver : public MacroSolver {
public:
  FullSolver(Graph& graph, std::ostream* os = nullptr)
      : MacroSolver(graph, os, {true, true, true}) {}
};

inline std::vector<std::string> find_fw_macro_candidate_execution_order(
    Graph& graph, std::ostream* os = nullptr) {
  MacroSolver solver(graph, os);
  return solver.find_forward_order();
}

inline std::vector<std::string> find_fw_ranked_execution_order(Graph& graph, std::ostream* os) {
  RankedSolver solver(graph, os);
  return solver.find_forward_order();
}

inline std::vector<std::string> find_fw_linear_execution_order(Graph& graph, std::ostream* os) {
  LinearSolver solver(graph, os);
  return solver.find_forward_order();
}

inline std::vector<std::string> find_fw_branching_execution_order(Graph& graph, std::ostream* os) {
  BranchingSolver solver(graph, os);
  return solver.find_forward_order();
}

inline std::vector<std::string> find_fw_joining_execution_order(Graph& graph, std::ostream* os) {
  JoiningSolver solver(graph, os);
  return solver.find_forward_order();
}

inline std::vector<std::string> find_bw_macro_candidate_execution_order(
    Graph& graph, std::ostream* os = nullptr) {
  MacroSolver solver(graph, os);
  return solver.find_backward_order();
}

inline std::vector<std::string> find_bw_ranked_execution_order(Graph& graph,
                                                               std::ostream* os = nullptr) {
  RankedSolver solver(graph, os);
  return solver.find_backward_order();
}

inline std::vector<std::string> find_bw_linear_execution_order(Graph& graph,
                                                               std::ostream* os = nullptr) {
  LinearSolver solver(graph, os);
  return solver.find_backward_order();
}

inline std::vector<std::string> find_bw_branching_execution_order(Graph& graph,
                                                                  std::ostream* os = nullptr) {
  BranchingSolver solver(graph, os);
  return solver.find_backward_order();
}

inline std::vector<std::string> find_bw_joining_execution_order(Graph& graph,
                                                                std::ostream* os = nullptr) {
  JoiningSolver solver(graph, os);
  return solver.find_backward_order();
}
