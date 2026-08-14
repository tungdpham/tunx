#include "nn/macro_solver.hpp"

#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <ostream>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>

#include "nn/edge_profile.hpp"
#include "nn/execution_plan.hpp"

namespace tunx {
namespace {

std::pair<long long, long long> rank(const MacroNode &macro) {
  return macro.b < 0 ? std::make_pair(0LL, macro.a) : std::make_pair(1LL, macro.b - macro.a);
}

bool operator<(const MacroNode &left, const MacroNode &right) { return rank(left) < rank(right); }
}  // namespace

bool MacroSolver::has_peers(const std::string &macro_id, const std::map<Node, int> &out_deg,
                            const std::set<std::string> &cached_tensors) const {
  for (const Edge &edge : macros_.at(macro_id).edges) {
    for (const Node &input : edge->producers()) {
      if (out_deg.at(input) <= 1) continue;
      if (cached_tensors.count(input->uid())) continue;
      int internal_consumers = 0;
      for (const Edge &inner_edge : macros_.at(macro_id).edges) {
        for (const Node &inner_input : inner_edge->producers()) {
          if (inner_input == input) ++internal_consumers;
        }
      }
      if (internal_consumers < out_deg.at(input)) return true;
    }
  }
  return false;
}

std::string MacroSolver::merge_macros(const std::string &parent, const std::string &child,
                                      int &next_macro_id, const char *reason) {
  const std::string id = "macro_" + std::to_string(next_macro_id++);
  const MacroNode &parent_macro = macros_.at(parent);
  const MacroNode &child_macro = macros_.at(child);
  MacroNode merged{id, parent_macro.edges, std::max(parent_macro.a, parent_macro.b + child_macro.a),
                   parent_macro.b + child_macro.b};
  merged.edges.insert(merged.edges.end(), child_macro.edges.begin(), child_macro.edges.end());
  if (log_stream_) {
    *log_stream_ << "Merging parent " << parent << " [a: " << parent_macro.a
                 << " b: " << parent_macro.b << "] with child " << child << " [a: " << child_macro.a
                 << " b: " << child_macro.b << "] into " << id << " [a: " << merged.a
                 << " b: " << merged.b << "] (reason: " << reason << ")\n";
  }
  std::set<std::string> merged_deps = macro_deps_.at(parent);
  for (const std::string &dependency : macro_deps_.at(child)) {
    if (dependency != parent) merged_deps.insert(dependency);
  }
  std::set<std::string> merged_dependents = macro_dependents_.at(parent);
  merged_dependents.erase(child);
  merged_dependents.insert(macro_dependents_.at(child).begin(), macro_dependents_.at(child).end());

  macros_[id] = std::move(merged);
  macro_deps_[id] = std::move(merged_deps);
  macro_dependents_[id] = std::move(merged_dependents);
  for (const std::string &dependency : macro_deps_.at(id)) {
    macro_dependents_.at(dependency).erase(parent);
    macro_dependents_.at(dependency).erase(child);
    macro_dependents_.at(dependency).insert(id);
  }
  for (const std::string &dependent : macro_dependents_.at(id)) {
    macro_deps_.at(dependent).erase(parent);
    macro_deps_.at(dependent).erase(child);
    macro_deps_.at(dependent).insert(id);
  }
  macros_.erase(parent);
  macros_.erase(child);
  macro_deps_.erase(parent);
  macro_deps_.erase(child);
  macro_dependents_.erase(parent);
  macro_dependents_.erase(child);
  return id;
}

std::map<std::string, int> MacroSolver::ancestor_distances(const std::string &macro_id) const {
  std::map<std::string, int> distances{{macro_id, 0}};
  std::queue<std::string> pending;
  pending.push(macro_id);
  while (!pending.empty()) {
    const std::string current = pending.front();
    pending.pop();
    for (const std::string &parent : macro_deps_.at(current)) {
      if (distances.emplace(parent, distances.at(current) + 1).second) pending.push(parent);
    }
  }
  return distances;
}

std::string MacroSolver::nearest_common_branching_ancestor(const std::string &first,
                                                           const std::string &second) const {
  const auto first_distances = ancestor_distances(first);
  const auto second_distances = ancestor_distances(second);
  std::string nearest;
  std::pair<int, int> best_distance;
  for (const auto &[ancestor, first_distance] : first_distances) {
    const auto second_it = second_distances.find(ancestor);
    if (second_it == second_distances.end() || macro_dependents_.at(ancestor).size() < 2) continue;
    const std::pair<int, int> distance = {std::max(first_distance, second_it->second),
                                          first_distance + second_it->second};
    if (nearest.empty() || distance < best_distance) {
      nearest = ancestor;
      best_distance = distance;
    }
  }
  return nearest;
}

std::string MacroSolver::merge_upward_recursively(std::string macro_id, int &next_macro_id) {
  while (macro_deps_.at(macro_id).size() == 1) {
    const std::string parent = *macro_deps_.at(macro_id).begin();
    if (macro_dependents_.at(parent).size() != 1 || !(macros_.at(macro_id) < macros_.at(parent)))
      break;
    macro_id = merge_macros(parent, macro_id, next_macro_id, "upward");
  }
  return macro_id;
}

std::string MacroSolver::merge_branch_recursively(std::string ancestor, int &next_macro_id) {
  bool merged = false;
  while (true) {
    std::string best_child;
    for (const std::string &child : macro_dependents_.at(ancestor)) {
      if (macro_deps_.at(child).size() != 1 || !(macros_.at(child) < macros_.at(ancestor)))
        continue;
      if (best_child.empty() || macros_.at(child) < macros_.at(best_child)) best_child = child;
    }
    if (best_child.empty())
      return merged ? merge_upward_recursively(ancestor, next_macro_id) : ancestor;
    ancestor = merge_macros(ancestor, best_child, next_macro_id, "branch");
    merged = true;
  }
}

std::string MacroSolver::prepare_join_branches(const std::string &join, int &next_macro_id) {
  const std::vector<std::string> parents(macro_deps_.at(join).begin(), macro_deps_.at(join).end());
  std::set<std::string> ancestors;
  for (size_t first = 0; first < parents.size(); ++first) {
    for (size_t second = first + 1; second < parents.size(); ++second) {
      const std::string ancestor =
          nearest_common_branching_ancestor(parents[first], parents[second]);
      if (!ancestor.empty()) ancestors.insert(ancestor);
    }
  }
  const auto distances = ancestor_distances(join);
  std::vector<std::string> ordered(ancestors.begin(), ancestors.end());
  std::sort(ordered.begin(), ordered.end(), [&](const std::string &left, const std::string &right) {
    return distances.at(left) < distances.at(right);
  });
  for (const std::string &ancestor : ordered) {
    if (!macros_.contains(ancestor)) continue;
    const std::string merged_ancestor = merge_branch_recursively(ancestor, next_macro_id);
    if (!macros_.contains(join)) return merged_ancestor;
  }
  return join;
}

ExecutionPlan MacroSolver::find_forward_order(const std::map<Edge, EdgeProfile> &edge_profiles) {
  const auto &edges = graph_.edges();

  std::map<Node, int> out_deg;
  for (const Edge &edge : edges) {
    for (const Node &producer : edge->producers()) {
      out_deg[producer]++;
    }
  }

  std::set<std::string> cached_tensors;
  for (const auto &[edge, profile] : edge_profiles) {
    for (const std::string &uid : profile.cached_inputs) {
      cached_tensors.insert(uid);
    }
  }

  macros_.clear();
  macro_deps_.clear();
  macro_dependents_.clear();

  std::map<Node, std::vector<std::string>> node_to_producer_edges;
  for (const Edge &edge : edges) {
    for (const Node &consumer : edge->consumers()) {
      node_to_producer_edges[consumer].push_back(edge->uid());
    }

    macro_deps_[edge->uid()] = {};
    macro_dependents_[edge->uid()] = {};

    const EdgeProfile &profile = edge_profiles.at(edge);
    MacroNode macro{
        edge->uid(),
        {edge},
        static_cast<long long>(profile.total_mem),
        static_cast<long long>(profile.net_mem),
    };
    macros_.emplace(edge->uid(), std::move(macro));
  }

  for (const Edge &child_edge : edges) {
    for (const Node &producer : child_edge->producers()) {
      for (const std::string &parent_uid : node_to_producer_edges[producer]) {
        macro_deps_[child_edge->uid()].insert(parent_uid);
        macro_dependents_[parent_uid].insert(child_edge->uid());
      }
    }
  }

  std::vector<std::string> topological_order;
  std::map<std::string, int> in_deg;
  for (const Edge &edge : edges) {
    in_deg[edge->uid()] = macro_deps_[edge->uid()].size();
  }
  std::queue<std::string> q;
  for (const Edge &edge : edges) {
    if (in_deg[edge->uid()] == 0) q.push(edge->uid());
  }
  while (!q.empty()) {
    std::string u = q.front();
    q.pop();
    topological_order.push_back(u);
    for (const std::string &v : macro_dependents_[u]) {
      if (--in_deg[v] == 0) q.push(v);
    }
  }

  int next_macro_id = 0;

  const std::string virtual_join_id = "__virtual_join__";
  std::vector<std::string> terminal_macros;
  for (const auto &[macro_id, macro] : macros_) {
    if (macro_dependents_.at(macro_id).empty()) terminal_macros.push_back(macro_id);
  }
  if (terminal_macros.size() > 1) {
    macros_[virtual_join_id] = {virtual_join_id, {}, 0, std::numeric_limits<long long>::max()};
    macro_deps_[virtual_join_id] = {};
    macro_dependents_[virtual_join_id] = {};
    for (const auto &terminal : terminal_macros) {
      macro_deps_[virtual_join_id].insert(terminal);
      macro_dependents_[terminal].insert(virtual_join_id);
    }
  }

  std::deque<std::string> pending(topological_order.begin(), topological_order.end());
  while (!pending.empty()) {
    const std::string current = pending.front();
    pending.pop_front();
    if (!macros_.contains(current)) continue;
    if (macro_deps_.at(current).size() == 1) {
      const std::string parent = *macro_deps_.at(current).begin();
      if (macro_dependents_.at(parent).size() == 1 && macros_.at(current) < macros_.at(parent)) {
        pending.push_front(merge_macros(parent, current, next_macro_id, "linear"));
        continue;
      }
    }
    if (macro_deps_.at(current).size() > 1) {
      const std::string prepared = prepare_join_branches(current, next_macro_id);
      if (prepared != current) {
        pending.push_front(prepared);
        continue;
      }
      std::string worst_parent;
      for (const std::string &parent : macro_deps_.at(current)) {
        if (macro_dependents_.at(parent).size() != 1 ||
            has_peers(parent, out_deg, cached_tensors) ||
            !(macros_.at(current) < macros_.at(parent))) {
          continue;
        }
        if (worst_parent.empty() || macros_.at(worst_parent) < macros_.at(parent)) {
          worst_parent = parent;
        }
      }
      if (!worst_parent.empty())
        pending.push_front(merge_macros(worst_parent, current, next_macro_id, "join"));
    }
  }

  if (macros_.contains(virtual_join_id)) {
    prepare_join_branches(virtual_join_id, next_macro_id);
    for (const auto &terminal : macro_deps_.at(virtual_join_id)) {
      macro_dependents_.at(terminal).erase(virtual_join_id);
    }
    macros_.erase(virtual_join_id);
    macro_deps_.erase(virtual_join_id);
    macro_dependents_.erase(virtual_join_id);
  }

  std::vector<Edge> final_order;
  std::set<std::string> executed;

  if (log_stream_) {
    *log_stream_ << "--- Macros before Kahn's ---\n";
    for (const auto &[id, macro] : macros_) {
      *log_stream_ << "Macro " << id << " [rank " << rank(macro).first << ", " << rank(macro).second
                   << "]:\n  Deps: ";
      for (const auto &dep : macro_deps_.at(id)) *log_stream_ << dep << " ";
      *log_stream_ << "\n  Dependents: ";
      for (const auto &dep : macro_dependents_.at(id)) *log_stream_ << dep << " ";
      *log_stream_ << "\n";
    }
    *log_stream_ << "----------------------------\n";
  }

  const auto compare_ready = [&](const std::string &left, const std::string &right) {
    return macros_.at(right) < macros_.at(left);
  };
  std::priority_queue<std::string, std::vector<std::string>, decltype(compare_ready)> ready(
      compare_ready);
  for (const auto &[id, macro] : macros_) {
    if (macro_deps_.at(id).empty()) ready.push(id);
  }
  while (final_order.size() < edges.size()) {
    if (ready.empty()) throw std::runtime_error("Graph has unresolved macro dependencies");
    const std::string best = ready.top();
    ready.pop();
    if (!executed.insert(best).second) continue;
    final_order.insert(final_order.end(), macros_.at(best).edges.begin(),
                       macros_.at(best).edges.end());
    for (const std::string &child : macro_dependents_.at(best)) {
      if (std::all_of(macro_deps_.at(child).begin(), macro_deps_.at(child).end(),
                      [&](const std::string &parent) { return executed.contains(parent); })) {
        ready.push(child);
      }
    }
  }

  ExecutionPlan plan;
  plan.order = final_order;
  return plan;
}

}  // namespace tunx