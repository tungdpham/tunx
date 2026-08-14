#include "nn/macro_solver.hpp"

#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>

#include "nn/edge_profile.hpp"
#include "nn/execution_plan.hpp"

namespace tunx {
namespace {

struct MacroNode {
  std::string id;
  std::vector<Edge> edges;
  long long a = 0;
  long long b = 0;
};

std::pair<long long, long long> rank(const MacroNode &macro) {
  return macro.b < 0 ? std::make_pair(0LL, macro.a) : std::make_pair(1LL, macro.b - macro.a);
}

bool operator<(const MacroNode &left, const MacroNode &right) { return rank(left) < rank(right); }
}  // namespace

ExecutionPlan MacroSolver::find_order(const std::map<Edge, EdgeProfile> &edge_profiles) {
  const auto &edges = graph_.edges();

  std::map<Node, int> out_deg;
  for (const Edge &edge : edges) {
    for (const Node &producer : edge->producers()) {
      out_deg[producer]++;
    }
  }

  std::map<std::string, MacroNode> macros;
  std::map<std::string, std::set<std::string>> deps;
  std::map<std::string, std::set<std::string>> dependents;

  std::map<Node, std::vector<std::string>> node_to_producer_edges;
  for (const Edge &edge : edges) {
    for (const Node &consumer : edge->consumers()) {
      node_to_producer_edges[consumer].push_back(edge->uid());
    }

    deps[edge->uid()] = {};
    dependents[edge->uid()] = {};

    const EdgeProfile &profile = edge_profiles.at(edge);
    MacroNode macro{
        edge->uid(),
        {edge},
        static_cast<long long>(profile.total_mem),
        static_cast<long long>(profile.net_mem),
    };
    macros.emplace(edge->uid(), std::move(macro));
  }

  for (const Edge &child_edge : edges) {
    for (const Node &producer : child_edge->producers()) {
      for (const std::string &parent_uid : node_to_producer_edges[producer]) {
        deps[child_edge->uid()].insert(parent_uid);
        dependents[parent_uid].insert(child_edge->uid());
      }
    }
  }

  std::vector<std::string> topological_order;
  std::map<std::string, int> in_deg;
  for (const Edge &edge : edges) {
    in_deg[edge->uid()] = deps[edge->uid()].size();
  }
  std::queue<std::string> q;
  for (const Edge &edge : edges) {
    if (in_deg[edge->uid()] == 0) q.push(edge->uid());
  }
  while (!q.empty()) {
    std::string u = q.front();
    q.pop();
    topological_order.push_back(u);
    for (const std::string &v : dependents[u]) {
      if (--in_deg[v] == 0) q.push(v);
    }
  }

  const auto has_peers = [&](const std::string &macro_id) {
    for (const Edge &edge : macros.at(macro_id).edges) {
      for (const Node &input : edge->producers()) {
        if (out_deg[input] <= 1) continue;
        int internal_consumers = 0;
        for (const Edge &inner_edge : macros.at(macro_id).edges) {
          for (const Node &inner_input : inner_edge->producers()) {
            if (inner_input == input) ++internal_consumers;
          }
        }
        if (internal_consumers < out_deg[input]) return true;
      }
    }
    return false;
  };

  int next_macro_id = 0;
  auto merge = [&](const std::string &parent, const std::string &child, const char *reason) {
    const std::string id = "macro_" + std::to_string(next_macro_id++);
    const MacroNode &parent_macro = macros.at(parent);
    const MacroNode &child_macro = macros.at(child);
    MacroNode merged{id, parent_macro.edges,
                     std::max(parent_macro.a, parent_macro.b + child_macro.a),
                     parent_macro.b + child_macro.b};
    merged.edges.insert(merged.edges.end(), child_macro.edges.begin(), child_macro.edges.end());
    if (log_stream_)
      *log_stream_ << "Merging " << reason << " " << parent << " with " << child << " into " << id
                   << '\n';

    std::set<std::string> merged_deps = deps.at(parent);
    for (const std::string &dependency : deps.at(child)) {
      if (dependency != parent) merged_deps.insert(dependency);
    }
    std::set<std::string> merged_dependents = dependents.at(parent);
    merged_dependents.erase(child);
    merged_dependents.insert(dependents.at(child).begin(), dependents.at(child).end());

    macros[id] = std::move(merged);
    deps[id] = std::move(merged_deps);
    dependents[id] = std::move(merged_dependents);
    for (const std::string &dependency : deps.at(id)) {
      dependents.at(dependency).erase(parent);
      dependents.at(dependency).erase(child);
      dependents.at(dependency).insert(id);
    }
    for (const std::string &dependent : dependents.at(id)) {
      deps.at(dependent).erase(parent);
      deps.at(dependent).erase(child);
      deps.at(dependent).insert(id);
    }
    macros.erase(parent);
    macros.erase(child);
    deps.erase(parent);
    deps.erase(child);
    dependents.erase(parent);
    dependents.erase(child);
    return id;
  };

  const auto ancestor_distances = [&](const std::string &start) {
    std::map<std::string, int> distances{{start, 0}};
    std::queue<std::string> pending;
    pending.push(start);
    while (!pending.empty()) {
      const std::string current = pending.front();
      pending.pop();
      for (const std::string &parent : deps.at(current)) {
        if (distances.emplace(parent, distances.at(current) + 1).second) pending.push(parent);
      }
    }
    return distances;
  };

  const auto nearest_common_branching_ancestor = [&](const std::string &first,
                                                     const std::string &second) {
    const auto first_distances = ancestor_distances(first);
    const auto second_distances = ancestor_distances(second);
    std::string nearest;
    std::pair<int, int> best_distance;
    for (const auto &[ancestor, first_distance] : first_distances) {
      const auto second_it = second_distances.find(ancestor);
      if (second_it == second_distances.end() || dependents.at(ancestor).size() < 2) continue;
      const std::pair<int, int> distance = {std::max(first_distance, second_it->second),
                                            first_distance + second_it->second};
      if (nearest.empty() || distance < best_distance) {
        nearest = ancestor;
        best_distance = distance;
      }
    }
    return nearest;
  };

  const auto merge_upward = [&](std::string id) {
    while (deps.at(id).size() == 1) {
      const std::string parent = *deps.at(id).begin();
      if (dependents.at(parent).size() != 1 || !(macros.at(id) < macros.at(parent))) break;
      id = merge(parent, id, "upward");
    }
    return id;
  };

  const auto merge_branch = [&](std::string ancestor) {
    bool merged = false;
    while (true) {
      std::string best_child;
      for (const std::string &child : dependents.at(ancestor)) {
        if (deps.at(child).size() != 1 || !(macros.at(child) < macros.at(ancestor))) continue;
        if (best_child.empty() || macros.at(child) < macros.at(best_child)) best_child = child;
      }
      if (best_child.empty()) return merged ? merge_upward(ancestor) : ancestor;
      ancestor = merge(ancestor, best_child, "branch");
      merged = true;
    }
  };

  const auto prepare_join = [&](const std::string &join) -> std::string {
    const std::vector<std::string> parents(deps.at(join).begin(), deps.at(join).end());
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
    std::sort(ordered.begin(), ordered.end(),
              [&](const std::string &left, const std::string &right) {
                return distances.at(left) < distances.at(right);
              });
    for (const std::string &ancestor : ordered) {
      if (!macros.contains(ancestor)) continue;
      const std::string merged_ancestor = merge_branch(ancestor);
      if (!macros.contains(join)) return merged_ancestor;
    }
    return join;
  };

  const std::string virtual_join_id = "__virtual_join__";
  std::vector<std::string> terminal_macros;
  for (const auto &[macro_id, macro] : macros) {
    if (dependents.at(macro_id).empty()) terminal_macros.push_back(macro_id);
  }
  if (terminal_macros.size() > 1) {
    macros[virtual_join_id] = {virtual_join_id, {}, 0, std::numeric_limits<long long>::max()};
    deps[virtual_join_id] = {};
    dependents[virtual_join_id] = {};
    for (const auto &terminal : terminal_macros) {
      deps[virtual_join_id].insert(terminal);
      dependents[terminal].insert(virtual_join_id);
    }
  }

  std::deque<std::string> pending(topological_order.begin(), topological_order.end());
  while (!pending.empty()) {
    const std::string current = pending.front();
    pending.pop_front();
    if (!macros.contains(current)) continue;
    if (deps.at(current).size() == 1) {
      const std::string parent = *deps.at(current).begin();
      if (dependents.at(parent).size() == 1 && macros.at(current) < macros.at(parent)) {
        pending.push_front(merge(parent, current, "linear"));
        continue;
      }
    }
    if (deps.at(current).size() > 1) {
      const std::string prepared = prepare_join(current);
      if (prepared != current) {
        pending.push_front(prepared);
        continue;
      }
      std::string worst_parent;
      for (const std::string &parent : deps.at(current)) {
        if (dependents.at(parent).size() != 1 || has_peers(parent) ||
            !(macros.at(current) < macros.at(parent))) {
          continue;
        }
        if (worst_parent.empty() || macros.at(worst_parent) < macros.at(parent)) {
          worst_parent = parent;
        }
      }
      if (!worst_parent.empty()) pending.push_front(merge(worst_parent, current, "join"));
    }
  }

  if (macros.contains(virtual_join_id)) {
    prepare_join(virtual_join_id);
    for (const auto &terminal : deps.at(virtual_join_id)) {
      dependents.at(terminal).erase(virtual_join_id);
    }
    macros.erase(virtual_join_id);
    deps.erase(virtual_join_id);
    dependents.erase(virtual_join_id);
  }

  std::vector<Edge> final_order;
  std::set<std::string> executed;
  const auto compare_ready = [&](const std::string &left, const std::string &right) {
    return macros.at(right) < macros.at(left);
  };
  std::priority_queue<std::string, std::vector<std::string>, decltype(compare_ready)> ready(
      compare_ready);
  for (const auto &[id, macro] : macros) {
    if (deps.at(id).empty()) ready.push(id);
  }
  while (final_order.size() < edges.size()) {
    if (ready.empty()) throw std::runtime_error("Graph has unresolved macro dependencies");
    const std::string best = ready.top();
    ready.pop();
    if (!executed.insert(best).second) continue;
    final_order.insert(final_order.end(), macros.at(best).edges.begin(),
                       macros.at(best).edges.end());
    for (const std::string &child : dependents.at(best)) {
      if (std::all_of(deps.at(child).begin(), deps.at(child).end(),
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