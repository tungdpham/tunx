#include "nn/macro_solver.hpp"

#include <algorithm>
#include <deque>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>

namespace tunx {
namespace {

struct MacroNode {
  std::string id;
  std::vector<size_t> edges;
  long long a = 0;
  long long b = 0;
};

std::pair<long long, long long> rank(const MacroNode &macro) {
  return macro.b < 0 ? std::make_pair(0LL, macro.a) : std::make_pair(1LL, macro.b - macro.a);
}

bool lower_rank(const MacroNode &left, const MacroNode &right) { return rank(left) < rank(right); }

size_t estimate_peak_memory(const Graph &graph, const TensorBundle &input_map,
                            const std::vector<size_t> &order, const ForwardPlanCacheEntry &plan) {
  const Vec<Node> nodes = graph.nodes();
  const Vec<Edge> edges = graph.edges();
  std::map<NodeImpl *, size_t> live_bytes;
  std::map<NodeImpl *, int> remaining_consumers;
  std::map<std::string, Node> nodes_by_uid;
  for (const Node &node : nodes) nodes_by_uid[node->uid()] = node;
  for (const auto &[uid, tensor] : input_map) {
    const auto node_it = nodes_by_uid.find(uid);
    if (node_it != nodes_by_uid.end()) live_bytes[node_it->second.get()] = tensor.num_bytes();
  }
  for (const Edge &edge : edges) {
    for (const Node &producer : edge->producers()) ++remaining_consumers[producer.get()];
  }
  for (const Node &node : nodes) {
    if (graph.is_output(node)) ++remaining_consumers[node.get()];
  }

  size_t live_total = 0;
  for (const auto &[node, bytes] : live_bytes) live_total += bytes;
  size_t peak = live_total;
  for (size_t edge_index : order) {
    const Edge &edge = edges.at(edge_index);
    const EdgeExecutionProfile &profile = plan.edge_profiles.at(edge_index);
    const size_t generated = profile.output_bytes + profile.retained_bytes;
    peak = std::max(peak, live_total + generated + profile.peak_bytes);

    for (size_t output_index = 0; output_index < edge->consumers().size(); ++output_index) {
      live_bytes[edge->consumers()[output_index].get()] =
          profile.output_tensor_bytes.at(output_index);
      live_total += profile.output_tensor_bytes.at(output_index);
    }
    live_total += profile.retained_bytes;
    for (const Node &producer : edge->producers()) {
      if (--remaining_consumers[producer.get()] == 0) {
        const auto live_it = live_bytes.find(producer.get());
        if (live_it != live_bytes.end()) {
          live_total -= live_it->second;
          live_bytes.erase(live_it);
        }
      }
    }
  }
  return peak;
}

}  // namespace

std::vector<size_t> MacroSolver::find_order(TensorBundle &input_map) {
  ForwardPlanCacheEntry &plan = graph_.forward_plan_cache(input_map);
  const auto &edges = graph_.edges_;
  std::vector<size_t> topological_order(edges.size());
  for (size_t index = 0; index < edges.size(); ++index) topological_order[index] = index;

  // Profiles are populated by the real first forward pass to avoid executing
  // stateful layers a second time solely for planning.
  if (!plan.profiled || plan.edge_profiles.size() != edges.size()) return topological_order;

  std::map<NodeImpl *, size_t> producer;
  std::map<NodeImpl *, int> consumer_count;
  for (size_t index = 0; index < edges.size(); ++index) {
    for (const Node &consumer : edges[index]->consumers()) producer[consumer.get()] = index;
    for (const Node &input : edges[index]->producers()) ++consumer_count[input.get()];
  }
  for (const Node &output : graph_.output_nodes_) ++consumer_count[output.get()];

  std::map<std::string, MacroNode> macros;
  std::map<std::string, std::set<std::string>> deps;
  std::map<std::string, std::set<std::string>> dependents;
  for (size_t index = 0; index < edges.size(); ++index) {
    const std::string id = std::to_string(index);
    const EdgeExecutionProfile &profile = plan.edge_profiles.at(index);
    long long consumed_bytes = 0;
    for (const Node &input : edges[index]->producers()) {
      if (!graph_.is_input(input) && consumer_count[input.get()] == 1) {
        const auto producer_it = producer.find(input.get());
        if (producer_it != producer.end()) {
          consumed_bytes +=
              static_cast<long long>(plan.edge_profiles.at(producer_it->second).output_bytes);
        }
      }
    }

    const long long generated_bytes =
        static_cast<long long>(profile.output_bytes + profile.retained_bytes);
    macros[id] = {id,
                  {index},
                  generated_bytes + static_cast<long long>(profile.peak_bytes),
                  generated_bytes - consumed_bytes};
    for (const Node &input : edges[index]->producers()) {
      const auto producer_it = producer.find(input.get());
      if (producer_it != producer.end()) deps[id].insert(std::to_string(producer_it->second));
    }
  }
  for (const auto &[child, parents] : deps) {
    for (const std::string &parent : parents) dependents[parent].insert(child);
  }
  for (const auto &[id, macro] : macros) {
    deps.try_emplace(id);
    dependents.try_emplace(id);
  }

  const auto has_peers = [&](const std::string &macro_id) {
    for (size_t edge_index : macros.at(macro_id).edges) {
      for (const Node &input : edges[edge_index]->producers()) {
        if (consumer_count[input.get()] <= 1) continue;
        int internal_consumers = 0;
        for (size_t inner_edge : macros.at(macro_id).edges) {
          for (const Node &inner_input : edges[inner_edge]->producers()) {
            if (inner_input == input) ++internal_consumers;
          }
        }
        if (internal_consumers < consumer_count[input.get()]) return true;
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
      if (dependents.at(parent).size() != 1 || !lower_rank(macros.at(id), macros.at(parent))) break;
      id = merge(parent, id, "upward");
    }
    return id;
  };

  const auto merge_branch = [&](std::string ancestor) {
    bool merged = false;
    while (true) {
      std::string best_child;
      for (const std::string &child : dependents.at(ancestor)) {
        if (deps.at(child).size() != 1 || !lower_rank(macros.at(child), macros.at(ancestor)))
          continue;
        if (best_child.empty() || lower_rank(macros.at(child), macros.at(best_child)))
          best_child = child;
      }
      if (best_child.empty()) return merged ? merge_upward(ancestor) : ancestor;
      ancestor = merge(ancestor, best_child, "branch");
      merged = true;
    }
  };

  const auto prepare_join = [&](const std::string &join) {
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
      merge_branch(ancestor);
    }
  };

  std::deque<std::string> pending;
  for (size_t index : topological_order) pending.push_back(std::to_string(index));
  while (!pending.empty()) {
    const std::string current = pending.front();
    pending.pop_front();
    if (!macros.contains(current)) continue;
    if (deps.at(current).size() == 1) {
      const std::string parent = *deps.at(current).begin();
      if (dependents.at(parent).size() == 1 && lower_rank(macros.at(current), macros.at(parent))) {
        pending.push_front(merge(parent, current, "linear"));
        continue;
      }
    }
    if (deps.at(current).size() <= 1) continue;
    prepare_join(current);
    if (!macros.contains(current)) continue;
    std::string worst_parent;
    for (const std::string &parent : deps.at(current)) {
      if (dependents.at(parent).size() != 1 || has_peers(parent) ||
          !lower_rank(macros.at(current), macros.at(parent))) {
        continue;
      }
      if (worst_parent.empty() || lower_rank(macros.at(worst_parent), macros.at(parent))) {
        worst_parent = parent;
      }
    }
    if (!worst_parent.empty()) pending.push_front(merge(worst_parent, current, "join"));
  }

  std::vector<size_t> final_order;
  std::set<std::string> executed;
  const auto compare_ready = [&](const std::string &left, const std::string &right) {
    return lower_rank(macros.at(right), macros.at(left));
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
  graph_.last_forward_memory_estimate_ = {
      .topological_peak_bytes = estimate_peak_memory(graph_, input_map, topological_order, plan),
      .macro_peak_bytes = estimate_peak_memory(graph_, input_map, final_order, plan),
  };
  return final_order;
}

}  // namespace tunx