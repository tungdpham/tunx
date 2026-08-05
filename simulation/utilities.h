#pragma once

#include "graph.h"
#include <algorithm>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

inline std::vector<std::string> topological_sort(Graph& graph, const std::vector<std::string>& ops) {
  std::unordered_map<std::string, std::string> tensor_producer;
  for (auto* node : graph.op_nodes()) {
    for (auto* t : node->outputs()) {
      tensor_producer[t->uuid()] = node->uuid();
    }
  }

  std::unordered_set<std::string> op_set(ops.begin(), ops.end());
  std::unordered_map<std::string, int> in_degree;
  std::unordered_map<std::string, std::vector<std::string>> adj;

  for (const auto& op_id : ops) {
    in_degree[op_id] = 0;
  }

  for (const auto& op_id : ops) {
    const auto& node = graph.get_op(op_id);
    for (auto* t : node.inputs()) {
      auto it = tensor_producer.find(t->uuid());
      if (it != tensor_producer.end()) {
        std::string prod_id = it->second;
        if (op_set.count(prod_id) && prod_id != op_id) {
          adj[prod_id].push_back(op_id);
          in_degree[op_id]++;
        }
      }
    }
  }

  std::queue<std::string> q;
  for (const auto& op_id : ops) {
    if (in_degree[op_id] == 0) {
      q.push(op_id);
    }
  }

  std::vector<std::string> sorted_ops;
  while (!q.empty()) {
    std::string u = q.front();
    q.pop();
    sorted_ops.push_back(u);

    for (const auto& v : adj[u]) {
      if (--in_degree[v] == 0) {
        q.push(v);
      }
    }
  }

  if (sorted_ops.size() != ops.size()) {
    throw std::runtime_error("Cycle detected in topological sort");
  }

  return sorted_ops;
}

inline std::vector<std::string> reverse_topological_sort(Graph& graph, const std::vector<std::string>& ops) {
  auto sorted = topological_sort(graph, ops);
  std::reverse(sorted.begin(), sorted.end());
  return sorted;
}
