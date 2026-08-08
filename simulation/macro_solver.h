#pragma once

#include <algorithm>
#include <deque>
#include <ostream>
#include <queue>
#include <string>
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

class MacroSolver {
public:
  std::vector<std::string> find_order(Graph& graph, std::ostream* os = nullptr) {
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

    // topo order, join contraction (looks backward at each macro)
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

        if (macros[Y] < macros[X]) {
          if (best_X == "" || macros[best_X] < macros[X]) {
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

    // reverse topo order, fork/branch contraction (looks forward at each macro)
    while (!dq.empty()) {
      std::string Y = dq.front();
      dq.pop_front();

      std::string best_Z = "";
      for (auto& Z : macro_dependents[Y]) {
        if (macro_deps[Z].size() != 1) continue;

        if (macros[Z] < macros[Y]) {
          if (best_Z == "" || macros[Z] < macros[best_Z]) {
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
      return macros[b] < macros[a];
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
};
