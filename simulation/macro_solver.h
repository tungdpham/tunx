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
private:
  // core
  Graph& graph_;
  std::ostream* os_;

  // generated
  std::map<std::string, MacroNode> macros_;
  std::map<std::string, std::set<std::string>> macro_deps_;
  std::map<std::string, std::set<std::string>> macro_dependents_;

public:
  MacroSolver(Graph& graph, std::ostream* os = nullptr)
      : graph_(graph),
        os_(os) {}

  std::vector<std::string> find_order() {
    std::vector<std::string> op_ids;
    for (auto& [uuid, node] : graph_.op_nodes()) {
      op_ids.push_back(node.uuid());
    }

    auto out_deg = get_out_deg(graph_);
    auto [deps, dependents] = get_dependencies(graph_);

    for (auto& id : op_ids) {
      auto& node = graph_.get_op(id);
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
      for (auto& X : macro_deps_[Y]) {
        if (macro_dependents_[X].size() != 1) continue;

        // drop if share input tensor with any peers
        bool has_peers = false;
        for (const auto& op_id : macros_[X].ops) {
          const auto& op_node = graph_.get_op(op_id);
          for (auto* t : op_node.inputs()) {
            if (out_deg[t->uuid()] > 1) {
              int internal_consumers = 0;
              for (const auto& inner_op_id : macros_[X].ops) {
                const auto& inner_op = graph_.get_op(inner_op_id);
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

        if (macros_[Y] < macros_[X]) {
          if (best_X == "" || macros_[best_X] < macros_[X]) {
            best_X = X;
          }
        }
      }

      if (best_X != "") {
        if (os_) {
          *os_ << "Merging forward macro " << macros_[best_X].id << " [";
          for (auto op : macros_[best_X].ops) {
            *os_ << op << " ";
          }
          *os_ << ", a:" << macros_[best_X].a << ", b:" << macros_[best_X].b << " ] with macro "
               << macros_[Y].id << " [";
          for (auto op : macros_[Y].ops) {
            *os_ << op << " ";
          }
          *os_ << ", a:" << macros_[Y].a << ", b:" << macros_[Y].b << " ] into macro "
               << "macro_" + std::to_string(next_macro_id) << std::endl;
        }
        std::string XY_id = "macro_" + std::to_string(next_macro_id++);
        MacroNode XY;
        XY.id = XY_id;
        XY.ops = macros_[best_X].ops;
        XY.ops.insert(XY.ops.end(), macros_[Y].ops.begin(), macros_[Y].ops.end());
        XY.a = std::max(macros_[best_X].a, macros_[best_X].b + macros_[Y].a);
        XY.b = macros_[best_X].b + macros_[Y].b;
        macros_[XY_id] = XY;

        macro_deps_[XY_id] = macro_deps_[best_X];
        for (auto& dep : macro_deps_[Y]) {
          if (dep != best_X) macro_deps_[XY_id].insert(dep);
        }

        macro_dependents_[XY_id] = macro_dependents_[best_X];
        macro_dependents_[XY_id].erase(Y);
        for (auto& child : macro_dependents_[Y]) {
          macro_dependents_[XY_id].insert(child);
        }

        for (auto& p : macro_deps_[XY_id]) {
          macro_dependents_[p].erase(best_X);
          macro_dependents_[p].erase(Y);
          macro_dependents_[p].insert(XY_id);
        }
        for (auto& child : macro_dependents_[XY_id]) {
          macro_deps_[child].erase(best_X);
          macro_deps_[child].erase(Y);
          macro_deps_[child].insert(XY_id);
        }

        macros_.erase(best_X);
        macros_.erase(Y);
        macro_deps_.erase(best_X);
        macro_deps_.erase(Y);
        macro_dependents_.erase(best_X);
        macro_dependents_.erase(Y);

        dq.push_front(XY_id);
      }
    }

    // sort new contracted graph in reverse topological order
    std::map<std::string, int> in_deg;
    for (const auto& [m_id, _] : macros_) {
      in_deg[m_id] = macro_deps_[m_id].size();
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
      for (const auto& v : macro_dependents_[u]) {
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
      for (auto& Z : macro_dependents_[Y]) {
        if (macro_deps_[Z].size() != 1) continue;

        if (macros_[Z] < macros_[Y]) {
          if (best_Z == "" || macros_[Z] < macros_[best_Z]) {
            best_Z = Z;
          }
        }
      }

      if (best_Z != "") {
        if (os_) {
          *os_ << "Merging backward macro " << macros_[Y].id << " [";
          for (auto op : macros_[Y].ops) {
            *os_ << op << " ";
          }
          *os_ << ", a: " << macros_[Y].a << ", b: " << macros_[Y].b << "] with macro "
               << macros_[best_Z].id << " [";
          for (auto op : macros_[best_Z].ops) {
            *os_ << op << " ";
          }
          *os_ << ", a: " << macros_[best_Z].a << ", b: " << macros_[best_Z].b << "] into macro "
               << "macro_" + std::to_string(next_macro_id) << std::endl;
        }
        std::string YZ_id = "macro_" + std::to_string(next_macro_id++);
        MacroNode YZ;
        YZ.id = YZ_id;
        YZ.ops = macros_[Y].ops;
        YZ.ops.insert(YZ.ops.end(), macros_[best_Z].ops.begin(), macros_[best_Z].ops.end());
        YZ.a = std::max(macros_[Y].a, macros_[Y].b + macros_[best_Z].a);
        YZ.b = macros_[Y].b + macros_[best_Z].b;
        macros_[YZ_id] = YZ;

        macro_deps_[YZ_id] = macro_deps_[Y];
        for (auto& dep : macro_deps_[best_Z]) {
          if (dep != Y) macro_deps_[YZ_id].insert(dep);
        }

        macro_dependents_[YZ_id] = macro_dependents_[Y];
        macro_dependents_[YZ_id].erase(best_Z);
        for (auto& child : macro_dependents_[best_Z]) {
          macro_dependents_[YZ_id].insert(child);
        }

        for (auto& p : macro_deps_[YZ_id]) {
          macro_dependents_[p].erase(Y);
          macro_dependents_[p].erase(best_Z);
          macro_dependents_[p].insert(YZ_id);
        }
        for (auto& child : macro_dependents_[YZ_id]) {
          macro_deps_[child].erase(Y);
          macro_deps_[child].erase(best_Z);
          macro_deps_[child].insert(YZ_id);
        }

        macros_.erase(Y);
        macros_.erase(best_Z);
        macro_deps_.erase(Y);
        macro_deps_.erase(best_Z);
        macro_dependents_.erase(Y);
        macro_dependents_.erase(best_Z);

        dq.push_front(YZ_id);
        continue;
      }
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
};
