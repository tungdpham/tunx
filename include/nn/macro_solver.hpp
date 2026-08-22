#pragma once

#include <map>
#include <ostream>
#include <set>
#include <string>
#include <vector>

#include "nn/edge_profile.hpp"
#include "nn/execution_plan.hpp"
#include "nn/graph.hpp"

namespace tunx {

struct MacroNode {
  std::string id;
  std::vector<Edge> edges;
  long long a = 0;
  long long b = 0;
};

struct SolverOptions {
  bool enable_linear = true;
  bool enable_branching = true;
  bool enable_joining = true;
};

class MacroSolver {
private:
  Graph &graph_;
  std::ostream *log_stream_;
  SolverOptions options_;

  // generated
  std::map<std::string, MacroNode> macros_;
  std::map<std::string, std::set<std::string>> macro_deps_;
  std::map<std::string, std::set<std::string>> macro_dependents_;

  bool has_peers_forward(const std::string &macro_id, const std::map<Node, int> &out_deg,
                         const std::set<std::string> &cached_tensors) const;
  bool has_peers_backward(const std::string &macro_id, const std::map<Node, int> &out_deg,
                          const std::set<std::string> &cached_tensors) const;
  std::string merge_macros(const std::string &parent, const std::string &child, int &next_macro_id,
                           const char *reason);
  std::map<std::string, int> ancestor_distances(const std::string &macro_id) const;
  std::string nearest_common_branching_ancestor(const std::string &first,
                                                const std::string &second) const;
  std::string merge_upward_recursively(std::string macro_id, int &next_macro_id);
  std::string merge_branch_recursively(std::string ancestor, int &next_macro_id);
  std::string prepare_join_branches(const std::string &join, int &next_macro_id);

public:
  MacroSolver(Graph &graph, std::ostream *log_stream = nullptr, SolverOptions options = {})
      : graph_(graph),
        log_stream_(log_stream),
        options_(options) {}

  ExecutionPlan find_forward_order(const std::map<Edge, EdgeProfile> &edge_profiles);
  ExecutionPlan find_backward_order(const std::map<Edge, EdgeProfile> &edge_profiles);
};

class RankedSolver : public MacroSolver {
public:
  RankedSolver(Graph &graph, std::ostream *os = nullptr)
      : MacroSolver(graph, os, {false, false, false}) {}
};

class LinearSolver : public MacroSolver {
public:
  LinearSolver(Graph &graph, std::ostream *os = nullptr)
      : MacroSolver(graph, os, {true, false, false}) {}
};

class BranchingSolver : public MacroSolver {
public:
  BranchingSolver(Graph &graph, std::ostream *os = nullptr)
      : MacroSolver(graph, os, {true, true, false}) {}
};

class JoiningSolver : public MacroSolver {
public:
  JoiningSolver(Graph &graph, std::ostream *os = nullptr)
      : MacroSolver(graph, os, {true, false, true}) {}
};

class FlatBJSolver : public MacroSolver {
public:
  FlatBJSolver(Graph &graph, std::ostream *os = nullptr)
      : MacroSolver(graph, os, {true, true, true}) {}
};

class FullSolver : public MacroSolver {
public:
  FullSolver(Graph &graph, std::ostream *os = nullptr)
      : MacroSolver(graph, os, {true, true, true}) {}
};

}  // namespace tunx