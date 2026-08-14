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

class MacroSolver {
private:
  Graph &graph_;
  std::ostream *log_stream_;

  // generated
  std::map<std::string, MacroNode> macros_;
  std::map<std::string, std::set<std::string>> macro_deps_;
  std::map<std::string, std::set<std::string>> macro_dependents_;

  bool has_peers(const std::string &macro_id, const std::map<Node, int> &out_deg,
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
  MacroSolver(Graph &graph, std::ostream *log_stream = nullptr)
      : graph_(graph),
        log_stream_(log_stream) {}

  ExecutionPlan find_forward_order(const std::map<Edge, EdgeProfile> &edge_profiles);
};
}  // namespace tunx