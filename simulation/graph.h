#pragma once

#include <fstream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

class ActivationNode {
private:
  std::string uuid_;
  size_t size_;

public:
  ActivationNode(std::string uuid, size_t size)
      : uuid_(uuid),
        size_(size) {}

  const std::string& uuid() const { return uuid_; }

  size_t size() const { return size_; }
};

class OperationNode {
private:
  std::string uuid_;
  size_t workspace_req_;
  std::vector<ActivationNode*> inputs_;
  std::vector<ActivationNode*> outputs_;
  std::vector<ActivationNode*> cache_;

public:
  OperationNode(std::string uuid, size_t workspace_req, const std::vector<ActivationNode*>& inputs,
                const std::vector<ActivationNode*>& outputs,
                const std::vector<ActivationNode*>& cache)
      : uuid_(uuid),
        workspace_req_(workspace_req),
        inputs_(inputs),
        outputs_(outputs),
        cache_(cache) {}

  const std::string& uuid() const { return uuid_; }

  size_t workspace_req() const { return workspace_req_; }

  const std::vector<ActivationNode*>& inputs() const { return inputs_; }

  const std::vector<ActivationNode*>& outputs() const { return outputs_; }

  const std::vector<ActivationNode*>& cache() const { return cache_; }
};

class Graph {
private:
  std::unordered_map<std::string, OperationNode> ops_;
  std::unordered_map<std::string, ActivationNode> acts_;
  std::vector<ActivationNode*> inputs_;
  std::vector<ActivationNode*> outputs_;

public:
  Graph() = default;

  ActivationNode* add_act(std::string uuid, size_t size) {
    auto [act, inserted] = acts_.insert({uuid, ActivationNode(uuid, size)});
    return inserted ? &act->second : nullptr;
  }

  OperationNode* add_op(std::string uuid, size_t workspace_req,
                        const std::vector<ActivationNode*>& inputs,
                        const std::vector<ActivationNode*>& outputs,
                        const std::vector<ActivationNode*>& cache = {}) {
    auto [op, inserted] =
        ops_.insert({uuid, OperationNode(uuid, workspace_req, inputs, outputs, cache)});
    return inserted ? &op->second : nullptr;
  }

  void set_inputs(const std::vector<ActivationNode*>& inputs) { inputs_ = inputs; }

  void set_outputs(const std::vector<ActivationNode*>& outputs) { outputs_ = outputs; }

  const OperationNode& get_op(const std::string& uuid) const { return ops_.at(uuid); }

  const ActivationNode& get_act(const std::string& uuid) const { return acts_.at(uuid); }

  const std::vector<ActivationNode*>& inputs() const { return inputs_; }

  const std::vector<ActivationNode*>& outputs() const { return outputs_; }

  std::unordered_map<std::string, OperationNode>& op_nodes() { return ops_; }

  const std::unordered_map<std::string, OperationNode>& op_nodes() const { return ops_; }

  std::unordered_map<std::string, ActivationNode>& act_nodes() { return acts_; }

  const std::unordered_map<std::string, ActivationNode>& act_nodes() const { return acts_; }
};

inline std::pair<std::map<std::string, std::set<std::string>>,
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

inline std::map<std::string, int> get_out_deg(Graph& graph) {
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

inline void save_graph_to_dot(Graph& graph, const std::string& filename) {
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