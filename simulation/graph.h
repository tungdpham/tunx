#pragma once

#include <algorithm>
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
  size_t residual_mem_;
  std::vector<ActivationNode*> inputs_;
  std::vector<ActivationNode*> outputs_;
  std::vector<ActivationNode*> cache_;

public:
  OperationNode(std::string uuid, size_t workspace_req, const std::vector<ActivationNode*>& inputs,
                const std::vector<ActivationNode*>& outputs,
                const std::vector<ActivationNode*>& cache = {}, size_t residual_mem = 0)
      : uuid_(uuid),
        workspace_req_(workspace_req),
        residual_mem_(residual_mem),
        inputs_(inputs),
        outputs_(outputs),
        cache_(cache) {}

  const std::string& uuid() const { return uuid_; }

  size_t workspace_req() const { return workspace_req_; }

  size_t residual_mem() const { return residual_mem_; }

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
                        const std::vector<ActivationNode*>& cache = {},
                        size_t residual_mem = 0) {
    auto [op, inserted] =
        ops_.insert({uuid, OperationNode(uuid, workspace_req, inputs, outputs, cache, residual_mem)});
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

inline void save_graph_to_dot(Graph& graph, const std::string& filename, bool is_backward = false) {
  std::ofstream out(filename);
  out << "digraph G {\n";
  for (auto& [uuid, act] : graph.act_nodes()) {
    out << "  \"" << act.uuid() << "\" [shape=ellipse, label=\"" << act.uuid() << "\\n("
        << act.size() << ")\"];\n";
  }

  std::map<std::string, int> out_deg;
  std::map<std::string, int> in_deg;
  if (!is_backward) {
    out_deg = get_out_deg(graph);
  } else {
    for (auto& [uuid, node] : graph.op_nodes()) {
      for (auto* t : node.outputs()) {
        in_deg[t->uuid()]++;
      }
    }
    for (auto* t : graph.inputs()) {
      in_deg[t->uuid()]++;
    }
  }

  for (auto& [uuid, op] : graph.op_nodes()) {
    long long all_outputs = 0;
    for (auto* t : op.outputs()) all_outputs += t->size();
    long long all_inputs = 0;
    for (auto* t : op.inputs()) all_inputs += t->size();
    long long workspace = op.workspace_req();
    long long residual = op.residual_mem();

    long long a = 0;
    long long b = 0;

    if (!is_backward) {
      a = all_outputs + workspace + residual;
      long long memory_consumes = 0;
      for (auto* t : op.inputs()) {
        if (out_deg[t->uuid()] == 1 &&
            std::find(op.cache().begin(), op.cache().end(), t) == op.cache().end()) {
          memory_consumes += t->size();
        }
      }
      b = all_outputs + residual - memory_consumes;
    } else {
      a = all_outputs + workspace;
      long long memory_consumes = residual;
      for (auto* t : op.outputs()) {
        if (in_deg[t->uuid()] == 1) {
          memory_consumes += t->size();
        }
      }
      b = all_inputs - memory_consumes;
    }

    out << "  \"" << op.uuid() << "\" [shape=box, label=\"" << op.uuid() << "\\n"
        << "ws: " << workspace << ", res: " << residual << "\\n"
        << "a: " << a << ", b: " << b << "\"];\n";

    for (auto* in : op.inputs()) {
      if (!is_backward) {
        out << "  \"" << in->uuid() << "\" -> \"" << op.uuid() << "\";\n";
      } else {
        out << "  \"" << op.uuid() << "\" -> \"" << in->uuid() << "\";\n";
      }
    }
    for (auto* out_act : op.outputs()) {
      if (!is_backward) {
        out << "  \"" << op.uuid() << "\" -> \"" << out_act->uuid() << "\";\n";
      } else {
        out << "  \"" << out_act->uuid() << "\" -> \"" << op.uuid() << "\";\n";
      }
    }
  }
  out << "}\n";
}