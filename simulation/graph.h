#pragma once

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

  std::string uuid() const { return uuid_; }

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

  std::string uuid() const { return uuid_; }

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

  std::vector<OperationNode*> op_nodes() {
    std::vector<OperationNode*> op_nodes;
    for (auto& [uuid, op] : ops_) {
      op_nodes.push_back(&op);
    }
    return op_nodes;
  }

  std::vector<ActivationNode*> act_nodes() {
    std::vector<ActivationNode*> act_nodes;
    for (auto& [uuid, act] : acts_) {
      act_nodes.push_back(&act);
    }
    return act_nodes;
  }
};