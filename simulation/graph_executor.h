#pragma once

#include <stdexcept>

#include "allocator.h"
#include "graph.h"
#include "tensor.h"

class GraphExecutor {
private:
  Graph& graph;
  std::unordered_map<ActivationNode*, Tensor> data_;
  std::unordered_map<OperationNode*, Tensor> workspace_;
  std::unordered_map<ActivationNode*, size_t> ref_count_;
  std::unordered_map<ActivationNode*, OperationNode*>
      act_deps_;  // which op node create this activation node
  std::unordered_map<ActivationNode*, std::vector<OperationNode*>>
      act_dependents_;  // which op nodes depends on this activation node

public:
  GraphExecutor(Graph& graph)
      : graph(graph) {
    for (auto& [uuid, act_node] : graph.act_nodes()) {
      ref_count_[&act_node] = 0;
      data_[&act_node] = nullptr;
    }

    for (auto& [uuid, op_node] : graph.op_nodes()) {
      for (ActivationNode* dep : op_node.inputs()) {
        act_dependents_[dep].push_back(&op_node);
      }
      for (ActivationNode* out : op_node.outputs()) {
        act_deps_[out] = &op_node;
      }
    }
  }

  void init_boundaries(Allocator* allocator) {
    for (ActivationNode* input : graph.inputs()) {
      ref_count_[input] = act_dependents_[input].size();  // ref count = out degree.
      data_[input] = Tensor(allocator, input->size());
    }

    for (ActivationNode* output : graph.outputs()) {
      ref_count_[output]++;  // increment for output since they need to be retained for extraction.
    }
  }

  void run_op_node(OperationNode* node, Allocator* allocator) {
    // sanity check
    if (!node) {
      throw std::invalid_argument("GraphExecutor::run_op_node: Node is null.");
    }
    for (ActivationNode* input : node->inputs()) {
      if (!data_[input]) {
        throw std::runtime_error("Input tensor is not available.");
      }
    }

    // allocate workspace
    if (node->workspace_req() > 0) {
      workspace_[node] = Tensor(allocator, node->workspace_req());
    }

    // allocate outputs
    for (ActivationNode* output : node->outputs()) {
      if (data_[output]) {
        throw std::runtime_error("GraphExecutor::run_op_node: Output tensor is already allocated.");
      }
      ref_count_[output] += act_dependents_[output].size();
      data_[output] = Tensor(allocator, output->size());
    }

    // execute some stuffs in real code here but we will skip since this is simulation

    // free inputs
    for (ActivationNode* input : node->inputs()) {
      if (ref_count_[input] > 0) {
        ref_count_[input]--;
        if (ref_count_[input] == 0) {
          data_[input] = nullptr;
        }
      }
    }

    // free workspace
    workspace_[node] = Tensor();
  }

  void undo_run_op_node(OperationNode* node, Allocator* allocator) {
    // sanity check
    if (!node) {
      throw std::invalid_argument("GraphExecutor::undo_run_op_node: Node is null.");
    }

    // free outputs
    for (ActivationNode* output : node->outputs()) {
      if (!data_[output]) {
        throw std::runtime_error("GraphExecutor::undo_run_op_node: Output tensor is not allocated.");
      }
      ref_count_[output] -= act_dependents_[output].size();
      data_[output] = nullptr;
    }

    // allocate inputs
    for (ActivationNode* input : node->inputs()) {
      if (ref_count_[input] == 0) {
        data_[input] = Tensor(allocator, input->size());
      }
      ref_count_[input]++;
    }
  }
};
