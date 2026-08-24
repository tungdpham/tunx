#pragma once

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "allocator.h"
#include "graph.h"
#include "tensor.h"

class GraphExecutor {
private:
  Graph& graph;
  std::unordered_map<ActivationNode*, Tensor> activation_data_;
  std::unordered_map<ActivationNode*, Tensor> gradient_data_;

  std::unordered_map<const OperationNode*, Tensor> workspace_;
  std::unordered_map<const OperationNode*, Tensor> residual_;

  std::unordered_map<ActivationNode*, size_t> forward_uses_;
  std::unordered_map<ActivationNode*, size_t> cache_uses_;
  std::unordered_map<ActivationNode*, size_t> grad_contributions_;
  std::unordered_map<ActivationNode*, size_t> grad_uses_;

  std::unordered_map<ActivationNode*, OperationNode*> act_deps_;
  std::unordered_map<ActivationNode*, std::vector<OperationNode*>> act_dependents_;
  std::unordered_map<ActivationNode*, std::vector<OperationNode*>> cache_dependents_;

  bool is_retained_graph_output(ActivationNode* node) {
    return std::find(graph.outputs().begin(), graph.outputs().end(), node) != graph.outputs().end();
  }

  bool is_retained_graph_input(ActivationNode* node) {
    return std::find(graph.inputs().begin(), graph.inputs().end(), node) != graph.inputs().end();
  }

  size_t expected_grad_contributions(ActivationNode* node) {
    return act_dependents_.at(node).size() + (is_retained_graph_output(node) ? 1 : 0);
  }

public:
  GraphExecutor(Graph& graph)
      : graph(graph) {
    for (auto& [uuid, act_node] : graph.act_nodes()) {
      activation_data_[&act_node] = nullptr;
      gradient_data_[&act_node] = nullptr;
      forward_uses_[&act_node] = 0;
      cache_uses_[&act_node] = 0;
      grad_contributions_[&act_node] = 0;
      grad_uses_[&act_node] = 0;
    }

    for (auto& [uuid, op_node] : graph.op_nodes()) {
      workspace_[&op_node] = nullptr;
      residual_[&op_node] = nullptr;

      for (ActivationNode* dep : op_node.inputs()) {
        act_dependents_[dep].push_back(&op_node);
      }
      for (ActivationNode* out : op_node.outputs()) {
        act_deps_[out] = &op_node;
      }
      for (ActivationNode* cached : op_node.cache()) {
        cache_dependents_[cached].push_back(&op_node);
      }
    }
  }

  void init_boundaries(Allocator* allocator) {
    for (ActivationNode* input : graph.inputs()) {
      forward_uses_[input] = act_dependents_[input].size();
      cache_uses_[input] = cache_dependents_[input].size();
      activation_data_[input] = Tensor(allocator, input->size());
    }
  }

  void transition_to_backward(Allocator* allocator) {
    for (ActivationNode* output : graph.outputs()) {
      if (forward_uses_[output] == 0 && cache_uses_[output] == 0) {
        activation_data_[output] = nullptr;
      }

      if (gradient_data_[output]) {
        throw std::runtime_error("Output gradient is already initialized.");
      }

      gradient_data_[output] = Tensor(allocator, output->size());

      // External/loss gradient contribution.
      grad_contributions_[output]++;

      // The gradient is consumed by the unique producer's backward op.
      grad_uses_[output] = act_deps_.count(output) ? 1 : 0;
    }
  }

  void run_op_node(const OperationNode* node, Allocator* allocator, bool save_residual = true) {
    if (!node) throw std::invalid_argument("GraphExecutor::run_op_node: Node is null.");

    for (ActivationNode* input : node->inputs()) {
      if (!activation_data_[input]) throw std::runtime_error("Input tensor is not available.");
    }

    if (node->workspace_req() > 0) {
      workspace_[node] = Tensor(allocator, node->workspace_req());
    }

    if (save_residual && node->residual_mem() > 0) {
      residual_[node] = Tensor(allocator, node->residual_mem());
    }

    for (ActivationNode* output : node->outputs()) {
      if (activation_data_[output])
        throw std::runtime_error("Output tensor is already allocated: " + output->uuid());
      forward_uses_[output] = act_dependents_[output].size();
      cache_uses_[output] = cache_dependents_[output].size();
      activation_data_[output] = Tensor(allocator, output->size());

      if (forward_uses_[output] == 0 && cache_uses_[output] == 0 &&
          !is_retained_graph_output(output)) {
        activation_data_[output] = nullptr;
      }
    }

    for (ActivationNode* input : node->inputs()) {
      if (forward_uses_[input] > 0) {
        forward_uses_[input]--;
        if (forward_uses_[input] == 0 && cache_uses_[input] == 0 &&
            !is_retained_graph_output(input)) {
          activation_data_[input] = nullptr;
        }
      }
    }

    workspace_[node] = nullptr;
  }

  void run_backward_op_node(const OperationNode* node, Allocator* allocator) {
    if (!node) throw std::invalid_argument("GraphExecutor::run_backward_op_node: Node is null.");

    for (ActivationNode* output : node->outputs()) {
      if (!gradient_data_[output]) {
        throw std::runtime_error("Output gradient is unavailable: " + output->uuid());
      }

      const size_t expected = expected_grad_contributions(output);

      if (grad_contributions_[output] != expected) {
        throw std::runtime_error("Output gradient is incomplete: " + output->uuid());
      }
    }

    for (ActivationNode* cached : node->cache()) {
      if (!activation_data_[cached]) throw std::runtime_error("Cached activation is unavailable.");
    }

    if (node->residual_mem() > 0 && !residual_[node]) {
      throw std::runtime_error("Backward residual is unavailable for operation: " + node->uuid());
    }

    if (node->workspace_req() > 0) {
      workspace_[node] = Tensor(allocator, node->workspace_req());
    }

    for (ActivationNode* input : node->inputs()) {
      if (!gradient_data_[input]) {
        gradient_data_[input] = Tensor(allocator, input->size());
        grad_uses_[input] = act_deps_.count(input) ? 1 : 0;
        if (is_retained_graph_input(input)) {
          grad_uses_[input]++;
        }
      }
      grad_contributions_[input]++;
    }

    for (ActivationNode* cached : node->cache()) {
      if (cache_uses_[cached] == 0)
        throw std::runtime_error("Invalid cached-input reference count.");
      cache_uses_[cached]--;
      if (cache_uses_[cached] == 0 && forward_uses_[cached] == 0) {
        activation_data_[cached] = nullptr;
      }
    }

    for (ActivationNode* output : node->outputs()) {
      if (grad_uses_[output] > 0) {
        grad_uses_[output]--;
        if (grad_uses_[output] == 0) {
          gradient_data_[output] = nullptr;
        }
      }
    }

    if (node->residual_mem() > 0) {
      residual_[node] = nullptr;
    }

    workspace_[node] = nullptr;
  }

  void undo_run_op_node(const OperationNode* node, Allocator* allocator) {
    if (!node) throw std::invalid_argument("GraphExecutor::undo_run_op_node: Node is null.");

    for (ActivationNode* output : node->outputs()) {
      forward_uses_[output] = 0;
      cache_uses_[output] = 0;
      activation_data_[output] = nullptr;
    }

    if (node->residual_mem() > 0) {
      residual_[node] = nullptr;
    }

    for (ActivationNode* input : node->inputs()) {
      if (forward_uses_[input] == 0 && cache_uses_[input] == 0 &&
          !is_retained_graph_output(input)) {
        activation_data_[input] = Tensor(allocator, input->size());
      }
      forward_uses_[input]++;
    }
  }

  void undo_run_backward_op_node(const OperationNode* node, Allocator* allocator) {
    if (!node)
      throw std::invalid_argument("GraphExecutor::undo_run_backward_op_node: Node is null.");

    for (ActivationNode* input : node->inputs()) {
      if (grad_contributions_[input] == 0)
        throw std::runtime_error("Output tensor is not allocated.");
      grad_contributions_[input]--;
      if (grad_contributions_[input] == 0) {
        gradient_data_[input] = nullptr;
        grad_uses_[input] = 0;
      }
    }

    for (ActivationNode* output : node->outputs()) {
      if (grad_uses_[output] == 0) {
        gradient_data_[output] = Tensor(allocator, output->size());
      }
      grad_uses_[output]++;
    }

    if (node->residual_mem() > 0) {
      residual_[node] = Tensor(allocator, node->residual_mem());
    }

    for (ActivationNode* cached : node->cache()) {
      if (cache_uses_[cached] == 0 && forward_uses_[cached] == 0) {
        activation_data_[cached] = Tensor(allocator, cached->size());
      }
      cache_uses_[cached]++;
    }
  }
  void reset() {
    for (auto& [uuid, act_node] : graph.act_nodes()) {
      activation_data_[&act_node] = nullptr;
      gradient_data_[&act_node] = nullptr;
      forward_uses_[&act_node] = 0;
      cache_uses_[&act_node] = 0;
      grad_contributions_[&act_node] = 0;
      grad_uses_[&act_node] = 0;
    }

    for (auto& [uuid, op_node] : graph.op_nodes()) {
      workspace_[&op_node] = nullptr;
      residual_[&op_node] = nullptr;
    }
  }
};
