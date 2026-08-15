#pragma once

#include <algorithm>
#include <stdexcept>

#include "allocator.h"
#include "graph.h"
#include "tensor.h"

class GraphExecutor {
private:
  Graph& graph;
  std::unordered_map<ActivationNode*, Tensor> data_;
  std::unordered_map<const OperationNode*, Tensor> workspace_;
  std::unordered_map<ActivationNode*, size_t> ref_count_;
  std::unordered_map<ActivationNode*, size_t> grad_accumulated_count_;
  std::unordered_map<ActivationNode*, OperationNode*>
      act_deps_;  // which op node create this activation node
  std::unordered_map<ActivationNode*, std::vector<OperationNode*>>
      act_dependents_;  // which op nodes depends on this activation node
  std::unordered_map<const OperationNode*, Tensor> residual_;

public:
  GraphExecutor(Graph& graph)
      : graph(graph) {
    for (auto& [uuid, act_node] : graph.act_nodes()) {
      ref_count_[&act_node] = 0;
      grad_accumulated_count_[&act_node] = 0;
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

  void init_boundaries_backward(Allocator* allocator) {
    for (ActivationNode* output : graph.outputs()) {
      ref_count_[output] = act_deps_.count(output) ? 1 : 0;
      data_[output] = Tensor(allocator, output->size());
    }

    for (ActivationNode* input : graph.inputs()) {
      ref_count_[input]++;  // increment for input since they need to be retained for extraction.
    }
  }

  void transition_to_backward(Allocator* allocator) {
    for (ActivationNode* output : graph.outputs()) {
      data_[output] = nullptr;
    }
    init_boundaries_backward(allocator);
  }

  void run_op_node(const OperationNode* node, Allocator* allocator, bool save_residual = true) {
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

    if (save_residual && node->residual_mem() > 0) {
      residual_[node] = Tensor(allocator, node->residual_mem());
    }

    // allocate outputs
    for (ActivationNode* output : node->outputs()) {
      if (data_[output]) {
        throw std::runtime_error("GraphExecutor::run_op_node: Output tensor is already allocated.");
      }
      ref_count_[output] += act_dependents_[output].size();
      data_[output] = Tensor(allocator, output->size());
      if (ref_count_[output] == 0) {
        data_[output] = nullptr;
      }
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

  void run_backward_op_node(const OperationNode* node, Allocator* allocator) {
    // sanity check
    if (!node) {
      throw std::invalid_argument("GraphExecutor::run_backward_op_node: Node is null.");
    }
    for (ActivationNode* output : node->outputs()) {
      if (!data_[output]) {
        throw std::runtime_error("Backward input tensor (forward output) is not available.");
      }
    }

    // allocate workspace
    if (node->workspace_req() > 0) {
      workspace_[node] = Tensor(allocator, node->workspace_req());
    }

    // allocate grad_inputs (forward inputs)
    for (ActivationNode* input : node->inputs()) {
      if (grad_accumulated_count_[input] == 0) {
        data_[input] = Tensor(allocator, input->size());
        ref_count_[input] = (act_deps_.count(input) > 0 ? 1 : 0);
        if (std::find(graph.inputs().begin(), graph.inputs().end(), input) != graph.inputs().end()) {
          ref_count_[input]++;
        }
      }
      grad_accumulated_count_[input]++;
    }

    // execute some stuffs in real code here but we will skip since this is simulation

    // free grad_outputs (forward outputs)
    for (ActivationNode* output : node->outputs()) {
      if (ref_count_[output] > 0) {
        ref_count_[output]--;
        if (ref_count_[output] == 0) {
          data_[output] = nullptr;
        }
      }
    }

    if (node->residual_mem() > 0) {
      residual_[node] = Tensor();
    }

    // free workspace
    workspace_[node] = Tensor();
  }

  void undo_run_op_node(const OperationNode* node, Allocator* allocator) {
    // sanity check
    if (!node) {
      throw std::invalid_argument("GraphExecutor::undo_run_op_node: Node is null.");
    }

    // free outputs
    for (ActivationNode* output : node->outputs()) {
      if (!data_[output] && ref_count_[output] != 0) {
        throw std::runtime_error(
            "GraphExecutor::undo_run_op_node: Output tensor is not allocated.");
      }
      ref_count_[output] -= act_dependents_[output].size();
      data_[output] = nullptr;
    }

    if (node->residual_mem() > 0) {
      residual_[node] = Tensor();
    }

    // allocate inputs
    for (ActivationNode* input : node->inputs()) {
      if (ref_count_[input] == 0) {
        data_[input] = Tensor(allocator, input->size());
      }
      ref_count_[input]++;
    }
  }

  void undo_run_backward_op_node(const OperationNode* node, Allocator* allocator) {
    // sanity check
    if (!node) {
      throw std::invalid_argument("GraphExecutor::undo_run_backward_op_node: Node is null.");
    }

    // free grad_inputs (forward inputs)
    for (ActivationNode* input : node->inputs()) {
      if (grad_accumulated_count_[input] == 0) {
        throw std::runtime_error("GraphExecutor::undo_run_backward_op_node: Output tensor is not allocated.");
      }
      grad_accumulated_count_[input]--;
      if (grad_accumulated_count_[input] == 0) {
        data_[input] = nullptr;
        ref_count_[input] = 0;
      }
    }

    // allocate grad_outputs (forward outputs)
    for (ActivationNode* output : node->outputs()) {
      if (ref_count_[output] == 0) {
        data_[output] = Tensor(allocator, output->size());
      }
      ref_count_[output]++;
    }

    if (node->residual_mem() > 0) {
      residual_[node] = Tensor(allocator, node->residual_mem());
    }
  }
};
