/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/graph_executor.hpp"

#include <fmt/core.h>

#include "nn/edge.hpp"
#include "nn/edge_profile.hpp"
#include "nn/execution_plan.hpp"
#include "nn/graph.hpp"
#include "nn/macro_solver.hpp"
#include "nn/tensor_bundle.hpp"
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

GraphExecutor::GraphExecutor(Graph &graph)
    : graph_(graph) {
  for (const auto &edge : graph_.edges()) {
    for (const auto &producer : edge->producers()) {
      ++data_ref_counts_[producer];
    }
    for (const auto &consumer : edge->consumers()) {
      ++grad_ref_counts_[consumer];
    }
  }
  for (const auto &output : graph_.outputs()) {
    ++data_ref_counts_[output];
  }
}

const Tensor &GraphExecutor::data(const Node &node) const { return data_.at(node).tensor; }

void GraphExecutor::set_data(const Node &node, const Tensor &tensor, int ref_count) {
  data_[node] = {tensor, ref_count};
}

void GraphExecutor::release_data(const Node &node) {
  auto it = data_.find(node);
  if (it != data_.end() && it->second.ref_count > 0 && --it->second.ref_count == 0) {
    data_.erase(it);
  }
}

const Tensor &GraphExecutor::grad(const Node &node) const { return grads_.at(node).tensor; }

void GraphExecutor::set_grad(const Node &node, const Tensor &tensor, int ref_count) {
  grads_[node] = {tensor, ref_count};
}

void GraphExecutor::accumulate_grad(const Node &node, const Tensor &tensor, int ref_count) {
  auto [it, inserted] = grads_.try_emplace(node, Entry{tensor, ref_count});
  if (!inserted) {
    it->second.tensor += tensor;
  }
}

void GraphExecutor::release_grad(const Node &node) {
  auto it = grads_.find(node);
  if (it != grads_.end() && it->second.ref_count > 0 && --it->second.ref_count == 0) {
    grads_.erase(it);
  }
}

void GraphExecutor::cleanup_released(std::map<Node, Entry> &entries) {
  for (auto it = entries.begin(); it != entries.end();) {
    if (it->second.ref_count == 0) {
      it = entries.erase(it);
    } else {
      ++it;
    }
  }
}

TensorBundle GraphExecutor::forward(TensorBundle &input_map) {
  std::map<std::string, Node> uid_to_node;
  for (const auto &node : graph_.nodes()) {
    uid_to_node[node->uid()] = node;
  }
  PlanKey key;
  for (const auto &[uid, tensor] : input_map) {
    auto &node = uid_to_node.at(uid);
    key.input_shapes[node] = tensor.shape();
  }

  auto it = plans_.find(key);
  if (it == plans_.end()) {
    MacroSolver planner(graph_, os_);
    std::map<Edge, EdgeProfile> edge_profiles = profile_forward(input_map);
    ExecutionPlan plan = planner.find_order(edge_profiles);
    it = plans_.emplace(key, plan).first;
  }
  active_plan_ = it->second;

  for (const auto &[uid, tensor] : input_map) {
    auto it = uid_to_node.find(uid);
    if (it == uid_to_node.end()) {
      throw std::runtime_error("Input UID not found in graph: " + uid);
    }
    Tensor device_tensor = tensor;
    if (tensor.device() != graph_.device()) {
      device_tensor = to_device(tensor, graph_.device(), graph_.handle().get_stream());
    }
    set_data(it->second, device_tensor, data_ref_counts_[it->second]);
  }

  TensorBundle output_map;
  for (Edge &edge : active_plan_.order) {
    forward_edge(edge);

    for (const Node &consumer : edge->consumers()) {
      if (graph_.is_output(consumer)) {
        output_map.set(consumer->uid(), data(consumer));
        release_data(consumer);
      }
    }
  }

  cleanup_released(data_);
  return output_map;
}

TensorBundle GraphExecutor::backward(TensorBundle &output_grad_map) {
  std::map<std::string, Node> uid_to_node;
  for (const auto &node : graph_.nodes()) {
    uid_to_node[node->uid()] = node;
  }
  for (const auto &[uid, tensor] : output_grad_map) {
    auto it = uid_to_node.find(uid);
    if (it == uid_to_node.end()) {
      throw std::runtime_error("Output UID not found in graph: " + uid);
    }
    Tensor device_tensor = tensor;
    if (tensor.device() != graph_.device()) {
      device_tensor = to_device(tensor, graph_.device(), graph_.handle().get_stream());
    }
    set_grad(it->second, device_tensor, grad_ref_counts_[it->second]);
  }

  TensorBundle grad_input_map;

  for (auto it = active_plan_.order.rbegin(); it != active_plan_.order.rend(); ++it) {
    Edge &edge = *it;
    backward_edge(edge);

    for (const auto &producer : edge->producers()) {
      if (graph_.is_input(producer)) {
        grad_input_map.set(producer->uid(), grad(producer));
        release_grad(producer);
      }
    }
  }

  cleanup_released(grads_);
  return grad_input_map;
}

std::map<Edge, EdgeProfile> GraphExecutor::profile_forward(TensorBundle &input_map) {
  std::map<Edge, EdgeProfile> edge_profiles;
  std::map<std::string, Node> uid_to_node;
  for (const auto &node : graph_.nodes()) {
    uid_to_node[node->uid()] = node;
  }
  for (const auto &[uid, tensor] : input_map) {
    auto it = uid_to_node.find(uid);
    if (it == uid_to_node.end()) {
      throw std::runtime_error("Input UID not found in graph: " + uid);
    }
    Tensor device_tensor = tensor;
    if (tensor.device() != graph_.device()) {
      device_tensor = to_device(tensor, graph_.device(), graph_.handle().get_stream());
    }
    set_data(it->second, device_tensor, data_ref_counts_[it->second]);
  }

  TensorBundle output_map;  // placeholder to ensure outputs arent prematurely deallocated

  // assuming sorted topologically
  for (const Edge &edge : graph_.edges()) {
    EdgeProfile profile = forward_edge(edge);
    edge_profiles[edge] = profile;
    for (const Node &consumer : edge->consumers()) {
      if (graph_.is_output(consumer)) {
        output_map.set(consumer->uid(), data(consumer));
      }
    }
  }

  // free memory for residuals since this is just profiling
  residuals_.clear();

  return edge_profiles;
}

ExecutionPlanStats GraphExecutor::profile_plan(TensorBundle &input_map, const ExecutionPlan &plan) {
  std::map<std::string, Node> uid_to_node;
  for (const auto &node : graph_.nodes()) {
    uid_to_node[node->uid()] = node;
  }
  for (const auto &[uid, tensor] : input_map) {
    auto it = uid_to_node.find(uid);
    if (it == uid_to_node.end()) {
      throw std::runtime_error("Input UID not found in graph: " + uid);
    }
    Tensor device_tensor = tensor;
    if (tensor.device() != graph_.device()) {
      device_tensor = to_device(tensor, graph_.device(), graph_.handle().get_stream());
    }
    set_data(it->second, device_tensor, data_ref_counts_[it->second]);
  }

  TensorBundle output_map;  // placeholder to ensure outputs arent prematurely deallocated

  auto *allocator = graph_.workspace_allocator();
  size_t peak_usage = 0;
  const size_t hook_id = allocator->add_allocation_hook(
      [&peak_usage](size_t usage) { peak_usage = std::max(peak_usage, usage); });

  ExecutionPlanStats stats;

  // assuming sorted topologically
  for (const Edge &edge : plan.order) {
    forward_edge(edge);

    EdgeMemStats edge_stat;
    edge_stat.layer_name = edge->layer()->name();
    edge_stat.allocated_mem = allocator->allocated();
    edge_stat.peak_mem = peak_usage;
    stats.edge_stats.push_back(edge_stat);

    for (const Node &consumer : edge->consumers()) {
      if (graph_.is_output(consumer)) {
        output_map.set(consumer->uid(), data(consumer));
      }
    }
  }

  allocator->remove_allocation_hook(hook_id);

  // free memory for residuals since this is just profiling
  residuals_.clear();

  stats.peak_mem = stats.edge_stats.size() > 0 ? stats.edge_stats.back().peak_mem : 0;

  return stats;
}

EdgeProfile GraphExecutor::forward_edge(const Edge &edge) {
  auto *allocator = graph_.workspace_allocator();
  Vec<Tensor> inputs;
  for (const auto &producer : edge->producers()) {
    if (!data(producer)) {
      throw std::runtime_error("Null input data while forwarding graph");
    }
    inputs.push_back(data(producer));
    release_data(producer);
  }
  Residuals residuals;
  Vec<Tensor> output_data;
  const size_t usage_before = allocator->allocated();
  size_t peak_usage = usage_before;
  const size_t hook_id = allocator->add_allocation_hook(
      [&peak_usage](size_t usage) { peak_usage = std::max(peak_usage, usage); });
  auto start = std::chrono::high_resolution_clock::now();
  output_data = edge->layer()->forward(inputs, residuals);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  const size_t usage_after = allocator->allocated();
  allocator->remove_allocation_hook(hook_id);

  EdgeProfile profile;
  profile.exec_time = elapsed.count();
  profile.total_mem = peak_usage - usage_before;
  profile.workspace_mem = peak_usage - usage_after;
  profile.output_mem = 0;
  for (const Tensor &output : output_data) {
    profile.output_mem += output.num_bytes();
  }
  profile.input_mem = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const Tensor &input = inputs[i];
    profile.input_mem += input.num_bytes();
    for (const auto &[name, residual] : residuals.tensors()) {
      if (input.data_as<void>() == residual.data_as<void>()) {
        profile.cached_inputs.push_back(edge->producers()[i]->uid());
        break;
      }
    }
  }
  for (const auto &[name, residual] : residuals.tensors()) {
    bool is_input_or_output = false;
    for (Tensor &input : inputs) {
      if (input.data_as<void>() == residual.data_as<void>()) {
        is_input_or_output = true;
        break;
      }
    }
    for (const Tensor &output : output_data) {
      if (output.data_as<void>() == residual.data_as<void>()) {
        is_input_or_output = true;
        break;
      }
    }
    if (is_input_or_output) {
      continue;
    }
    profile.secondary_mem += residual.num_bytes();
  }

  residuals_[edge] = std::move(residuals);
  for (size_t index = 0; index < edge->consumers().size(); ++index) {
    const Node &consumer = edge->consumers()[index];
    set_data(consumer, output_data[index], data_ref_counts_[consumer]);
  }
  inputs = Vec<Tensor>();
  output_data = Vec<Tensor>();
  profile.net_mem = allocator->allocated() - usage_before;
  return profile;
}

EdgeProfile GraphExecutor::backward_edge(const Edge &edge) {
  auto *allocator = graph_.workspace_allocator();
  Vec<Tensor> grad_outputs;
  for (const auto &consumer : edge->consumers()) {
    if (!grad(consumer)) {
      throw std::runtime_error("Null output gradient while backwarding graph");
    }
    grad_outputs.push_back(grad(consumer));
    release_grad(consumer);
  }
  auto residuals_it = residuals_.find(edge);
  if (residuals_it == residuals_.end()) {
    throw std::runtime_error("Residuals not found for the given graph executor");
  }
  const size_t usage_before = allocator->allocated();
  size_t peak_usage = usage_before;
  const size_t hook_id = allocator->add_allocation_hook(
      [&peak_usage](size_t usage) { peak_usage = std::max(peak_usage, usage); });
  auto start = std::chrono::high_resolution_clock::now();
  Vec<Tensor> grad_inputs = edge->layer()->backward(grad_outputs, residuals_it->second);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  const size_t usage_after = allocator->allocated();
  allocator->remove_allocation_hook(hook_id);
  residuals_.erase(residuals_it);

  for (size_t index = 0; index < edge->producers().size(); ++index) {
    const Node &producer = edge->producers()[index];
    accumulate_grad(producer, grad_inputs[index], grad_ref_counts_[producer]);
  }
  EdgeProfile profile;
  profile.exec_time = elapsed.count();
  profile.total_mem = peak_usage - usage_before;
  profile.workspace_mem = peak_usage - usage_after;
  profile.input_mem = 0;
  for (const Tensor &grad_output : grad_outputs) {
    profile.input_mem += grad_output.num_bytes();
  }
  profile.output_mem = 0;
  for (const Tensor &grad_input : grad_inputs) {
    profile.output_mem += grad_input.num_bytes();
  }
  profile.secondary_mem = 0;     // none since backward
  grad_outputs = Vec<Tensor>();  // free inputs;
  profile.net_mem = allocator->allocated() - usage_before;
  return profile;
}

}  // namespace tunx