/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/graph_executor.hpp"

#include <fmt/core.h>

#include <chrono>
#include <unordered_map>

#include "nn/graph.hpp"
#include "nn/macro_solver.hpp"
#include "tensor/ops.hpp"

namespace tunx {

GraphExecutor::GraphExecutor(Graph &graph)
    : graph_(graph) {
  for (const auto &edge : graph_.edges_) {
    for (const auto &producer : edge->producers()) {
      ++data_ref_counts_[producer];
    }
    for (const auto &consumer : edge->consumers()) {
      ++grad_ref_counts_[consumer];
    }
  }
  for (const auto &output : graph_.output_nodes_) {
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
    it->second.tensor = Tensor();
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
    it->second.tensor = Tensor();
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
  for (const auto &node : graph_.nodes_) {
    uid_to_node[node->uid()] = node;
  }
  for (const auto &[uid, tensor] : input_map) {
    auto it = uid_to_node.find(uid);
    if (it == uid_to_node.end()) {
      throw std::runtime_error("Input UID not found in graph: " + uid);
    }
    Tensor device_tensor = tensor;
    if (tensor.device() != graph_.device()) {
      device_tensor = to_device(tensor, graph_.device(), graph_.engine_handle_.get_stream());
    }
    set_data(it->second, device_tensor, data_ref_counts_[it->second]);
  }

  MacroSolver planner(graph_);
  const std::vector<size_t> execution_order = planner.find_order(input_map);
  active_plan_ = &graph_.forward_plan_cache(input_map);
  graph_.last_forward_execution_order_ = execution_order;
  graph_.last_forward_used_macro_plan_ = active_plan_->profiled;

  TensorBundle output_map;
  for (size_t edge_index : execution_order) {
    Edge &edge = graph_.edges_.at(edge_index);
    const auto start = std::chrono::high_resolution_clock::now();
    forward_edge(edge, edge_index);
    const auto end = std::chrono::high_resolution_clock::now();
    const double duration = std::chrono::duration<double, std::milli>(end - start).count();
    const std::string timing_key = edge->layer()->name() + " forward";
    if (graph_.timing_map_.find(timing_key) == graph_.timing_map_.end()) {
      graph_.timing_order_.push_back(timing_key);
    }
    graph_.timing_map_[timing_key] += duration;

    for (const auto &consumer : edge->consumers()) {
      if (graph_.is_output(consumer)) {
        output_map.set(consumer->uid(), data(consumer));
      }
    }
  }

  active_plan_->profiled = active_plan_->edge_profiles.size() == graph_.edges_.size();
  ++active_plan_->execution_count;
  active_plan_ = nullptr;
  cleanup_released(data_);
  return output_map;
}

TensorBundle GraphExecutor::backward(TensorBundle &output_grad_map) {
  std::map<std::string, Node> uid_to_node;
  for (const auto &node : graph_.nodes_) {
    uid_to_node[node->uid()] = node;
  }
  for (const auto &[uid, tensor] : output_grad_map) {
    auto it = uid_to_node.find(uid);
    if (it == uid_to_node.end()) {
      throw std::runtime_error("Output UID not found in graph: " + uid);
    }
    Tensor device_tensor = tensor;
    if (tensor.device() != graph_.device()) {
      device_tensor = to_device(tensor, graph_.device(), graph_.engine_handle_.get_stream());
    }
    set_grad(it->second, device_tensor, grad_ref_counts_[it->second]);
  }

  TensorBundle grad_input_map;
  for (auto it = graph_.last_forward_execution_order_.rbegin();
       it != graph_.last_forward_execution_order_.rend(); ++it) {
    Edge &edge = graph_.edges_.at(*it);
    const auto start = std::chrono::high_resolution_clock::now();
    backward_edge(edge);
    const auto end = std::chrono::high_resolution_clock::now();
    const double duration = std::chrono::duration<double, std::milli>(end - start).count();
    const std::string timing_key = edge->layer()->name() + " backward";
    if (graph_.timing_map_.find(timing_key) == graph_.timing_map_.end()) {
      graph_.timing_order_.push_back(timing_key);
    }
    graph_.timing_map_[timing_key] += duration;

    for (const auto &producer : edge->producers()) {
      if (graph_.is_input(producer)) {
        grad_input_map.set(producer->uid(), grad(producer));
      }
    }
  }

  cleanup_released(grads_);
  return grad_input_map;
}

void GraphExecutor::clear_grads() { grads_.clear(); }

void GraphExecutor::forward_edge(Edge &edge, size_t edge_index) {
  Vec<Tensor> input_data;
  for (const auto &producer : edge->producers()) {
    if (!data(producer)) {
      throw std::runtime_error("Null input data while forwarding graph");
    }
    input_data.push_back(data(producer));
    release_data(producer);
  }

  Residuals residuals;
  Vec<Tensor> output_data;
  size_t workspace_bytes = 0;
  if (graph_.workspace_allocator_) {
    const size_t usage_before = graph_.workspace_allocator_->allocated();
    size_t peak_usage = usage_before;
    const size_t hook_id = graph_.workspace_allocator_->add_allocation_hook(
        [&peak_usage](size_t usage) { peak_usage = std::max(peak_usage, usage); });
    output_data = edge->layer()->forward(input_data, residuals);
    const size_t usage_after = graph_.workspace_allocator_->allocated();
    graph_.workspace_allocator_->remove_allocation_hook(hook_id);

    const size_t peak_edge_usage = peak_usage - usage_before;
    const size_t retained = usage_after - usage_before;
    workspace_bytes = peak_usage > usage_after ? peak_usage - usage_after : 0;

    if (graph_.enable_memory_profiling_ && graph_.memory_profile_logger_) {
      std::unordered_map<std::string, std::string> row;
      row["layer"] = edge->layer()->name();
      row["peak_usage_bytes"] = std::to_string(peak_edge_usage);
      row["retained_bytes"] = std::to_string(retained);
      row["unused_bytes"] = std::to_string(graph_.workspace_allocator_->unused());
      row["reserved_bytes"] = std::to_string(graph_.workspace_allocator_->reserved());
      graph_.memory_profile_logger_->log(row);
    } else if (graph_.enable_memory_profiling_) {
      fmt::print(
          "Layer {} peak usage: {:.2f} MB, retained: {:.2f} MB, unused: {:.2f} MB, reserved: "
          "{:.2f} MB\n",
          edge->layer()->name(), static_cast<double>(peak_edge_usage) / 1024 / 1024,
          static_cast<double>(retained) / 1024 / 1024,
          static_cast<double>(graph_.workspace_allocator_->unused()) / 1024 / 1024,
          static_cast<double>(graph_.workspace_allocator_->reserved()) / 1024 / 1024);
    }
  } else {
    output_data = edge->layer()->forward(input_data, residuals);
  }

  size_t output_bytes = 0;
  Vec<size_t> output_tensor_bytes;
  output_tensor_bytes.reserve(output_data.size());
  for (const Tensor &output : output_data) {
    output_tensor_bytes.push_back(output.num_bytes());
    output_bytes += output.num_bytes();
  }
  if (active_plan_) {
    active_plan_->edge_profiles[edge_index] = {
        .peak_bytes = workspace_bytes,
        .retained_bytes = residuals.num_bytes(),
        .output_bytes = output_bytes,
        .output_tensor_bytes = std::move(output_tensor_bytes),
    };
  }
  residuals_[edge] = std::move(residuals);
  for (size_t index = 0; index < edge->consumers().size(); ++index) {
    const Node &consumer = edge->consumers()[index];
    set_data(consumer, output_data[index], data_ref_counts_[consumer]);
  }
}

void GraphExecutor::backward_edge(Edge &edge) {
  Vec<Tensor> output_grads;
  for (const auto &consumer : edge->consumers()) {
    if (!grad(consumer)) {
      throw std::runtime_error("Null output gradient while backwarding graph");
    }
    output_grads.push_back(grad(consumer));
    release_grad(consumer);
  }
  auto residuals_it = residuals_.find(edge);
  if (residuals_it == residuals_.end()) {
    throw std::runtime_error("Residuals not found for the given graph executor");
  }
  Vec<Tensor> grad_inputs = edge->layer()->backward(output_grads, residuals_it->second);
  residuals_.erase(residuals_it);

  for (size_t index = 0; index < edge->producers().size(); ++index) {
    const Node &producer = edge->producers()[index];
    accumulate_grad(producer, grad_inputs[index], grad_ref_counts_[producer]);
  }
}

}  // namespace tunx