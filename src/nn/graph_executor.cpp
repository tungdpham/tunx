/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/graph_executor.hpp"

#include <fmt/core.h>

#include <ostream>

#include "device/device_allocator.hpp"
#include "device/device_manager.hpp"
#include "device/pool_allocator.hpp"
#include "nn/edge.hpp"
#include "nn/edge_profile.hpp"
#include "nn/execution_plan.hpp"
#include "nn/graph.hpp"
#include "nn/macro_solver.hpp"
#include "nn/memory_packer.hpp"
#include "nn/tensor_bundle.hpp"
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

GraphExecutor::GraphExecutor(Graph &graph)
    : graph_(graph),
      host_allocator_(std::make_unique<OffloadAllocator>(&DeviceAllocator::instance(getHost()))) {
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
  for (const auto &input : graph_.inputs()) {
    ++grad_ref_counts_[input];
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

const BuiltPlan &GraphExecutor::build_plans(TensorBundle &input_map, SolverOptions options,
                                            bool bootstrap_offload) {
  std::map<std::string, Node> uid_to_node;
  for (const auto &node : graph_.nodes()) {
    uid_to_node[node->uid()] = node;
  }
  bool is_training = false;
  for (const auto &edge : graph_.edges()) {
    if (edge->layer() && edge->layer()->is_training()) {
      is_training = true;
      break;
    }
  }

  PlanKey key;
  key.enable_naive = options.enable_naive;
  key.enable_linear = options.enable_linear;
  key.enable_branching = options.enable_branching;
  key.enable_joining = options.enable_joining;
  key.is_training = is_training;
  key.device = graph_.device().get_id();

  for (const auto &[uid, tensor] : input_map) {
    auto node = uid_to_node[uid];
    key.input_shapes[node] = tensor.shape();
    key.input_dtypes[node] = tensor.dtype();
  }
  if (built_plans_.count(key)) {
    return built_plans_.at(key);
  }
  MacroSolver planner(graph_, os_, options);
  auto [output_map, forward_edge_profiles, node_profiles] = profile_edges_forward(input_map, true);
  ExecutionPlan forward_plan;
  if (options.enable_naive) {
    forward_plan.order = graph_.edges();
  } else {
    forward_plan = planner.find_forward_order(forward_edge_profiles);
  }

  output_map.clear();

  for (const auto &[uid, tensor] : input_map) {
    auto node = uid_to_node[uid];
    Tensor device_tensor = tensor;
    if (tensor.device() != graph_.device()) {
      device_tensor = Tensor(tensor.shape(), tensor.dtype(), *graph_.workspace_allocator());
      copy(tensor, device_tensor, graph_.handle().get_stream());
    }
    set_data(node, device_tensor, data_ref_counts_[node]);
  }

  bool should_offload = is_training && bootstrap_offload;

  auto s = graph_.handle().get_stream();
  for (const Edge &edge : forward_plan.order) {
    forward_edge(edge);

    if (should_offload) {
      std::unordered_map<void *, Tensor> offloaded_aliases;
      residuals_[edge].apply_tensors([&](Tensor &tensor) {
        if (!tensor) return;
        if (tensor.device() != getHost()) {
          void *source = tensor.data_as<void>();
          auto found = offloaded_aliases.find(source);
          if (found != offloaded_aliases.end()) {
            tensor = found->second;
            return;
          }
          Tensor offloaded(tensor.shape(), tensor.dtype(), *host_allocator_);
          copy(tensor, offloaded, s);
          s.sync();
          offloaded_aliases.emplace(source, offloaded);
          tensor = std::move(offloaded);
        }
      });
    }

    for (const Node &consumer : edge->consumers()) {
      if (graph_.is_output(consumer)) {
        output_map.set(consumer->uid(), data(consumer));
        release_data(consumer);
      }
    }
  }
  cleanup_released(data_);

  TensorBundle output_grad_map;
  auto &grad_allocator = PoolAllocator::instance(graph_.device(), graph_.handle().get_stream());
  for (const auto &[uid, tensor] : output_map) {
    Tensor t(tensor.shape(), tensor.dtype(), grad_allocator);
    // zero init so that it does not corrupt param gradients.
    fill(t, 0.0, graph_.handle().get_stream());
    output_grad_map.set(uid, t);
  }

  output_map.clear();

  graph_.workspace_allocator()->evict_unused();

  PlanKey backward_key;
  for (const auto &[uid, tensor] : output_grad_map) {
    auto node = uid_to_node[uid];
    backward_key.input_shapes[node] = tensor.shape();
    backward_key.input_dtypes[node] = tensor.dtype();
  }

  std::map<Edge, EdgeProfile> backward_edge_profiles;
  ExecutionPlan backward_plan;
  if (is_training) {
    auto profile_res = profile_edges_backward(output_grad_map, should_offload);
    backward_edge_profiles = std::move(profile_res.second);
    profile_res.first.clear();
    graph_.workspace_allocator()->evict_unused();
    if (options.enable_naive) {
      for (auto it = graph_.edges().rbegin(); it != graph_.edges().rend(); ++it) {
        backward_plan.order.push_back(*it);
      }
    } else {
      backward_plan = planner.find_backward_order(backward_edge_profiles);
    }
  }

  BuiltPlan plan{forward_plan,
                 backward_plan,
                 std::move(forward_edge_profiles),
                 std::move(backward_edge_profiles),
                 std::move(node_profiles),
                 nullptr};

  clear_residuals();
  host_allocator_->evict_unused();

  pack_memory(plan, input_map, output_grad_map);

  output_grad_map.clear();

  auto [it, inserted] = built_plans_.emplace(key, std::move(plan));

  graph_.workspace_allocator()->evict_unused();

  return it->second;
}

TensorBundle GraphExecutor::forward(TensorBundle &input_map) {
  std::map<std::string, Node> uid_to_node;
  for (const auto &node : graph_.nodes()) {
    uid_to_node[node->uid()] = node;
  }
  PlanKey key;
  key.enable_naive = false;
  key.enable_linear = true;
  key.enable_branching = true;
  key.enable_joining = true;
  for (const auto &[uid, tensor] : input_map) {
    auto &node = uid_to_node.at(uid);
    key.input_shapes[node] = tensor.shape();
  }

  auto it = built_plans_.find(key);
  if (it == built_plans_.end()) {
    it = built_plans_.emplace(key, build_plans(input_map)).first;
  }
  active_built_plan_ = it->second;
  active_built_plan_.packed_allocator->set_backend_allocator(graph_.workspace_allocator());

  for (const auto &[uid, tensor] : input_map) {
    auto it = uid_to_node.find(uid);
    if (it == uid_to_node.end()) {
      throw std::runtime_error("Input UID not found in graph: " + uid);
    }
    Tensor device_tensor = tensor;
    if (tensor.device() != graph_.device()) {
      device_tensor = Tensor(tensor.shape(), tensor.dtype(), *graph_.workspace_allocator());
      copy(tensor, device_tensor, graph_.handle().get_stream());
    }
    set_data(it->second, device_tensor, data_ref_counts_[it->second]);
  }
  for (auto &edge : graph_.edges()) {
    if (edge->layer())
      edge->layer()->set_workspace_allocator(active_built_plan_.packed_allocator.get());
  }

  TensorBundle output_map;
  for (Edge &edge : active_built_plan_.forward_plan.order) {
    active_built_plan_.packed_allocator->set_current_edge(edge->uid() + "_fw");
    forward_edge(edge);

    for (const Node &consumer : edge->consumers()) {
      if (graph_.is_output(consumer)) {
        output_map.set(consumer->uid(), data(consumer));
        release_data(consumer);
      }
    }
  }

  cleanup_released(data_);
  active_built_plan_.packed_allocator->evict_unused();  // return memory back to backend allocator
  for (auto &edge : graph_.edges()) {
    if (edge->layer()) edge->layer()->set_workspace_allocator(graph_.workspace_allocator());
  }
  return output_map;
}

TensorBundle GraphExecutor::backward(TensorBundle &output_grad_map) {
  std::map<std::string, Node> uid_to_node;
  for (const auto &node : graph_.nodes()) {
    uid_to_node[node->uid()] = node;
  }

  active_built_plan_.packed_allocator->set_backend_allocator(graph_.workspace_allocator());

  for (const auto &[uid, tensor] : output_grad_map) {
    auto it = uid_to_node.find(uid);
    if (it == uid_to_node.end()) {
      throw std::runtime_error("Output UID not found in graph: " + uid);
    }
    Tensor device_tensor = tensor;
    if (tensor.device() != graph_.device()) {
      device_tensor = Tensor(tensor.shape(), tensor.dtype(), *graph_.workspace_allocator());
      copy(tensor, device_tensor, graph_.handle().get_stream());
    }
    set_grad(it->second, device_tensor, grad_ref_counts_[it->second]);
  }
  TensorBundle grad_input_map;

  for (auto &edge : graph_.edges()) {
    if (edge->layer())
      edge->layer()->set_workspace_allocator(active_built_plan_.packed_allocator.get());
  }

  for (Edge &edge : active_built_plan_.backward_plan.order) {
    active_built_plan_.packed_allocator->set_current_edge(edge->uid() + "_bw");
    backward_edge(edge);
    residuals_.erase(edge);

    for (const auto &producer : edge->producers()) {
      if (graph_.is_input(producer)) {
        grad_input_map.set(producer->uid(), grad(producer));
        release_grad(producer);
      }
    }
  }

  cleanup_released(grads_);
  active_built_plan_.packed_allocator->evict_unused();  // return memory back to backend allocator
  for (auto &edge : graph_.edges()) {
    if (edge->layer()) edge->layer()->set_workspace_allocator(graph_.workspace_allocator());
  }
  return grad_input_map;
}

std::tuple<TensorBundle, std::map<Edge, EdgeProfile>, std::map<Node, size_t>>
GraphExecutor::profile_edges_forward(TensorBundle &input_map, bool discard_residuals) {
  std::map<Edge, EdgeProfile> edge_profiles;
  std::map<Node, size_t> node_profiles;
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
      device_tensor = Tensor(tensor.shape(), tensor.dtype(), *graph_.workspace_allocator());
      copy(tensor, device_tensor, graph_.handle().get_stream());
    }
    set_data(it->second, device_tensor, data_ref_counts_[it->second]);
    node_profiles[it->second] = device_tensor.num_bytes();
  }
  TensorBundle output_map;  // placeholder to ensure outputs arent prematurely deallocated

  // assuming sorted topologically
  for (const Edge &edge : graph_.edges()) {
    EdgeProfile profile = profile_edge_forward(edge);
    edge_profiles[edge] = profile;
    for (const Node &consumer : edge->consumers()) {
      node_profiles[consumer] = data(consumer).num_bytes();
      if (graph_.is_output(consumer)) {
        output_map.set(consumer->uid(), data(consumer));
        release_data(consumer);
      }
    }
    if (discard_residuals) {
      residuals_.erase(edge);
    }
  }
  cleanup_released(data_);
  return {output_map, edge_profiles, node_profiles};
}

ExecutionPlanStats GraphExecutor::profile_forward_plan(TensorBundle &input_map,
                                                       const ExecutionPlan &plan) {
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
      device_tensor = Tensor(tensor.shape(), tensor.dtype(), *graph_.workspace_allocator());
      copy(tensor, device_tensor, graph_.handle().get_stream());
    }
    set_data(it->second, device_tensor, data_ref_counts_[it->second]);
  }
  TensorBundle output_map;  // placeholder to ensure outputs arent prematurely deallocated

  auto *allocator = graph_.workspace_allocator();
  size_t peak_usage = 0;
  const size_t hook_id = allocator->add_allocation_hook(
      [&peak_usage](size_t usage) { peak_usage = std::max(peak_usage, usage); });

  ExecutionPlanStats stats;

  for (auto &edge : graph_.edges()) {
    if (edge->layer()) edge->layer()->set_workspace_allocator(allocator);
  }

  PackedAllocator *packed_alloc = dynamic_cast<PackedAllocator *>(allocator);

  // assuming sorted topologically
  for (const Edge &edge : plan.order) {
    if (packed_alloc) {
      packed_alloc->set_current_edge(edge->uid() + "_fw");
    }
    EdgeProfile profile = profile_edge_forward(edge);

    int64_t last_workspace_mem =
        stats.edge_stats.empty() ? 0 : stats.edge_stats.back().workspaces_mem;

    EdgeMemStats edge_stat;
    edge_stat.layer_name = edge->layer()->name();
    edge_stat.allocated_mem = allocator->allocated();
    edge_stat.reserved_mem = allocator->reserved();
    edge_stat.peak_mem = peak_usage;
    edge_stat.fragmented_mem =
        allocator->reserved() - (allocator->allocated() + allocator->unused());
    edge_stat.cached_mem = allocator->unused();
    edge_stat.activations_mem = allocator->allocated() + host_allocator_->allocated();
    edge_stat.gradients_mem = 0;
    edge_stat.host_mem = host_allocator_->allocated();
    edge_stat.workspaces_mem = std::max(last_workspace_mem, profile.workspace_mem);
    stats.edge_stats.push_back(edge_stat);

    last_workspace_mem = std::max(last_workspace_mem, profile.workspace_mem);

    for (const Node &consumer : edge->consumers()) {
      if (graph_.is_output(consumer)) {
        output_map.set(consumer->uid(), data(consumer));
        release_data(consumer);
      }
    }
  }

  allocator->remove_allocation_hook(hook_id);

  stats.peak_mem = stats.edge_stats.size() > 0 ? stats.edge_stats.back().peak_mem : 0;

  // free memory for residuals since this is just profiling
  clear_residuals();
  output_map.clear();
  cleanup_released(data_);
  allocator->evict_unused();

  return stats;
}

std::pair<TensorBundle, std::map<Edge, EdgeProfile>> GraphExecutor::profile_edges_backward(
    TensorBundle &output_grad_map, bool prefetch_residuals) {
  std::map<Edge, EdgeProfile> edge_profiles;
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
      device_tensor = Tensor(tensor.shape(), tensor.dtype(), *graph_.workspace_allocator());
      copy(tensor, device_tensor, graph_.handle().get_stream());
    }
    set_grad(it->second, device_tensor, grad_ref_counts_[it->second]);
  }
  TensorBundle grad_input_map;

  auto *allocator = graph_.workspace_allocator();

  for (auto it = graph_.edges().rbegin(); it != graph_.edges().rend(); ++it) {
    const Edge &edge = *it;

    if (prefetch_residuals) {
      residuals_[edge].apply_tensors([this](Tensor &tensor) {
        if (!tensor) return;
        if (tensor.device() != graph_.device()) {
          Tensor device_tensor(tensor.shape(), tensor.dtype(), *graph_.workspace_allocator());
          copy(tensor, device_tensor, graph_.handle().get_stream());
          tensor = device_tensor;
        }
      });
    }

    EdgeProfile profile = profile_edge_backward(edge);
    edge_profiles[edge] = profile;

    for (const auto &producer : edge->producers()) {
      if (graph_.is_input(producer)) {
        grad_input_map.set(producer->uid(), grad(producer));
        release_grad(producer);
      }
    }
  }
  cleanup_released(grads_);
  allocator->evict_unused();

  return {grad_input_map, edge_profiles};
}

ExecutionPlanStats GraphExecutor::profile_backward_plan(TensorBundle &input_map,
                                                        const ExecutionPlan &forward_plan,
                                                        const ExecutionPlan &backward_plan) {
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
      device_tensor = Tensor(tensor.shape(), tensor.dtype(), *graph_.workspace_allocator());
      copy(tensor, device_tensor, graph_.handle().get_stream());
    }
    set_data(it->second, device_tensor, data_ref_counts_[it->second]);
  }
  TensorBundle output_map;  // placeholder to ensure outputs arent prematurely deallocated

  tunx::PackedAllocator *packed_alloc_fw =
      dynamic_cast<tunx::PackedAllocator *>(graph_.workspace_allocator());
  for (const Edge &edge : forward_plan.order) {
    if (packed_alloc_fw) {
      packed_alloc_fw->set_current_edge(edge->uid() + "_fw");
    }
    forward_edge(edge);

    for (const Node &consumer : edge->consumers()) {
      if (graph_.is_output(consumer)) {
        output_map.set(consumer->uid(), data(consumer));
        release_data(consumer);
      }
    }
  }

  auto &mem_pool = PoolAllocator::instance(graph_.device(), graph_.handle().get_stream());
  TensorBundle output_grad_map;
  for (const auto &[uid, tensor] : output_map) {
    output_grad_map.set(uid, Tensor(tensor.shape(), tensor.dtype(), mem_pool));
  }

  output_map.clear();

  graph_.workspace_allocator()->evict_unused();

  for (const auto &[uid, tensor] : output_grad_map) {
    auto it = uid_to_node.find(uid);
    if (it == uid_to_node.end()) {
      throw std::runtime_error("Output UID not found in graph: " + uid);
    }
    Tensor device_tensor = tensor;
    if (tensor.device() != graph_.device()) {
      device_tensor = Tensor(tensor.shape(), tensor.dtype(), *graph_.workspace_allocator());
      copy(tensor, device_tensor, graph_.handle().get_stream());
    }
    set_grad(it->second, device_tensor, grad_ref_counts_[it->second]);
  }
  output_grad_map.clear();

  TensorBundle grad_input_map;

  auto *allocator = graph_.workspace_allocator();
  size_t peak_usage = 0;
  const size_t hook_id = allocator->add_allocation_hook(
      [&peak_usage](size_t usage) { peak_usage = std::max(peak_usage, usage); });

  ExecutionPlanStats stats;

  int64_t last_workspace_mem = 0;
  int64_t last_activations_mem = allocator->allocated();
  int64_t last_gradients_mem = 0;
  int64_t param_gradients_mem = 0;
  for (const auto &param : graph_.params()) {
    if (param.grad()) {
      param_gradients_mem += param.grad().num_bytes();
    }
  }

  for (auto &edge : graph_.edges()) {
    if (edge->layer()) edge->layer()->set_workspace_allocator(allocator);
  }

  tunx::PackedAllocator *packed_alloc = dynamic_cast<tunx::PackedAllocator *>(allocator);

  for (const Edge &edge : backward_plan.order) {
    if (packed_alloc) {
      packed_alloc->set_current_edge(edge->uid() + "_bw");
    }

    EdgeProfile profile = profile_edge_backward(edge);

    last_gradients_mem += profile.output_mem - profile.input_mem;
    // profile.net_mem in backward = outputs - inputs - residuals_freed;
    // and activations freed = residuals_freed;
    // so net activations = -(outputs-inputs-profile.net_mem)
    last_activations_mem -= profile.output_mem - profile.input_mem - profile.net_mem;

    EdgeMemStats edge_stat;
    edge_stat.layer_name = edge->layer()->name();
    edge_stat.allocated_mem = allocator->allocated();
    edge_stat.reserved_mem = allocator->reserved();
    edge_stat.peak_mem = peak_usage;
    edge_stat.fragmented_mem =
        allocator->reserved() - (allocator->allocated() + allocator->unused());
    edge_stat.cached_mem = allocator->unused();
    edge_stat.activations_mem = last_activations_mem + host_allocator_->allocated();
    edge_stat.gradients_mem = param_gradients_mem + last_gradients_mem;
    edge_stat.host_mem = host_allocator_->allocated();
    edge_stat.workspaces_mem = std::max(last_workspace_mem, profile.workspace_mem);
    stats.edge_stats.push_back(edge_stat);

    last_workspace_mem = std::max(last_workspace_mem, profile.workspace_mem);

    for (const auto &producer : edge->producers()) {
      if (graph_.is_input(producer)) {
        grad_input_map.set(producer->uid(), grad(producer));
        release_grad(producer);
      }
    }
  }

  allocator->remove_allocation_hook(hook_id);

  stats.peak_mem = stats.edge_stats.size() > 0 ? stats.edge_stats.back().peak_mem : 0;

  grad_input_map.clear();
  cleanup_released(grads_);
  allocator->evict_unused();

  return stats;
}

void GraphExecutor::forward_edge(const Edge &edge) {
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
  output_data = edge->layer()->forward(inputs, residuals);

  residuals_[edge] = std::move(residuals);
  for (size_t index = 0; index < edge->consumers().size(); ++index) {
    const Node &consumer = edge->consumers()[index];
    set_data(consumer, output_data[index], data_ref_counts_[consumer]);
  }
}

void GraphExecutor::backward_edge(const Edge &edge) {
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
  Vec<Tensor> grad_inputs = edge->layer()->backward(grad_outputs, residuals_it->second);

  for (size_t index = 0; index < edge->producers().size(); ++index) {
    const Node &producer = edge->producers()[index];
    accumulate_grad(producer, grad_inputs[index], grad_ref_counts_[producer]);
  }
}

EdgeProfile GraphExecutor::profile_edge_forward(const Edge &edge) {
  // keep a copy to check cached inputs
  std::map<Node, Tensor> inputs;
  std::map<Node, Tensor> outputs;
  for (const Node &producer : edge->producers()) {
    inputs[producer] = data(producer);
  }
  auto *allocator = edge->layer()->workspace_allocator();
  const size_t usage_before = allocator->allocated();
  size_t peak_usage = usage_before;
  const size_t hook_id = allocator->add_allocation_hook(
      [&peak_usage](size_t usage) { peak_usage = std::max(peak_usage, usage); });
  auto stream = graph_.handle().get_stream();
  std::vector<Tensor> param_snapshots;
  std::vector<Tensor> grad_snapshots;
  for (const auto &param : edge->layer()->params()) {
    if (param.size() > 0) {
      Tensor snapshot(param.shape(), param.dtype(), *graph_.workspace_allocator());
      copy(param.data(), snapshot, stream);
      param_snapshots.push_back(snapshot);
    } else {
      param_snapshots.push_back(Tensor());
    }
    if (param.size() > 0 && param.grad()) {
      Tensor grad_snapshot(param.shape(), param.dtype(), *graph_.workspace_allocator());
      copy(param.grad(), grad_snapshot, stream);
      grad_snapshots.push_back(grad_snapshot);
    } else {
      grad_snapshots.push_back(Tensor());
    }
  }

  stream.sync();
  auto start = std::chrono::high_resolution_clock::now();
  forward_edge(edge);
  stream.sync();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;

  auto params = edge->layer()->params();
  for (size_t i = 0; i < params.size(); ++i) {
    if (params[i].size() > 0) {
      copy(param_snapshots[i], params[i].data(), stream);
      if (params[i].grad()) {
        copy(grad_snapshots[i], params[i].grad(), stream);
      }
    }
  }
  stream.sync();
  const size_t usage_after = allocator->allocated();
  allocator->remove_allocation_hook(hook_id);
  for (const Node &consumer : edge->consumers()) {
    outputs[consumer] = data(consumer);
  }
  auto &residuals = residuals_.at(edge);

  EdgeProfile profile;
  profile.exec_time = elapsed.count();
  profile.total_mem = peak_usage - usage_before;
  profile.workspace_mem = peak_usage - usage_after;
  profile.output_mem = 0;
  for (const auto &[node, tensor] : outputs) {
    profile.output_mem += tensor.num_bytes();
    for (const auto &[name, residual] : residuals.tensors()) {
      if (tensor.data_as<void>() == residual.data_as<void>()) {
        profile.cached_nodes.push_back(node->uid());
        break;
      }
    }
  }
  profile.input_mem = 0;
  for (const auto &[node, tensor] : inputs) {
    profile.input_mem += tensor.num_bytes();
    for (const auto &[name, residual] : residuals.tensors()) {
      if (tensor.data_as<void>() == residual.data_as<void>()) {
        profile.cached_nodes.push_back(node->uid());
        break;
      }
    }
  }
  profile.secondary_mem = 0;
  for (const auto &[name, residual] : residuals.tensors()) {
    bool is_input = false;
    for (const auto &[node, tensor] : inputs) {
      if (tensor.data_as<void>() == residual.data_as<void>()) {
        is_input = true;
        break;
      }
    }
    bool is_output = false;
    for (const auto &[node, tensor] : outputs) {
      if (tensor.data_as<void>() == residual.data_as<void>()) {
        is_output = true;
        break;
      }
    }
    if (!is_input && !is_output) {
      profile.secondary_mem += residual.num_bytes();
    }
  }
  inputs.clear();   // free to see real net mem
  outputs.clear();  // free to see real net mem
  profile.net_mem = allocator->allocated() - usage_before;

  residuals_[edge] = std::move(residuals);
  return profile;
}

EdgeProfile GraphExecutor::profile_edge_backward(const Edge &edge) {
  std::map<Node, Tensor> grad_outputs;
  std::map<Node, Tensor> grad_inputs;
  for (const auto &consumer : edge->consumers()) {
    grad_outputs[consumer] = grad(consumer);
  }

  std::map<Node, bool> is_first_to_init;
  for (const auto &producer : edge->producers()) {
    is_first_to_init[producer] = grads_.find(producer) == grads_.end();
  }

  auto *allocator = edge->layer()->workspace_allocator();
  const size_t usage_before = allocator->allocated();
  size_t peak_usage = usage_before;
  const size_t hook_id = allocator->add_allocation_hook(
      [&peak_usage](size_t usage) { peak_usage = std::max(peak_usage, usage); });
  auto stream = graph_.handle().get_stream();
  std::vector<Tensor> param_snapshots;
  std::vector<Tensor> grad_snapshots;
  for (const auto &param : edge->layer()->params()) {
    if (param.size() > 0) {
      Tensor snapshot(param.shape(), param.dtype(), *graph_.workspace_allocator());
      copy(param.data(), snapshot, stream);
      param_snapshots.push_back(snapshot);
    } else {
      param_snapshots.push_back(Tensor());
    }
    if (param.size() > 0 && param.grad()) {
      Tensor grad_snapshot(param.shape(), param.dtype(), *graph_.workspace_allocator());
      copy(param.grad(), grad_snapshot, stream);
      grad_snapshots.push_back(grad_snapshot);
    } else {
      grad_snapshots.push_back(Tensor());
    }
  }

  stream.sync();
  auto start = std::chrono::high_resolution_clock::now();
  backward_edge(edge);
  size_t usage_after = allocator->allocated();
  allocator->remove_allocation_hook(hook_id);
  stream.sync();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;

  auto params = edge->layer()->params();
  for (size_t i = 0; i < params.size(); ++i) {
    if (params[i].size() > 0) {
      copy(param_snapshots[i], params[i].data(), stream);
      if (params[i].grad()) {
        copy(grad_snapshots[i], params[i].grad(), stream);
      }
    }
  }
  stream.sync();

  for (const auto &producer : edge->producers()) {
    grad_inputs[producer] = grad(producer);
  }

  EdgeProfile profile;
  profile.exec_time = elapsed.count();
  profile.total_mem = peak_usage - usage_before;
  profile.workspace_mem = peak_usage - usage_after;
  profile.input_mem = 0;
  for (const auto &[node, grad_output] : grad_outputs) {
    profile.input_mem += grad_output.num_bytes();
  }
  profile.output_mem = 0;
  // gradients are accumulated so only first one is newly allocated
  // so we only count the first one's memory as newly allocated
  for (const auto &[node, grad_input] : grad_inputs) {
    if (is_first_to_init[node]) {
      profile.output_mem += grad_input.num_bytes();
    }
  }
  profile.secondary_mem = 0;  // none since backward
  grad_outputs.clear();       // clear to see real net memory
  grad_inputs.clear();        // clear to see real net memory
  residuals_.erase(edge);
  profile.net_mem = allocator->allocated() - usage_before;
  return profile;
}

void GraphExecutor::pack_memory(BuiltPlan &plan, TensorBundle &input_map,
                                TensorBundle &output_grad_map) {
  TrackingAllocator tracker(graph_.workspace_allocator());

  std::map<std::string, Node> uid_to_node;
  for (const auto &node : graph_.nodes()) uid_to_node[node->uid()] = node;

  for (const auto &[uid, tensor] : input_map) {
    auto it = uid_to_node.find(uid);
    Tensor device_tensor = tensor;
    if (tensor.device() != graph_.device()) {
      device_tensor = Tensor(tensor.shape(), tensor.dtype(), *graph_.workspace_allocator());
      copy(tensor, device_tensor, graph_.handle().get_stream());
    }
    set_data(it->second, device_tensor, data_ref_counts_[it->second]);
  }

  for (auto &edge : graph_.edges()) {
    if (edge->layer()) edge->layer()->set_workspace_allocator(&tracker);
  }

  TensorBundle output_map;

  for (const Edge &edge : plan.forward_plan.order) {
    tracker.set_current_edge(edge->uid() + "_fw");
    forward_edge(edge);
    tracker.step();

    for (const Node &consumer : edge->consumers()) {
      if (graph_.is_output(consumer)) {
        output_map.set(consumer->uid(), data(consumer));
        release_data(consumer);
      }
    }
  }

  for (const auto &[uid, tensor] : output_grad_map) {
    auto it = uid_to_node.find(uid);
    Tensor device_tensor = tensor;
    if (tensor.device() != graph_.device() && tensor) {
      device_tensor = Tensor(tensor.shape(), tensor.dtype(), *graph_.workspace_allocator());
      copy(tensor, device_tensor, graph_.handle().get_stream());
    }
    set_grad(it->second, device_tensor, grad_ref_counts_[it->second]);
  }

  TensorBundle grad_input_map;

  if (plan.backward_plan.order.size() > 0) {
    for (const Edge &edge : plan.backward_plan.order) {
      tracker.set_current_edge(edge->uid() + "_bw");
      backward_edge(edge);
      tracker.step();
      residuals_.erase(edge);

      for (const auto &producer : edge->producers()) {
        if (graph_.is_input(producer)) {
          grad_input_map.set(producer->uid(), grad(producer));
          release_grad(producer);
        }
      }
    }
  }

  output_map.clear();
  grad_input_map.clear();
  cleanup_released(data_);
  cleanup_released(grads_);
  data_.clear();
  grads_.clear();
  clear_residuals();

  tracker.step();

  for (auto &edge : graph_.edges()) {
    if (edge->layer()) edge->layer()->set_workspace_allocator(graph_.workspace_allocator());
  }

  auto pack = MemoryPacker::pack(tracker.get_allocations());
  if (os_) {
    *os_ << "Memory Packer: Peak memory packed size = " << pack.peak_memory << " bytes\n";
  }
  plan.packed_allocator = PackedAllocator::create(pack.peak_memory, pack.offsets);
  plan.packed_allocator->set_backend_allocator(graph_.workspace_allocator());
}

}  // namespace tunx