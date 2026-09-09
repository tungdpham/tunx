#pragma once

#include <functional>
#include <map>
#include <ostream>
#include <tuple>

#include "device/offload_allocator.hpp"
#include "nn/edge.hpp"
#include "nn/edge_profile.hpp"
#include "nn/execution_plan.hpp"
#include "nn/graph.hpp"
#include "nn/macro_solver.hpp"
#include "nn/memory_packer.hpp"
#include "nn/tensor_bundle.hpp"

namespace tunx {

struct BuiltPlan {
  ExecutionPlan forward_plan;
  ExecutionPlan backward_plan;
  std::map<Edge, EdgeProfile> forward_edge_profiles;
  std::map<Edge, EdgeProfile> backward_edge_profiles;
  std::map<Node, size_t> node_profiles;
  std::shared_ptr<PackedAllocator> packed_allocator;

  double extraction_time_ms = 0.0;
  double scheduling_time_ms = 0.0;
  double lifetime_time_ms = 0.0;
  double packing_time_ms = 0.0;
  double total_time_ms = 0.0;
};

class GraphExecutor {
public:
  explicit GraphExecutor(Graph &graph, bool bootstrap_offload = true);

  Graph &graph() { return graph_; }
  const Graph &graph() const { return graph_; }

  TensorBundle forward(TensorBundle &input_map);
  TensorBundle backward(TensorBundle &output_grad_map);

  const BuiltPlan &build_plans(TensorBundle &input_map, SolverOptions options = {});

  ExecutionPlanStats profile_forward_plan(TensorBundle &input_map, const ExecutionPlan &plan);
  ExecutionPlanStats profile_backward_plan(TensorBundle &input_map,
                                           const ExecutionPlan &forward_plan,
                                           const ExecutionPlan &backward_plan);

  ExecutionPlan &active_forward_plan() { return active_built_plan_.forward_plan; }
  const ExecutionPlan &active_forward_plan() const { return active_built_plan_.forward_plan; }
  ExecutionPlan &active_backward_plan() { return active_built_plan_.backward_plan; }
  const ExecutionPlan &active_backward_plan() const { return active_built_plan_.backward_plan; }

  std::tuple<TensorBundle, std::map<Edge, EdgeProfile>, std::map<Node, size_t>>
  profile_edges_forward(TensorBundle &input_map, bool discard_residuals = false);
  // edge_order, when non-null, overrides the default naive reverse-topological edge
  // iteration order. This is used to re-profile backward buffer roles (GradientOutput vs
  // GradientContribution vs Workspace) using the actual solver-scheduled order, since that
  // classification is order-sensitive at multi-producer (fan-out/accumulate) nodes.
  std::pair<TensorBundle, std::map<Edge, EdgeProfile>> profile_edges_backward(
      TensorBundle &output_grad_map, bool prefetch_residuals = false,
      const Vec<Edge> *edge_order = nullptr);

  std::map<Edge, Residuals> &residuals() { return residuals_; }
  const std::map<Edge, Residuals> &residuals() const { return residuals_; }
  void clear_residuals() { residuals_.clear(); }

  const Tensor &data(const Node &node) const;
  const Tensor &grad(const Node &node) const;
  bool has_data(const Node &node) const { return data_.count(node) > 0; }
  bool has_grad(const Node &node) const { return grads_.count(node) > 0; }

  // Returns a snapshot of all currently-live activation tensors (node uid -> tensor).
  std::map<std::string, Tensor> snapshot_data() const {
    std::map<std::string, Tensor> out;
    for (const auto &[node, entry] : data_) {
      if (entry.ref_count > 0) out[node->uid()] = entry.tensor;
    }
    return out;
  }

  // Returns a snapshot of all currently-live gradient tensors (node uid -> tensor).
  std::map<std::string, Tensor> snapshot_grads() const {
    std::map<std::string, Tensor> out;
    for (const auto &[node, entry] : grads_) {
      if (entry.ref_count > 0) out[node->uid()] = entry.tensor;
    }
    return out;
  }

  // Set a hook called during each backward_edge().
  // - consumers_grads: grad w.r.t. each consumer node's output (= PyTorch's grad_output);
  //   captured before release_grad() so these are the gradients flowing INTO the layer.
  // - producers_grads: grad w.r.t. each producer node's output (= PyTorch's grad_input);
  //   captured after accumulate_grad() so these are the gradients flowing OUT of the layer.
  // Both maps are keyed by node UID.
  // Pass nullptr to clear a previously-set hook.
  using BackwardGradHook =
      std::function<void(const Edge &, const std::map<std::string, Tensor> &consumers_grads,
                         const std::map<std::string, Tensor> &producers_grads)>;
  void set_backward_grad_hook(BackwardGradHook hook) { backward_grad_hook_ = std::move(hook); }

  using ForwardHook = std::function<void(const Edge &, const std::map<std::string, Tensor> &)>;
  void set_forward_hook(ForwardHook hook) { forward_hook_ = std::move(hook); }

  void set_log_stream(std::ostream *os) { os_ = os; }

private:
  struct PlanKey {
    std::map<Node, Vec<size_t>> input_shapes;
    std::map<Node, DType_t> input_dtypes;
    bool enable_naive;
    bool enable_linear;
    bool enable_branching;
    bool enable_joining;
    bool is_training;
    int device;
    bool operator<(const PlanKey &other) const {
      if (enable_naive != other.enable_naive) return enable_naive < other.enable_naive;
      if (enable_linear != other.enable_linear) return enable_linear < other.enable_linear;
      if (enable_branching != other.enable_branching)
        return enable_branching < other.enable_branching;
      if (enable_joining != other.enable_joining) return enable_joining < other.enable_joining;
      if (is_training != other.is_training) return is_training < other.is_training;
      if (device != other.device) return device < other.device;
      if (input_dtypes != other.input_dtypes) return input_dtypes < other.input_dtypes;
      return input_shapes < other.input_shapes;
    }
  };

  struct Entry {
    Tensor tensor;
    int ref_count = 0;
  };

  Graph &graph_;
  bool bootstrap_offload_ = true;
  std::ostream *os_ = nullptr;
  std::unique_ptr<OffloadAllocator> host_allocator_;
  BuiltPlan active_built_plan_;
  std::map<PlanKey, BuiltPlan> built_plans_;
  std::map<Node, Entry> data_;
  std::map<Node, Entry> grads_;
  std::map<Edge, Residuals> residuals_;
  std::map<Node, int> data_ref_counts_;
  std::map<Node, int> grad_ref_counts_;
  BackwardGradHook backward_grad_hook_;
  ForwardHook forward_hook_;

  void set_data(const Node &node, const Tensor &tensor, int ref_count);
  void release_data(const Node &node);
  void set_grad(const Node &node, const Tensor &tensor, int ref_count);
  void accumulate_grad(const Node &node, const Tensor &tensor, int ref_count);
  void release_grad(const Node &node);
  void forward_edge(const Edge &edge);
  void backward_edge(const Edge &edge);
  void cleanup_released(std::map<Node, Entry> &entries);

  EdgeProfile profile_edge_forward(const Edge &edge);
  EdgeProfile profile_edge_backward(const Edge &edge);

  void pack_memory(BuiltPlan &plan, TensorBundle &input_map, TensorBundle &output_grad_map);
};

}  // namespace tunx