#pragma once

#include <map>
#include <ostream>
#include <tuple>

#include "nn/edge.hpp"
#include "nn/edge_profile.hpp"
#include "nn/execution_plan.hpp"
#include "nn/graph.hpp"
#include "nn/layer.hpp"
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
};

class GraphExecutor {
public:
  explicit GraphExecutor(Graph &graph);

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
  profile_edges_forward(TensorBundle &input_map);
  std::pair<TensorBundle, std::map<Edge, EdgeProfile>> profile_edges_backward(
      TensorBundle &output_grad_map);

  std::map<Edge, Residuals> &residuals() { return residuals_; }
  const std::map<Edge, Residuals> &residuals() const { return residuals_; }
  void clear_residuals() { residuals_.clear(); }

  void set_log_stream(std::ostream *os) { os_ = os; }

private:
  struct PlanKey {
    std::map<Node, Vec<size_t>> input_shapes;
    bool enable_linear;
    bool enable_branching;
    bool enable_joining;
    bool operator<(const PlanKey &other) const {
      if (enable_linear != other.enable_linear) return enable_linear < other.enable_linear;
      if (enable_branching != other.enable_branching)
        return enable_branching < other.enable_branching;
      if (enable_joining != other.enable_joining) return enable_joining < other.enable_joining;
      return input_shapes < other.input_shapes;
    }
  };

  struct Entry {
    Tensor tensor;
    int ref_count = 0;
  };

  Graph &graph_;
  std::ostream *os_ = nullptr;
  BuiltPlan active_built_plan_;
  std::map<PlanKey, BuiltPlan> built_plans_;
  std::map<Node, Entry> data_;
  std::map<Node, Entry> grads_;
  std::map<Edge, Residuals> residuals_;
  std::map<Node, int> data_ref_counts_;
  std::map<Node, int> grad_ref_counts_;

  const Tensor &data(const Node &node) const;
  void set_data(const Node &node, const Tensor &tensor, int ref_count);
  void release_data(const Node &node);
  const Tensor &grad(const Node &node) const;
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