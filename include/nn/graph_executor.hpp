#pragma once

#include <map>

#include "nn/edge.hpp"
#include "nn/edge_profile.hpp"
#include "nn/execution_plan.hpp"
#include "nn/graph.hpp"
#include "nn/layer.hpp"
#include "nn/tensor_bundle.hpp"

namespace tunx {

class GraphExecutor {
public:
  explicit GraphExecutor(Graph &graph);

  Graph &graph() { return graph_; }
  const Graph &graph() const { return graph_; }

  TensorBundle forward(TensorBundle &input_map);
  TensorBundle backward(TensorBundle &output_grad_map);

  std::map<Edge, EdgeProfile> profile_forward(TensorBundle &input_map);
  ExecutionPlanStats profile_plan(TensorBundle &input_map, const ExecutionPlan &plan);

  ExecutionPlan &active_plan() { return active_plan_; }
  const ExecutionPlan &active_plan() const { return active_plan_; }

  std::map<Edge, Residuals> &residuals() { return residuals_; }
  const std::map<Edge, Residuals> &residuals() const { return residuals_; }

private:
  struct PlanKey {
    std::map<Node, Vec<size_t>> input_shapes;
    bool operator<(const PlanKey &other) const { return input_shapes < other.input_shapes; }
  };

  struct Entry {
    Tensor tensor;
    int ref_count = 0;
  };

  Graph &graph_;
  ExecutionPlan active_plan_;
  std::map<PlanKey, ExecutionPlan> plans_;
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
  EdgeProfile forward_edge(const Edge &edge);
  EdgeProfile backward_edge(const Edge &edge);
  void cleanup_released(std::map<Node, Entry> &entries);
};

}  // namespace tunx