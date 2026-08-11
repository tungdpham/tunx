#pragma once

#include <map>

#include "nn/edge.hpp"
#include "nn/edge_profile.hpp"
#include "nn/execution_plan.hpp"
#include "nn/tensor_bundle.hpp"

namespace tunx {
class Graph;

class GraphExecutor {
public:
  explicit GraphExecutor(Graph &graph);

  TensorBundle forward(TensorBundle &input_map);
  TensorBundle backward(TensorBundle &output_grad_map);
  void clear_grads();

  std::map<Edge, EdgeProfile> profile_forward(TensorBundle &input_map);

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
  EdgeProfile forward_edge(Edge &edge);
  EdgeProfile backward_edge(Edge &edge);
  void cleanup_released(std::map<Node, Entry> &entries);
};

}  // namespace tunx