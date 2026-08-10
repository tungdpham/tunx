#pragma once

#include <map>

#include "nn/edge.hpp"
#include "nn/tensor_bundle.hpp"

namespace tunx {
class Graph;
struct ForwardPlanCacheEntry;

class GraphExecutor {
public:
  explicit GraphExecutor(Graph &graph);

  TensorBundle forward(TensorBundle &input_map);
  TensorBundle backward(TensorBundle &output_grad_map);
  void clear_grads();

private:
  struct Entry {
    Tensor tensor;
    int ref_count = 0;
  };

  Graph &graph_;
  ForwardPlanCacheEntry *active_plan_ = nullptr;
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
  void forward_edge(Edge &edge, size_t edge_index);
  void backward_edge(Edge &edge);
  void cleanup_released(std::map<Node, Entry> &entries);
};

}  // namespace tunx