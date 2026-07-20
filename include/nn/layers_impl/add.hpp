/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "nn/graph.hpp"
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

namespace internal {
class AddImpl : public LayerImpl {
protected:
  Vec<Tensor> forward_impl(const Vec<Tensor> &inputs, Residuals &residuals) override;
  Vec<Tensor> backward_impl(const Vec<Tensor> &grad_outputs, Residuals &residuals) override;

public:
  static constexpr const char *TYPE_NAME = "add";

  explicit AddImpl(const std::string &name = "add") { this->name_ = name; }

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;

  Node operator()(const Node &a, const Node &b) {
    if (!a || !b) {
      throw std::runtime_error("AddImpl: input nodes cannot be null");
    }
    Graph *graph = a->graph();
    if (graph != b->graph()) {
      throw std::runtime_error("AddImpl: both input nodes must belong to the same graph");
    }
    Node output = graph->make_node();
    std::shared_ptr<LayerImpl> self = shared_from_this();
    graph->add_edge(self, {a, b}, {output});
    return output;
  }

  static std::shared_ptr<AddImpl> create_from_config(const LayerConfig &config);
};

}  // namespace internal

class Add : public LayerRef<internal::AddImpl> {
public:
  Add(const std::string &name = "add")
      : LayerRef(std::make_shared<internal::AddImpl>(name)) {}

  using LayerRef<internal::AddImpl>::LayerRef;
};

}  // namespace tunx
