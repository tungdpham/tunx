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

// Element-wise subtraction layer.
namespace internal {
class SubImpl : public LayerImpl {
protected:
  Vec<Tensor> forward_impl(const Vec<Tensor> &inputs, Residuals &residuals) override;
  Vec<Tensor> backward_impl(const Vec<Tensor> &grad_outputs, Residuals &residuals) override;

public:
  static constexpr const char *TYPE_NAME = "sub";

  explicit SubImpl(const std::string &name = "sub") { this->name_ = name; }

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;

  Node operator()(const Node &a, const Node &b) {
    if (!a || !b) {
      throw std::runtime_error("SubImpl: input nodes cannot be null");
    }
    Graph *graph = a->graph();
    if (graph != b->graph()) {
      throw std::runtime_error("SubImpl: both input nodes must belong to the same graph");
    }
    Node output = graph->make_node();
    std::shared_ptr<LayerImpl> self = shared_from_this();
    graph->add_edge(self, {a, b}, {output});
    return output;
  }

  static std::shared_ptr<SubImpl> create_from_config(const LayerConfig &config);
};

}  // namespace internal

class Sub : public LayerRef<internal::SubImpl> {
public:
  explicit Sub(const std::string &name = "sub")
      : LayerRef(std::make_shared<internal::SubImpl>(name)) {}

  using LayerRef<internal::SubImpl>::LayerRef;
};

}  // namespace tunx
