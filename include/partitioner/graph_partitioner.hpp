#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "nn/graph.hpp"

namespace tunx {

struct GraphPartition {
  Graph graph;
  std::vector<std::string> input_uids;
  std::vector<std::string> output_uids;
  size_t start_layer = 0;
  size_t layer_count = 0;
};

class PartitionerBase {
public:
  virtual ~PartitionerBase() = default;
  virtual std::vector<GraphPartition> partition(const Graph &graph) const = 0;
};

class GraphPartitioner : public PartitionerBase {
public:
  explicit GraphPartitioner(std::vector<double> layer_ratios);
  explicit GraphPartitioner(std::initializer_list<double> layer_ratios)
      : GraphPartitioner(std::vector<double>(layer_ratios)) {}
  explicit GraphPartitioner(std::vector<size_t> layer_ratios);

  std::vector<GraphPartition> partition(const Graph &graph) const override;
  const std::vector<double> &layer_ratios() const { return layer_ratios_; }

private:
  std::vector<size_t> resolve_layer_counts(size_t total_layers) const;

  std::vector<double> layer_ratios_;
};

struct DeviceMesh {
  std::vector<double> compute_powers;
  std::vector<double> link_speeds;
};

class ComputeBandwidthPartitioner : public PartitionerBase {
public:
  ComputeBandwidthPartitioner(
      DeviceMesh mesh,
      std::function<double(const Edge &)> compute_cost_fn = nullptr,
      std::function<double(const Node &)> activation_size_fn = nullptr);

  std::vector<GraphPartition> partition(const Graph &graph) const override;

private:
  DeviceMesh mesh_;
  std::function<double(const Edge &)> compute_cost_fn_;
  std::function<double(const Node &)> activation_size_fn_;
};

}  // namespace tunx