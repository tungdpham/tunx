#include "partitioner/graph_partitioner.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace tunx {
namespace {

using WeakNode = std::weak_ptr<NodeImpl>;
using WeakNodeSet = std::set<WeakNode, std::owner_less<WeakNode>>;
using EdgeMap = std::map<WeakNode, Vec<Edge>, std::owner_less<WeakNode>>;

Vec<Edge> topologically_sorted_edges(const Graph &graph) {
  const Vec<Edge> original_edges = graph.edges();

  WeakNodeSet produced_nodes;
  for (const auto &edge : original_edges) {
    for (const auto &consumer : edge->consumers()) {
      produced_nodes.insert(consumer);
    }
  }

  EdgeMap node_to_dependent_edges;
  for (const auto &edge : original_edges) {
    for (const auto &producer : edge->producers()) {
      node_to_dependent_edges[producer].push_back(edge);
    }
  }

  std::map<Edge, size_t> pending_dependencies;
  std::queue<Edge> ready_edges;

  for (const auto &edge : original_edges) {
    size_t dependency_count = 0;
    for (const auto &producer : edge->producers()) {
      if (produced_nodes.count(producer) > 0) {
        ++dependency_count;
      }
    }
    pending_dependencies[edge] = dependency_count;
    if (dependency_count == 0) {
      ready_edges.push(edge);
    }
  }

  Vec<Edge> sorted_edges;
  sorted_edges.reserve(original_edges.size());

  while (!ready_edges.empty()) {
    Edge edge = ready_edges.front();
    ready_edges.pop();
    sorted_edges.push_back(edge);

    for (const auto &consumer : edge->consumers()) {
      const auto it = node_to_dependent_edges.find(consumer);
      if (it == node_to_dependent_edges.end()) {
        continue;
      }

      for (const auto &dependent_edge : it->second) {
        size_t &remaining = pending_dependencies[dependent_edge];
        if (remaining == 0) {
          continue;
        }
        --remaining;
        if (remaining == 0) {
          ready_edges.push(dependent_edge);
        }
      }
    }
  }

  if (sorted_edges.size() != original_edges.size()) {
    throw std::runtime_error("Graph contains a cycle; partitioning requires a DAG");
  }

  return sorted_edges;
}

Node ensure_partition_node(Graph &graph, const std::string &uid,
                           std::unordered_map<std::string, Node> &uid_to_node,
                           std::vector<std::string> &node_order) {
  const auto it = uid_to_node.find(uid);
  if (it != uid_to_node.end()) {
    return it->second;
  }

  Node node = graph.make_node(uid);
  uid_to_node.emplace(uid, node);
  node_order.push_back(uid);
  return node;
}

GraphPartition build_partition(const Graph &graph, const Vec<Edge> &edges, size_t start_layer) {
  const Vec<Edge> sorted_edges = topologically_sorted_edges(graph);
  size_t end_layer = start_layer + edges.size();

  GraphPartition partition;
  partition.start_layer = start_layer;
  partition.layer_count = edges.size();

  std::unordered_map<std::string, Node> uid_to_node;
  std::vector<std::string> node_order;
  std::unordered_set<std::string> produced_in_partition;
  std::unordered_set<std::string> consumed_in_partition;
  std::unordered_set<std::string> consumed_later;

  for (const auto &edge : edges) {
    Vec<Node> producers;
    producers.reserve(edge->producers().size());
    for (const auto &producer : edge->producers()) {
      const std::string &uid = producer->uid();
      consumed_in_partition.insert(uid);
      producers.push_back(ensure_partition_node(partition.graph, uid, uid_to_node, node_order));
    }

    Vec<Node> consumers;
    consumers.reserve(edge->consumers().size());
    for (const auto &consumer : edge->consumers()) {
      const std::string &uid = consumer->uid();
      produced_in_partition.insert(uid);
      consumers.push_back(ensure_partition_node(partition.graph, uid, uid_to_node, node_order));
    }

    partition.graph.add_edge(edge->layer(), producers, consumers);
  }

  for (size_t edge_index = end_layer; edge_index < sorted_edges.size(); ++edge_index) {
    const Edge &later_edge = sorted_edges[edge_index];
    for (const auto &producer : later_edge->producers()) {
      const std::string &uid = producer->uid();
      if (produced_in_partition.count(uid) > 0) {
        consumed_later.insert(uid);
      }
    }
  }

  for (const auto &uid : node_order) {
    if (consumed_in_partition.count(uid) > 0 && produced_in_partition.count(uid) == 0) {
      partition.input_uids.push_back(uid);
    }

    const bool produced_here = produced_in_partition.count(uid) > 0;
    const bool consumed_here = consumed_in_partition.count(uid) > 0;
    const bool needed_by_later_partition = consumed_later.count(uid) > 0;
    if (produced_here && (!consumed_here || needed_by_later_partition)) {
      partition.output_uids.push_back(uid);
    }
  }

  for (const auto &uid : partition.input_uids) {
    partition.graph.set_input(uid_to_node.at(uid));
  }
  for (const auto &uid : partition.output_uids) {
    partition.graph.set_output(uid_to_node.at(uid));
  }

  return partition;
}

}  // namespace

GraphPartitioner::GraphPartitioner(std::vector<double> layer_ratios)
    : layer_ratios_(std::move(layer_ratios)) {}

GraphPartitioner::GraphPartitioner(std::vector<size_t> layer_ratios) {
  layer_ratios_.reserve(layer_ratios.size());
  for (size_t layer_ratio : layer_ratios) {
    layer_ratios_.push_back(static_cast<double>(layer_ratio));
  }
}

std::vector<GraphPartition> GraphPartitioner::partition(const Graph &graph) const {
  const Vec<Edge> sorted_edges = topologically_sorted_edges(graph);
  const std::vector<size_t> layer_counts = resolve_layer_counts(sorted_edges.size());

  std::vector<GraphPartition> partitions;
  partitions.reserve(layer_counts.size());

  std::cout << "\n=== Partition Metrics (GraphPartitioner) ===" << std::endl;
  size_t offset = 0;
  for (size_t i = 0; i < layer_counts.size(); ++i) {
    size_t layer_count = layer_counts[i];
    std::cout << "Worker " << (i + 1) << " edges: " << layer_count << " (edges " << offset << " to " << (offset + layer_count - 1) << ")" << std::endl;
    
    Vec<Edge> partition_edges;
    partition_edges.reserve(layer_count);
    for (size_t j = 0; j < layer_count; ++j) {
      partition_edges.push_back(sorted_edges[offset + j]);
    }

    partitions.push_back(build_partition(graph, partition_edges, offset));
    offset += layer_count;
  }
  std::cout << "============================================\n" << std::endl;

  return partitions;
}

std::vector<size_t> GraphPartitioner::resolve_layer_counts(size_t total_layers) const {
  if (layer_ratios_.empty()) {
    if (total_layers == 0) {
      return {};
    }
    throw std::runtime_error("GraphPartitioner requires at least one partition ratio");
  }

  if (total_layers == 0) {
    throw std::runtime_error("GraphPartitioner cannot apply partition ratios to an empty graph");
  }

  double ratio_sum = 0.0;
  std::vector<size_t> layer_counts(layer_ratios_.size(), 0);
  struct FractionalCount {
    size_t index;
    double remainder;
  };
  std::vector<FractionalCount> fractional_counts;
  fractional_counts.reserve(layer_ratios_.size());

  for (const double layer_ratio : layer_ratios_) {
    if (!std::isfinite(layer_ratio) || layer_ratio <= 0.0) {
      throw std::runtime_error(
          "GraphPartitioner partition ratios must be finite and greater than zero");
    }
    ratio_sum += layer_ratio;
  }

  for (size_t i = 0; i < layer_ratios_.size(); ++i) {
    const double exact_layer_count =
        (layer_ratios_[i] / ratio_sum) * static_cast<double>(total_layers);
    size_t resolved_layer_count = static_cast<size_t>(std::floor(exact_layer_count));
    layer_counts[i] = resolved_layer_count;
    fractional_counts.push_back({i, exact_layer_count - static_cast<double>(resolved_layer_count)});
  }

  size_t assigned_layers = std::accumulate(layer_counts.begin(), layer_counts.end(), size_t{0});
  std::sort(fractional_counts.begin(), fractional_counts.end(),
            [](const FractionalCount &lhs, const FractionalCount &rhs) {
              if (lhs.remainder == rhs.remainder) {
                return lhs.index < rhs.index;
              }
              return lhs.remainder > rhs.remainder;
            });

  for (size_t i = 0; i < total_layers - assigned_layers; ++i) {
    ++layer_counts[fractional_counts[i].index];
  }

  for (size_t layer_count : layer_counts) {
    if (layer_count == 0) {
      throw std::runtime_error(
          "GraphPartitioner partition ratios resolve to an empty partition for this graph");
    }
  }

  return layer_counts;
}

ComputeBandwidthPartitioner::ComputeBandwidthPartitioner(
    DeviceMesh mesh,
    std::function<double(const Edge &)> compute_cost_fn,
    std::function<double(const Node &)> activation_size_fn)
    : mesh_(std::move(mesh)),
      compute_cost_fn_(std::move(compute_cost_fn)),
      activation_size_fn_(std::move(activation_size_fn)) {
  if (compute_cost_fn_ == nullptr) {
    compute_cost_fn_ = [](const Edge &) { return 1.0; };
  }
  if (activation_size_fn_ == nullptr) {
    activation_size_fn_ = [](const Node &) { return 1.0; };
  }
}

std::vector<GraphPartition> ComputeBandwidthPartitioner::partition(const Graph &graph) const {
  const Vec<Edge> sorted_edges = topologically_sorted_edges(graph);
  const size_t N = sorted_edges.size();
  const size_t K = mesh_.compute_powers.size();

  if (K == 0) {
    throw std::runtime_error("ComputeBandwidthPartitioner requires at least one machine in the mesh");
  }
  if (mesh_.link_speeds.size() != K - 1) {
    throw std::runtime_error("ComputeBandwidthPartitioner link_speeds size must be compute_powers size - 1");
  }
  if (N < K) {
    throw std::runtime_error("ComputeBandwidthPartitioner cannot partition graph with fewer edges than machines");
  }
  for (double p : mesh_.compute_powers) {
    if (p <= 0.0) throw std::runtime_error("Compute power must be positive");
  }
  for (double l : mesh_.link_speeds) {
    if (l <= 0.0) throw std::runtime_error("Link speed must be positive");
  }

  std::unordered_map<std::string, size_t> producer_idx;
  std::unordered_map<std::string, size_t> last_consumer_idx;
  std::unordered_map<std::string, Node> uid_to_node;

  for (size_t i = 0; i < N; ++i) {
    const Edge &edge = sorted_edges[i];
    for (const auto &input_node : edge->producers()) {
      const std::string &uid = input_node->uid();
      uid_to_node[uid] = input_node;
      if (producer_idx.find(uid) == producer_idx.end()) {
        producer_idx[uid] = 0; 
      }
      last_consumer_idx[uid] = std::max(last_consumer_idx[uid], i + 1);
    }
    
    for (const auto &output_node : edge->consumers()) {
      const std::string &uid = output_node->uid();
      uid_to_node[uid] = output_node;
      producer_idx[uid] = i + 1;
      if (last_consumer_idx.find(uid) == last_consumer_idx.end()) {
        last_consumer_idx[uid] = i + 1;
      } else {
        last_consumer_idx[uid] = std::max(last_consumer_idx[uid], i + 1);
      }
    }
  }

  std::vector<double> prefix_compute(N + 1, 0.0);
  for (size_t i = 0; i < N; ++i) {
    prefix_compute[i + 1] = prefix_compute[i] + compute_cost_fn_(sorted_edges[i]);
  }

  std::vector<double> boundary_size(N + 1, 0.0);
  for (const auto &[uid, node] : uid_to_node) {
    size_t prod = producer_idx[uid];
    size_t cons = last_consumer_idx[uid];
    if (prod < cons) {
      double size = activation_size_fn_(node);
      for (size_t j = prod; j < cons; ++j) {
        if (j > 0 && j < N) {
          boundary_size[j] += size;
        }
      }
    }
  }

  const double INF = std::numeric_limits<double>::infinity();
  std::vector<std::vector<double>> dp(K + 1, std::vector<double>(N + 1, INF));
  std::vector<std::vector<size_t>> parent(K + 1, std::vector<size_t>(N + 1, 0));

  dp[0][0] = 0.0;

  for (size_t k = 1; k <= K; ++k) {
    double P_k = mesh_.compute_powers[k - 1];
    double L_k = (k < K) ? mesh_.link_speeds[k - 1] : 1.0;

    for (size_t i = 1; i <= N; ++i) {
      for (size_t p = 0; p < i; ++p) {
        if (dp[k - 1][p] == INF) continue;

        double compute_time = (prefix_compute[i] - prefix_compute[p]) / P_k;
        double comm_time = (k < K) ? ((boundary_size[i] / L_k) * 1000.0) : 0.0;

        double bottleneck = std::max({dp[k - 1][p], compute_time, comm_time});

        if (bottleneck < dp[k][i]) {
          dp[k][i] = bottleneck;
          parent[k][i] = p;
        }
      }
    }
  }

  if (dp[K][N] == INF) {
    throw std::runtime_error("ComputeBandwidthPartitioner failed to find a valid partition");
  }

  std::vector<size_t> cuts(K + 1, 0);
  cuts[K] = N;
  size_t curr = N;
  for (size_t k = K; k >= 1; --k) {
    cuts[k - 1] = parent[k][curr];
    curr = cuts[k - 1];
  }

  std::cout << "\n=== Partition Metrics (ComputeBandwidthPartitioner) ===" << std::endl;
  std::cout << "Predicted J (ms): " << dp[K][N] << std::endl;
  for (size_t k = 0; k < K; ++k) {
    size_t start = cuts[k];
    size_t end = cuts[k + 1];
    std::cout << "Worker " << (k + 1) << " edges: " << (end - start) << " (edges " << start << " to " << (end - 1) << ")" << std::endl;
    if (k < K - 1) {
      std::cout << "Boundary " << (k + 1) << " -> " << (k + 2) << " size (MiB): " << (boundary_size[end] / (1024.0 * 1024.0)) << std::endl;
    }
  }
  std::cout << "========================================================\n" << std::endl;

  std::vector<GraphPartition> partitions;
  partitions.reserve(K);
  for (size_t k = 0; k < K; ++k) {
    size_t start = cuts[k];
    size_t end = cuts[k + 1];
    if (start == end) {
      throw std::runtime_error("ComputeBandwidthPartitioner produced an empty partition");
    }

    Vec<Edge> partition_edges;
    partition_edges.reserve(end - start);
    for (size_t i = start; i < end; ++i) {
      partition_edges.push_back(sorted_edges[i]);
    }

    partitions.push_back(build_partition(graph, partition_edges, start));
  }

  return partitions;
}

}  // namespace tunx