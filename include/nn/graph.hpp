#pragma once

#include <array>
#include <initializer_list>
#include <iosfwd>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "device/del_allocator_v2.hpp"
#include "device/iallocator.hpp"
#include "device/stream.hpp"
#include "nn/edge.hpp"
#include "nn/engines/engine_handle.hpp"
#include "nn/node.hpp"
#include "nn/param.hpp"
#include "nn/tensor_bundle.hpp"
#include "type/type.hpp"

namespace tunx {

enum class ExecutionMode {
  TRAIN,
  EVAL,
};

struct SPSCRegion {
  Vec<size_t> edge_indices;
};

struct MPSCRegion {
  size_t join_edge_index = 0;
  Vec<Vec<size_t>> branch_edge_indices;
};

struct GraphRegionSummary {
  Vec<SPSCRegion> spsc_regions;
  Vec<MPSCRegion> mpsc_regions;
  Vec<size_t> unsupported_edge_indices;
};

struct EdgeExecutionProfile {
  size_t peak_bytes = 0;
  size_t retained_bytes = 0;
};

struct ForwardPlanCacheEntry {
  ExecutionMode mode = ExecutionMode::TRAIN;
  std::map<std::string, Vec<size_t>> input_shapes;
  GraphRegionSummary regions;
  std::map<size_t, EdgeExecutionProfile> edge_profiles;
  size_t execution_count = 0;
  bool profiled = false;
};

struct GraphOpts {
  Engine engine = nullptr;
  stream s = nullptr;
  unsigned long long seed = 0;
  DType_t io_dtype = DType_t::FP32;
  DType_t param_dtype = DType_t::FP32;
  DType_t compute_dtype = DType_t::FP32;
};

class Graph {
public:
  Graph() = default;

  void save_state(std::ostream &stream) const;
  static Graph load_state(std::istream &stream, IAllocator &allocator);

  void compile(IAllocator &allocator, GraphOpts opts = GraphOpts{});

  Engine &engine() { return engine_; }
  const Engine &engine() const { return engine_; }

  engine_handle &handle() { return engine_handle_; }
  const engine_handle &handle() const { return engine_handle_; }

  Vec<Node> nodes() const { return nodes_; }
  Vec<Edge> edges() const { return edges_; }

  Vec<std::string> input_uids() const;
  Vec<std::string> output_uids() const;

  Device &device() const { return param_allocator_->device(); }

  const DELAllocatorV2 *workspace_allocator() const { return workspace_allocator_.get(); }
  DELAllocatorV2 *workspace_allocator() { return workspace_allocator_.get(); }

  void add_edge(std::shared_ptr<LayerImpl> layer, const Vec<Node> &producers,
                const Vec<Node> &consumers);

  void add_edge(std::shared_ptr<LayerImpl> layer, std::initializer_list<Node> producers,
                std::initializer_list<Node> consumers);

  void sort();

  TensorBundle forward(TensorBundle &input_map, size_t pid = 0);
  TensorBundle backward(TensorBundle &output_grad_map, size_t pid = 0);

  Node make_node(std::string uid = "");

  void set_mode(ExecutionMode mode);

  void set_input(Node node);
  void set_output(Node node);
  bool is_input(const Node &node) const { return input_nodes_.count(node) > 0; }
  bool is_output(const Node &node) const { return output_nodes_.count(node) > 0; }

  Node input(const std::string &uid = "input");

  template <typename... Args>
  std::array<Node, sizeof...(Args)> inputs(Args... uids) {
    return {make_node(uids)...};
  }

  void zero_grads();

  Vec<Param> params();

  std::vector<std::pair<std::string, double>> profiling_details() const {
    std::vector<std::pair<std::string, double>> res;
    for (const auto &k : timing_order_) {
      res.push_back({k, timing_map_.at(k)});
    }
    return res;
  }

  void clear_profiling_details() {
    timing_map_.clear();
    timing_order_.clear();
  }

  void enable_memory_profiling(bool enable, std::ostream *os = nullptr) {
    enable_memory_profiling_ = enable;
    memory_profile_stream_ = os;
  }

private:
  // backend
  IAllocator *param_allocator_;
  std::shared_ptr<DELAllocatorV2> workspace_allocator_;
  Engine engine_;
  engine_handle engine_handle_;
  DType_t io_dtype_;
  DType_t param_dtype_;
  DType_t compute_dtype_;

  // connectivity
  Vec<Node> nodes_;
  Vec<Edge> edges_;
  std::set<Node> input_nodes_;
  std::set<Node> output_nodes_;
  std::map<std::string, double> timing_map_;  // layer name -> total time taken.
  std::vector<std::string> timing_order_;
  std::map<std::weak_ptr<NodeImpl>, int, std::owner_less<std::weak_ptr<NodeImpl>>> in_degree_;
  std::map<std::weak_ptr<NodeImpl>, int, std::owner_less<std::weak_ptr<NodeImpl>>> out_degree_;
  ExecutionMode mode_ = ExecutionMode::TRAIN;
  size_t node_count_ = 0;
  std::set<std::string> used_uids_;
  bool enable_memory_profiling_ = false;
  std::ostream *memory_profile_stream_ = nullptr;

  int node_in_degree(const Node &node) const;
  int node_out_degree(const Node &node) const;
  std::string generate_uid();

  Vec<Node> inputs();
  Vec<Node> outputs();

  void on_add_edge(const Edge &edge);
  void forward_edge(Edge &edge, size_t pid = 0);
  void backward_edge(Edge &edge, size_t pid = 0);
};

}  // namespace tunx