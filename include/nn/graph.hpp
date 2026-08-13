#pragma once

#include <array>
#include <initializer_list>
#include <iosfwd>
#include <iostream>
#include <memory>
#include <set>
#include <string>

#include "device/del_allocator_v2.hpp"
#include "device/iallocator.hpp"
#include "device/stream.hpp"
#include "nn/edge.hpp"
#include "nn/engines/engine_handle.hpp"
#include "nn/node.hpp"
#include "nn/param.hpp"
#include "type/type.hpp"

namespace tunx {

enum class ExecutionMode {
  TRAIN,
  EVAL,
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
  Graph(const Graph &) = delete;
  Graph &operator=(const Graph &) = delete;
  Graph(Graph &&other) noexcept;
  Graph &operator=(Graph &&other) noexcept = delete;
  ~Graph();

  void save_state(std::ostream &stream) const;
  static Graph load_state(std::istream &stream, IAllocator &allocator);

  void compile(IAllocator &allocator, GraphOpts opts = GraphOpts{});

  Engine &engine() { return engine_; }
  const Engine &engine() const { return engine_; }

  engine_handle &handle() { return engine_handle_; }
  const engine_handle &handle() const { return engine_handle_; }

  const Vec<Node> &nodes() const { return nodes_; }
  const Vec<Edge> &edges() const { return edges_; }
  const std::set<Node> &outputs() const { return output_nodes_; }
  const std::set<Node> &inputs() const { return input_nodes_; }

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
  void save_dot(const std::string &filename) const;

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

  Vec<Param> params();

private:
  // backend
  IAllocator *param_allocator_;
  std::shared_ptr<DELAllocatorV2> workspace_allocator_;
  Engine engine_;
  engine_handle engine_handle_;

  // connectivity
  Vec<Node> nodes_;
  Vec<Edge> edges_;
  std::set<Node> input_nodes_;
  std::set<Node> output_nodes_;
  ExecutionMode mode_ = ExecutionMode::TRAIN;
  size_t node_count_ = 0;
  size_t edge_count_ = 0;
  std::set<std::string> used_node_uids_;
  std::set<std::string> used_edge_uids_;

  std::string generate_node_uid();
  std::string generate_edge_uid();
};

}  // namespace tunx