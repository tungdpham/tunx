/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/graph.hpp"

#include <fmt/core.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <istream>
#include <ostream>
#include <unordered_map>

#include "device/device_type.hpp"
#include "device/stream.hpp"
#include "nn/engines/cpu_engine.hpp"
#include "nn/engines/engine_handle.hpp"
#include "tensor/ops.hpp"
#ifdef TUNX_USE_CUDA
#include "nn/engines/cuda_engine.hpp"
#include "nn/engines/cudnn_engine.hpp"
#endif
#include "nn/layer.hpp"
#include "nn/layer_factory.hpp"
#include "nn/param.hpp"

namespace tunx {

static Engine get_default_engine(Device &device) {
  switch (device.device_type()) {
    case DeviceType::CPU:
      return make_engine<CPUEngine>();
#ifdef TUNX_USE_CUDA
    case DeviceType::CUDA:
#ifdef TUNX_USE_CUDNN
      return make_engine<CuDNNEngine>();
#endif
      return make_engine<CUDAEngine>();
#endif
    default:
      throw std::runtime_error("Unsupported device type for graph");
  }
}

void Graph::compile(IAllocator &allocator, GraphOpts opts) {
  sort();
  std::set<LayerImpl *> unique_layers;
  for (const auto &edge : edges_) {
    LayerImpl *layer_ptr = edge->layer().get();
    if (unique_layers.count(layer_ptr) == 0) {
      unique_layers.insert(layer_ptr);
    }
  }

  // if stream not provided or invalid, take device's default stream
  Device &device = allocator.device();
  stream s = opts.s ? opts.s : device.default_stream();

  engine_ = opts.engine ? opts.engine : get_default_engine(allocator.device());
  param_allocator_ = &allocator;
  workspace_allocator_ = DELAllocatorV2::instance(device, s).get();
  engine_handle_ = engine_->create_handle(s);

  io_dtype_ = opts.io_dtype;
  param_dtype_ = opts.param_dtype;
  compute_dtype_ = opts.compute_dtype;

  InitOptions layer_opts{
      .ws_allocator = workspace_allocator_,
      .engine = engine_,
      .handle = engine_handle_,
      .seed = opts.seed,
      .io_dtype = opts.io_dtype,
      .param_dtype = opts.param_dtype,
      .compute_dtype = opts.compute_dtype,
  };

  for (LayerImpl *layer_ptr : unique_layers) {
    layer_ptr->init(*param_allocator_, layer_opts);
  }

  // sanity check: ensure there are output nodes
  if (output_nodes_.empty()) {
    throw std::runtime_error("Graph must have at least one output node");
  }
}

Vec<std::string> Graph::input_uids() const {
  Vec<std::string> uids;
  uids.reserve(input_nodes_.size());
  for (const auto &node : nodes_) {
    if (is_input(node)) {
      uids.push_back(node->uid());
    }
  }
  return uids;
}

Vec<std::string> Graph::output_uids() const {
  Vec<std::string> uids;
  uids.reserve(output_nodes_.size());
  for (const auto &node : nodes_) {
    if (is_output(node)) {
      uids.push_back(node->uid());
    }
  }
  return uids;
}

void Graph::add_edge(std::shared_ptr<LayerImpl> layer, const Vec<Node> &producers,
                     const Vec<Node> &consumers) {
  Edge edge = std::make_shared<EdgeImpl>(layer, producers, consumers);
  std::string uid = edge->layer()->name();
  if (used_edge_uids_.count(uid) > 0) {
    uid = generate_edge_uid();
  } else {
    used_edge_uids_.insert(uid);
  }
  edge->set_uid(uid);
  edges_.push_back(edge);
}

void Graph::add_edge(std::shared_ptr<LayerImpl> layer, std::initializer_list<Node> producers,
                     std::initializer_list<Node> consumers) {
  add_edge(layer, Vec<Node>(producers), Vec<Node>(consumers));
}

void Graph::sort() {
  std::set<std::weak_ptr<NodeImpl>, std::owner_less<std::weak_ptr<NodeImpl>>> produced_nodes;
  for (const auto &edge : edges_) {
    for (const auto &consumer : edge->consumers()) {
      produced_nodes.insert(consumer);
    }
  }

  std::map<std::weak_ptr<NodeImpl>, Vec<Edge>, std::owner_less<std::weak_ptr<NodeImpl>>>
      node_to_dependent_edges;
  for (const auto &edge : edges_) {
    for (const auto &producer : edge->producers()) {
      node_to_dependent_edges[producer].push_back(edge);
    }
  }

  std::map<Edge, int> pending;
  Vec<Edge> ready;

  for (auto it = edges_.rbegin(); it != edges_.rend(); ++it) {
    const auto &edge = *it;
    int count = 0;
    for (const auto &producer : edge->producers()) {
      if (produced_nodes.count(producer)) {
        ++count;
      }
    }
    pending[edge] = count;
    if (count == 0) {
      ready.push_back(edge);
    }
  }

  Vec<Edge> sorted;
  sorted.reserve(edges_.size());

  while (!ready.empty()) {
    Edge e = ready.back();
    ready.pop_back();
    sorted.push_back(e);

    for (auto consumer_it = e->consumers().rbegin(); consumer_it != e->consumers().rend();
         ++consumer_it) {
      const auto &consumer = *consumer_it;
      auto it = node_to_dependent_edges.find(consumer);
      if (it != node_to_dependent_edges.end()) {
        for (auto dep_it = it->second.rbegin(); dep_it != it->second.rend(); ++dep_it) {
          const auto &dep_edge = *dep_it;
          if (--pending[dep_edge] == 0) {
            ready.push_back(dep_edge);
          }
        }
      }
    }
  }

  if (sorted.size() != edges_.size()) {
    throw std::runtime_error("Graph contains a cycle; topological sort failed");
  }

  edges_ = std::move(sorted);

  std::set<NodeImpl *> placed;
  Vec<Node> sorted_nodes;
  sorted_nodes.reserve(nodes_.size());

  for (const auto &node : nodes_) {
    if (!produced_nodes.count(node)) {
      sorted_nodes.push_back(node);
      placed.insert(node.get());
    }
  }

  for (const auto &edge : edges_) {
    for (const auto &consumer : edge->consumers()) {
      if (placed.insert(consumer.get()).second) {
        sorted_nodes.push_back(consumer);
      }
    }
  }

  nodes_ = std::move(sorted_nodes);
}

void Graph::save_dot(const std::string &filename, const std::map<Edge, EdgeProfile> *edge_profiles,
                     const std::map<Node, size_t> *node_profiles) const {
  std::ofstream output(filename);
  if (!output) throw std::runtime_error("Failed to open DOT file: " + filename);

  output << "digraph Graph {\n  rankdir=LR;\n";
  for (const Node &node : nodes_) {
    output << "  \"node_" << node->uid() << "\" [shape=ellipse, label=\"" << node->uid();
    if (node_profiles && node_profiles->count(node)) {
      output << "\\nsize: " << node_profiles->at(node);
    }
    output << "\"];\n";
  }
  for (size_t index = 0; index < edges_.size(); ++index) {
    const Edge &edge = edges_[index];
    output << "  \"edge_" << index << "\" [shape=box, label=\"" << index << ": "
           << edge->layer()->name();
    if (edge_profiles && edge_profiles->count(edge)) {
      const auto &prof = edge_profiles->at(edge);
      output << "\\na: " << prof.total_mem;
      output << "\\nb: " << prof.net_mem;
      output << "\\nworkspace: " << prof.workspace_mem;
      output << "\\nresidual: " << prof.total_mem - prof.workspace_mem - prof.output_mem;
    }
    output << "\"];\n";
    for (const Node &producer : edge->producers()) {
      output << "  \"node_" << producer->uid() << "\" -> \"edge_" << index << "\";\n";
    }
    for (const Node &consumer : edge->consumers()) {
      output << "  \"edge_" << index << "\" -> \"node_" << consumer->uid() << "\";\n";
    }
  }
  output << "}\n";
}

Node Graph::make_node(std::string uid) {
  if (uid.empty()) {
    uid = generate_node_uid();
  } else if (used_node_uids_.count(uid) > 0) {
    throw std::runtime_error("Duplicate node UID: " + uid);
  } else {
    used_node_uids_.insert(uid);
  }
  Node node = std::make_shared<NodeImpl>(this, uid);
  nodes_.push_back(node);
  return node;
}

void Graph::set_mode(ExecutionMode mode) {
  mode_ = mode;
  for (const auto &edge : edges_) {
    edge->layer()->set_training(mode == ExecutionMode::TRAIN);
  }
}

void Graph::set_input(Node node) {
  if (std::find(nodes_.begin(), nodes_.end(), node) == nodes_.end()) {
    throw std::runtime_error("Input node does not belong to graph");
  }
  input_nodes_.insert(node);
}

void Graph::set_output(Node node) {
  if (std::find(nodes_.begin(), nodes_.end(), node) == nodes_.end()) {
    throw std::runtime_error("Output node does not belong to graph");
  }
  output_nodes_.insert(node);
}

Node Graph::input(const std::string &uid) {
  Node node = make_node(uid);
  set_input(node);
  return node;
}

Graph::Graph(Graph &&other) noexcept
    : param_allocator_(other.param_allocator_),
      workspace_allocator_(std::move(other.workspace_allocator_)),
      engine_(std::move(other.engine_)),
      engine_handle_(std::move(other.engine_handle_)),
      nodes_(std::move(other.nodes_)),
      edges_(std::move(other.edges_)),
      input_nodes_(std::move(other.input_nodes_)),
      output_nodes_(std::move(other.output_nodes_)),
      mode_(other.mode_),
      node_count_(other.node_count_),
      used_node_uids_(std::move(other.used_node_uids_)),
      used_edge_uids_(std::move(other.used_edge_uids_)) {}

Vec<Param> Graph::params() {
  Vec<Param> params;
  for (auto &edge : edges_) {
    auto layer = edge->layer();
    auto layer_params = layer->params();
    params.insert(params.end(), layer_params.begin(), layer_params.end());
  }
  return params;
}

std::string Graph::generate_node_uid() {
  std::string uid;
  do {
    uid = "node_" + std::to_string(node_count_++);
  } while (used_node_uids_.count(uid) > 0);
  used_node_uids_.insert(uid);
  return uid;
}

std::string Graph::generate_edge_uid() {
  std::string uid;
  do {
    uid = "edge_" + std::to_string(edge_count_++);
  } while (used_edge_uids_.count(uid) > 0);
  used_edge_uids_.insert(uid);
  return uid;
}

Graph::~Graph() = default;

namespace {

constexpr std::array<char, 4> kGraphStateMagic{'T', 'U', 'N', 'X'};
constexpr std::uint32_t kGraphStateVersion = 5;

template <typename T>
void write_binary(std::ostream &os, const T &value) {
  os.write(reinterpret_cast<const char *>(&value), sizeof(T));
  if (!os) {
    throw std::runtime_error("Failed to write graph state");
  }
}

template <typename T>
T read_binary(std::istream &os) {
  T value{};
  os.read(reinterpret_cast<char *>(&value), sizeof(T));
  if (!os) {
    throw std::runtime_error("Failed to read graph state");
  }
  return value;
}

void write_string(std::ostream &os, const std::string &value) {
  const size_t size = value.size();
  write_binary(os, size);
  os.write(value.data(), static_cast<std::streamsize>(size));
  if (!os) {
    throw std::runtime_error("Failed to write graph state string");
  }
}

std::string read_string(std::istream &os) {
  const size_t size = read_binary<size_t>(os);
  std::string value(size, '\0');
  os.read(value.data(), static_cast<std::streamsize>(size));
  if (!os) {
    throw std::runtime_error("Failed to read graph state string");
  }
  return value;
}

Layer load_layer_config(std::istream &os) {
  nlohmann::json config_json = nlohmann::json::parse(read_string(os));
  LayerConfig config = LayerConfig::from_json(config_json);
  LayerFactory::register_defaults();
  return LayerFactory::create(config);
}

void save_layer_config(std::ostream &os, const std::shared_ptr<LayerImpl> &layer) {
  nlohmann::json config_json = layer->get_config().to_json();
  write_string(os, config_json.dump());
}

void save_node_index_set(std::ostream &os, const std::set<Node> &nodes,
                         const std::unordered_map<NodeImpl *, size_t> &node_indices,
                         const char *context) {
  write_binary(os, nodes.size());
  for (const auto &node : nodes) {
    const auto it = node_indices.find(node.get());
    if (it == node_indices.end()) {
      throw std::runtime_error(std::string("Internal error while saving graph ") + context +
                               " node state");
    }
    write_binary(os, it->second);
  }
}

void load_node_index_set(std::istream &os, Graph &graph, const Vec<Node> &nodes,
                         const char *context, const std::function<void(const Node &)> &mark_node) {
  const size_t count = read_binary<size_t>(os);
  for (size_t i = 0; i < count; ++i) {
    const size_t node_index = read_binary<size_t>(os);
    if (node_index >= nodes.size()) {
      throw std::runtime_error(std::string("Graph state references an invalid ") + context +
                               " node index");
    }
    mark_node(nodes[node_index]);
  }
}

}  // namespace

// TODO : save params like io dtype, param dtype, compute dtype to preserve precision.
void Graph::save_state(std::ostream &os) const {
  if (!os) {
    throw std::runtime_error("Stream is not ready for writing");
  }

  std::unordered_map<NodeImpl *, size_t> node_indices;
  node_indices.reserve(nodes_.size());
  for (size_t i = 0; i < nodes_.size(); ++i) {
    node_indices.emplace(nodes_[i].get(), i);
  }

  std::unordered_map<LayerImpl *, size_t> layer_indices;
  layer_indices.reserve(edges_.size());
  Vec<std::shared_ptr<LayerImpl>> unique_layers;
  unique_layers.reserve(edges_.size());
  for (const auto &edge : edges_) {
    LayerImpl *layer_ptr = edge->layer().get();
    if (layer_indices.count(layer_ptr) == 0) {
      layer_indices.emplace(layer_ptr, unique_layers.size());
      unique_layers.push_back(edge->layer());
    }
  }

  os.write(kGraphStateMagic.data(), static_cast<std::streamsize>(kGraphStateMagic.size()));
  if (!os) {
    throw std::runtime_error("Failed to write graph state header");
  }
  write_binary(os, kGraphStateVersion);

  write_binary(os, static_cast<int>(io_dtype_));
  write_binary(os, static_cast<int>(param_dtype_));
  write_binary(os, static_cast<int>(compute_dtype_));

  write_binary(os, nodes_.size());
  for (const auto &node : nodes_) {
    write_string(os, node->uid());
  }

  save_node_index_set(os, input_nodes_, node_indices, "input");
  save_node_index_set(os, output_nodes_, node_indices, "output");

  write_binary(os, unique_layers.size());
  for (const auto &layer : unique_layers) {
    save_layer_config(os, layer);
  }

  write_binary(os, edges_.size());
  for (const auto &edge : edges_) {
    auto layer_it = layer_indices.find(edge->layer().get());
    if (layer_it == layer_indices.end()) {
      throw std::runtime_error("Internal error while saving graph state");
    }
    write_binary(os, layer_it->second);

    write_binary(os, edge->producers().size());
    for (const auto &producer : edge->producers()) {
      auto node_it = node_indices.find(producer.get());
      if (node_it == node_indices.end()) {
        throw std::runtime_error("Graph edge producer is not registered as a node");
      }
      write_binary(os, node_it->second);
    }

    write_binary(os, edge->consumers().size());
    for (const auto &consumer : edge->consumers()) {
      auto node_it = node_indices.find(consumer.get());
      if (node_it == node_indices.end()) {
        throw std::runtime_error("Graph edge consumer is not registered as a node");
      }
      write_binary(os, node_it->second);
    }
  }

  write_binary(os, unique_layers.size());
  for (const auto &layer : unique_layers) {
    Vec<Param> params = layer->params();
    write_binary(os, params.size());
    for (const Param &param : params) {
      if (!param) {
        throw std::runtime_error("Cannot save uninitialized layer parameter");
      }
      save(param.data(), os);
      bool has_grad = static_cast<bool>(param.grad());
      os.write(reinterpret_cast<const char *>(&has_grad), sizeof(bool));
      if (has_grad) {
        save(param.grad(), os);
      }
    }
  }
}

Graph Graph::load_state(std::istream &os, IAllocator &allocator) {
  if (!os) {
    throw std::runtime_error("Stream is not ready for reading");
  }

  std::array<char, 4> magic{};
  os.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!os) {
    throw std::runtime_error("Failed to read graph state header");
  }
  if (magic != kGraphStateMagic) {
    throw std::runtime_error("Invalid graph state file header");
  }

  const std::uint32_t version = read_binary<std::uint32_t>(os);
  if (version != 1 && version != kGraphStateVersion && version != 3) {
    throw std::runtime_error("Unsupported graph state version: " + std::to_string(version));
  }

  Graph graph;

  if (version >= 4) {
    graph.io_dtype_ = static_cast<DType_t>(read_binary<int>(os));
    graph.param_dtype_ = static_cast<DType_t>(read_binary<int>(os));
    graph.compute_dtype_ = static_cast<DType_t>(read_binary<int>(os));
  }

  const size_t node_count = read_binary<size_t>(os);
  Vec<Node> nodes;
  nodes.reserve(node_count);
  for (size_t i = 0; i < node_count; ++i) {
    nodes.push_back(graph.make_node(read_string(os)));
  }

  if (version >= 2) {
    load_node_index_set(os, graph, nodes, "input",
                        [&graph](const Node &node) { graph.set_input(node); });
    load_node_index_set(os, graph, nodes, "output",
                        [&graph](const Node &node) { graph.set_output(node); });
  }

  const size_t layer_count = read_binary<size_t>(os);
  Vec<std::shared_ptr<LayerImpl>> layers;
  layers.reserve(layer_count);
  for (size_t i = 0; i < layer_count; ++i) {
    Layer layer = load_layer_config(os);
    layers.push_back(static_cast<std::shared_ptr<LayerImpl>>(layer));
  }

  const size_t edge_count = read_binary<size_t>(os);
  for (size_t i = 0; i < edge_count; ++i) {
    const size_t layer_index = read_binary<size_t>(os);
    if (layer_index >= layers.size()) {
      throw std::runtime_error("Graph state references an invalid layer index");
    }

    const size_t producer_count = read_binary<size_t>(os);
    Vec<Node> producers;
    producers.reserve(producer_count);
    for (size_t j = 0; j < producer_count; ++j) {
      const size_t node_index = read_binary<size_t>(os);
      if (node_index >= nodes.size()) {
        throw std::runtime_error("Graph state references an invalid producer node index");
      }
      producers.push_back(nodes[node_index]);
    }

    const size_t consumer_count = read_binary<size_t>(os);
    Vec<Node> consumers;
    consumers.reserve(consumer_count);
    for (size_t j = 0; j < consumer_count; ++j) {
      const size_t node_index = read_binary<size_t>(os);
      if (node_index >= nodes.size()) {
        throw std::runtime_error("Graph state references an invalid consumer node index");
      }
      consumers.push_back(nodes[node_index]);
    }

    graph.add_edge(layers[layer_index], producers, consumers);
  }

  GraphOpts opts{
      .io_dtype = graph.io_dtype_,
      .param_dtype = graph.param_dtype_,
      .compute_dtype = graph.compute_dtype_,
  };
  graph.compile(allocator, opts);

  const size_t param_layer_count = read_binary<size_t>(os);
  if (param_layer_count != layers.size()) {
    throw std::runtime_error("Graph state parameter section does not match layer count");
  }

  for (size_t i = 0; i < layers.size(); ++i) {
    Vec<Param> params = layers[i]->params();
    const size_t param_count = read_binary<size_t>(os);
    if (param_count != params.size()) {
      throw std::runtime_error("Graph state parameter count does not match layer definition");
    }
    for (auto &param : params) {
      load(param.data(), os);
      if (version >= 5) {
        bool has_grad;
        os.read(reinterpret_cast<char *>(&has_grad), sizeof(bool));
        if (has_grad) {
          load(param.grad(), os);
        } else {
          param.grad() = Tensor();
        }
      }
    }
  }

  return graph;
}

}  // namespace tunx