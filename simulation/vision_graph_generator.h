#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

#include "graph.h"
#include "macro_solver.h"
#include "allocator.h"
#include "graph_executor.h"

// Forward declaration of simulation helpers
std::vector<std::string> find_fw_fork_join_execution_order(Graph& graph, size_t max_states = 100000, bool* oracle_complete = nullptr);

struct Shape4D {
  size_t n, c, h, w;

  size_t bytes(size_t dtype_bytes = 4) const {
    return n * c * h * w * dtype_bytes;
  }
};

struct OpProfile {
  double workspace_ratio;
  double residual_ratio;
  bool cache_input;
};

const OpProfile kConv       {0.50, 0.25, true};
const OpProfile kBatchNorm  {0.05, 0.10, true};
const OpProfile kRelu       {0.00, 0.125, false};
const OpProfile kAvgPool    {0.00, 0.00, false};
const OpProfile kAdd        {0.00, 0.00, false};
const OpProfile kConcat     {0.05, 0.00, false};
const OpProfile kTranspose  {0.50, 0.00, true};

struct IRTensor {
  std::string id;
  size_t size;
};

struct IROp {
  std::string semantic_name;
  std::string randomized_id;
  OpProfile profile;
  std::vector<int> inputs;
  std::vector<int> outputs;
  size_t workspace;
  size_t residual;
};

struct IRGraph {
  std::vector<IRTensor> tensors;
  std::vector<IROp> ops;
  std::vector<int> graph_inputs;
  std::vector<int> graph_outputs;
};

struct VTensor {
  int id;
  Shape4D shape;
};

class VisionBuilder {
 public:
  VisionBuilder(uint32_t seed) : rng_(seed) {}

  std::string random_id(const std::string& prefix) {
    return prefix + "_" + std::to_string(rng_());
  }
  
  uint32_t random_between(uint32_t low, uint32_t high) {
    std::uniform_int_distribution<uint32_t> dist(low, high);
    return dist(rng_);
  }

  VTensor add_input(const std::string& name, Shape4D shape) {
    int id = ir_.tensors.size();
    ir_.tensors.push_back({random_id(name), shape.bytes()});
    ir_.graph_inputs.push_back(id);
    return {id, shape};
  }

  void add_graph_output(const VTensor& tensor) {
    ir_.graph_outputs.push_back(tensor.id);
  }

  void set_outputs(const std::vector<VTensor>& tensors) {
    ir_.graph_outputs.clear();
    for (const auto& t : tensors) {
      ir_.graph_outputs.push_back(t.id);
    }
  }

  VTensor unary(const std::string& name,
                const VTensor& input,
                Shape4D output_shape,
                const OpProfile& profile) {
    int out_id = ir_.tensors.size();
    ir_.tensors.push_back({random_id(name + "_out"), output_shape.bytes()});

    const size_t workspace = static_cast<size_t>(profile.workspace_ratio * output_shape.bytes());
    const size_t residual = static_cast<size_t>(profile.residual_ratio * output_shape.bytes());

    IROp op;
    op.semantic_name = name;
    op.randomized_id = random_id(name);
    op.profile = profile;
    op.inputs = {input.id};
    op.outputs = {out_id};
    op.workspace = workspace;
    op.residual = residual;
    ir_.ops.push_back(op);

    return {out_id, output_shape};
  }

  VTensor add_or_concat(const std::string& name, const std::vector<VTensor>& inputs, bool is_add = true) {
    Shape4D out_shape = inputs[0].shape;
    if (!is_add) {
        out_shape.c = 0;
        for (const auto& t : inputs) out_shape.c += t.shape.c;
    }
    
    int out_id = ir_.tensors.size();
    ir_.tensors.push_back({random_id(name + "_out"), out_shape.bytes()});

    const OpProfile& profile = is_add ? kAdd : kConcat;
    const size_t workspace = static_cast<size_t>(profile.workspace_ratio * out_shape.bytes());
    const size_t residual = static_cast<size_t>(profile.residual_ratio * out_shape.bytes());

    IROp op;
    op.semantic_name = name;
    op.randomized_id = random_id(name);
    op.profile = profile;
    for (const auto& t : inputs) op.inputs.push_back(t.id);
    op.outputs = {out_id};
    op.workspace = workspace;
    op.residual = residual;
    ir_.ops.push_back(op);

    return {out_id, out_shape};
  }
  
  std::vector<VTensor> split(const std::string& name, const VTensor& input, size_t branches) {
      std::vector<VTensor> outputs;
      IROp op;
      op.semantic_name = name;
      op.randomized_id = random_id(name);
      op.profile = OpProfile{0.05, 0.0, false};
      op.inputs = {input.id};
      op.workspace = static_cast<size_t>(0.05 * input.shape.bytes());
      op.residual = 0;

      for(size_t i = 0; i < branches; ++i) {
          const int output_id = static_cast<int>(ir_.tensors.size());
          ir_.tensors.push_back({random_id(name + "_out_" + std::to_string(i)), input.shape.bytes()});
          op.outputs.push_back(output_id);
          outputs.push_back({output_id, input.shape});
      }
      ir_.ops.push_back(std::move(op));
      return outputs;
  }

  IRGraph take_ir() { return std::move(ir_); }
  std::mt19937& rng() { return rng_; }

 private:
  IRGraph ir_;
  std::mt19937 rng_;
};

// Vision Primitives
inline OpProfile sample_atomic_profile(VisionBuilder& b) {
  static const std::vector<double> workspace = {0.0, 0.125, 0.25, 0.5, 1.0};
  static const std::vector<double> residual = {0.0, 0.125, 0.25, 0.5};

  return {
      workspace[b.random_between(0, workspace.size() - 1)],
      residual[b.random_between(0, residual.size() - 1)],
      b.random_between(0, 1) == 1
  };
}

inline VTensor atomic_block(VisionBuilder& b, const VTensor& input) {
    return b.unary("atomic_block", input, input.shape, sample_atomic_profile(b));
}

inline VTensor normalize_for_add(VisionBuilder& b, const VTensor& input, Shape4D target) {
  if (input.shape.c == target.c &&
      input.shape.h == target.h &&
      input.shape.w == target.w) {
    return input;
  }
  return b.unary("projection", input, target, kConv);
}

inline VTensor conv_bn_relu(VisionBuilder& b, VTensor in, size_t out_channels) {
  Shape4D out_shape = {in.shape.n, out_channels, in.shape.h, in.shape.w};
  auto c = b.unary("conv", in, out_shape, kConv);
  auto bn = b.unary("bn", c, out_shape, kBatchNorm);
  return b.unary("relu", bn, out_shape, kRelu);
}

inline VTensor expand_reduce(VisionBuilder& b, const VTensor& input, size_t scale, const std::string& name) {
  Shape4D expanded_shape = input.shape;
  expanded_shape.h *= scale;
  expanded_shape.w *= scale;
  VTensor expanded = b.unary(name + "_expand", input, expanded_shape, kTranspose);
  VTensor reduced = b.unary(name + "_reduce", expanded, input.shape, kAvgPool);
  return reduced;
}

inline VTensor random_block(VisionBuilder& b, const VTensor& input) {
    uint32_t type = b.random_between(0, 2);
    if (type == 0) {
        return conv_bn_relu(b, input, b.random_between(64, 256));
    } else if (type == 1) {
        auto c = b.unary("conv", input, input.shape, kConv);
        return b.unary("relu", c, input.shape, kRelu);
    } else {
        auto p = b.unary("pool", input, input.shape, kAvgPool);
        return b.unary("conv", p, input.shape, kConv);
    }
}

inline VTensor random_sequence(VisionBuilder& b, VTensor input, int length = 3) {
    for (int i = 0; i < length; ++i) {
        input = random_block(b, input);
    }
    return input;
}

// Grammars
enum class MotifTarget {
  Rank,
  Linear,
  Branch,
  Join,
  ForkJoin
};

inline IRGraph build_rank_candidate(uint32_t seed, bool scaled = false) {
  VisionBuilder b(seed);
  Shape4D canonical = {1, 128, 28, 28};
  if (!scaled) {
    for (size_t i = 0; i < b.random_between(4, 6); ++i) {
      auto input = b.add_input("input", canonical);
      auto output = atomic_block(b, input);
      b.add_graph_output(output);
    }
  } else {
    auto input = b.add_input("input", canonical);
    auto p1 = expand_reduce(b, input, 4, "er");
    auto p2 = random_sequence(b, input, 5);
    b.set_outputs({p1, p2});
  }
  return b.take_ir();
}

inline IRGraph build_linear_candidate(uint32_t seed, bool scaled = false) {
  VisionBuilder b(seed);
  Shape4D canonical = {1, 64, 28, 28};
  auto x1 = b.add_input("stream", canonical);
  auto x2 = b.add_input("stream", canonical);
  auto x3 = b.add_input("stream", canonical);

  if (!scaled) {
      auto a1 = b.unary("expand", x1, canonical, kTranspose);
      auto a2 = b.unary("reduce", a1, canonical, kAvgPool);
      auto c1 = atomic_block(b, x2);
      auto c2 = atomic_block(b, c1);
      auto d1 = atomic_block(b, x3);
      auto d2 = atomic_block(b, d1);
      b.set_outputs({a2, c2, d2});
  } else {
      auto a = random_sequence(b, x1, 4);
      auto c = random_sequence(b, x2, 4);
      auto d = random_sequence(b, x3, 4);
      b.set_outputs({a, c, d});
  }
  return b.take_ir();
}

inline VTensor random_branch_subtree(VisionBuilder& b, VTensor input, int depth) {
    if (depth <= 0) return random_block(b, input);
    
    if (b.random_between(0, 1) == 0) {
        return random_branch_subtree(b, random_block(b, input), depth - 1);
    } else {
        auto branch_inputs = b.split("split", input, b.random_between(2, 3));
        std::vector<VTensor> outputs;
        for (auto& bi : branch_inputs) {
            outputs.push_back(random_branch_subtree(b, bi, depth - 1));
        }
        return b.add_or_concat("concat", outputs, false);
    }
}

inline IRGraph build_branch_candidate(uint32_t seed, bool scaled = false) {
  VisionBuilder b(seed);
  Shape4D canonical = {1, 128, 28, 28};
  Shape4D small = {1, 16, 28, 28};
  auto input = b.add_input("input", canonical);
  auto distractor = b.add_input("distractor", canonical);
  
  if (!scaled) {
    auto fork = b.split("fork", input, 3);
    for (auto child : fork) {
      b.add_graph_output(b.unary("reduce", child, small, kConv));
    }
    b.add_graph_output(atomic_block(b, distractor));
  } else {
    auto root = random_block(b, input);
    auto branch_inputs = b.split("split", root, 3);
    std::vector<VTensor> outs;
    for (auto bi : branch_inputs) {
      outs.push_back(random_sequence(b, bi, 3));
    }
    outs.push_back(random_sequence(b, distractor, 3));
    b.set_outputs(outs);
  }
  return b.take_ir();
}

inline IRGraph build_join_candidate(uint32_t seed, bool scaled = false) {
    VisionBuilder b(seed);
    Shape4D canonical = {1, 128, 28, 28};
    Shape4D small = {1, 16, 28, 28};
    auto distractor = b.add_input("distractor", canonical);
    
    if (!scaled) {
        std::vector<VTensor> predecessors;
        for (size_t i = 0; i < 3; ++i) {
          auto input = b.add_input("join_input", small);
          predecessors.push_back(b.unary("expand", input, canonical, kConv));
        }
        auto joined = b.add_or_concat("join", predecessors, true);
        auto output = b.unary("reduce", joined, small, kConv);
        b.set_outputs({output, atomic_block(b, distractor)});
    } else {
        std::vector<VTensor> predecessors;
        for (size_t i = 0; i < 3; ++i) {
          auto input = b.add_input("join_input", canonical);
          predecessors.push_back(random_sequence(b, input, 3));
        }
        std::vector<VTensor> normalized;
        for(auto& p : predecessors) {
            normalized.push_back(normalize_for_add(b, p, canonical));
        }
        auto joined = b.add_or_concat("join", normalized, true);
        auto output = random_block(b, joined);
        b.set_outputs({output, random_sequence(b, distractor, 3)});
    }
    return b.take_ir();
}

inline VTensor nested_fork_join(VisionBuilder& b, VTensor input, int depth, Shape4D target) {
  if (depth == 0)
    return random_sequence(b, input, 1);

  auto branches = b.split("split", input, b.random_between(2, 3));
  branches[0] = nested_fork_join(b, branches[0], depth - 1, target);

  for (size_t i = 1; i < branches.size(); ++i)
    branches[i] = random_sequence(b, branches[i], 1);

  std::vector<VTensor> normalized;
  for(auto& branch : branches) {
      normalized.push_back(normalize_for_add(b, branch, target));
  }
  return b.add_or_concat("join", normalized, true);
}

inline IRGraph build_fork_join_candidate(uint32_t seed, bool scaled = false) {
    VisionBuilder b(seed);
    Shape4D canonical = {1, 64, 28, 28};
    auto input = b.add_input("input", canonical);
    
    if (!scaled) {
        auto outer = b.split("outer_fork", input, 2);
        auto inner = b.split("inner_fork", outer[0], 2);

        auto i0 = atomic_block(b, inner[0]);
        auto i1 = atomic_block(b, inner[1]);
        auto inner_join = b.add_or_concat("inner_join", {i0, i1}, true);

        auto other = atomic_block(b, outer[1]);
        auto final_out = b.add_or_concat("outer_join", {inner_join, other}, true);
        
        auto distractor_in = b.add_input("distractor", canonical);
        auto distractor_out = atomic_block(b, distractor_in);
        
        b.set_outputs({final_out, distractor_out});
    } else {
        auto main_out = nested_fork_join(b, input, 2, canonical);
        auto distractor_in = b.add_input("distractor_in", canonical);
        auto distractor_out = random_sequence(b, distractor_in, 3);
        b.set_outputs({main_out, distractor_out});
    }
    
    return b.take_ir();
}

inline IRGraph build_candidate(MotifTarget target, uint32_t seed, bool scaled = false) {
    switch (target) {
        case MotifTarget::Rank: return build_rank_candidate(seed, scaled);
        case MotifTarget::Linear: return build_linear_candidate(seed, scaled);
        case MotifTarget::Branch: return build_branch_candidate(seed, scaled);
        case MotifTarget::Join: return build_join_candidate(seed, scaled);
        case MotifTarget::ForkJoin: return build_fork_join_candidate(seed, scaled);
    }
    return IRGraph();
}

inline Graph materialize(const IRGraph& ir) {
    Graph g;
    std::map<int, ActivationNode*> tensors;
    
    for (size_t i = 0; i < ir.tensors.size(); ++i) {
        tensors[i] = g.add_act(ir.tensors[i].id, ir.tensors[i].size);
    }
    
    std::vector<ActivationNode*> inputs;
    for (int id : ir.graph_inputs) inputs.push_back(tensors[id]);
    g.set_inputs(inputs);
    
    for (const auto& op : ir.ops) {
        std::vector<ActivationNode*> in_nodes, out_nodes, cache_nodes;
        for (int id : op.inputs) {
            in_nodes.push_back(tensors[id]);
            if (op.profile.cache_input) cache_nodes.push_back(tensors[id]);
        }
        for (int id : op.outputs) out_nodes.push_back(tensors[id]);
        
        g.add_op(op.randomized_id, op.workspace, in_nodes, out_nodes, cache_nodes, op.residual);
    }
    
    std::vector<ActivationNode*> outputs;
    for (int id : ir.graph_outputs) outputs.push_back(tensors[id]);
    g.set_outputs(outputs);
    
    return g;
}

inline void randomize_ids_and_materialization(IRGraph& ir, uint32_t seed) {
    std::mt19937 rng(seed);
    std::shuffle(ir.ops.begin(), ir.ops.end(), rng);
}

struct AblationPeaks {
  size_t dfs;
  size_t rank;
  size_t linear;
  size_t branch;
  size_t join;
  size_t flat_branch_join;
  size_t full;
  size_t oracle;
  bool oracle_complete;
};

// Simulation Helpers
extern std::vector<std::string> find_fw_naive_dfs_execution_order(Graph& graph);
extern std::vector<std::string> find_fw_ranked_execution_order(Graph& graph, std::ostream* os = nullptr);
extern std::vector<std::string> find_fw_linear_execution_order(Graph& graph, std::ostream* os = nullptr);
extern std::vector<std::string> find_fw_branching_execution_order(Graph& graph, std::ostream* os = nullptr);
extern std::vector<std::string> find_fw_joining_execution_order(Graph& graph, std::ostream* os = nullptr);

inline std::vector<std::string> find_fw_flat_bj_execution_order(Graph& graph, std::ostream* os = nullptr) {
    FlatBJSolver solver(graph, os);
    return solver.find_forward_order();
}

inline std::vector<std::string> find_fw_full_execution_order(Graph& graph, std::ostream* os = nullptr) {
    FullSolver solver(graph, os);
    return solver.find_forward_order();
}

extern std::map<std::string, double> rank_execution_orders(
    Graph& g, const std::map<std::string, std::vector<std::string>>& orders);

inline size_t evaluate_peak(Graph& g, const std::vector<std::string>& order) {
    if (order.empty()) return std::numeric_limits<size_t>::max();
    Allocator allocator;
    GraphExecutor executor(g);
    size_t peak = 0;
    allocator.subscribe("peak", [&](size_t new_mem) {
      if (new_mem > peak) peak = new_mem;
    });
    executor.init_boundaries(&allocator);
    for (const auto& op_id : order) {
        executor.run_op_node(&g.get_op(op_id), &allocator);
    }
    allocator.unsubscribe("peak");
    return peak;
}

inline AblationPeaks evaluate_ablations(Graph& g) {
    std::map<std::string, std::vector<std::string>> orders = {
        {"DFS", find_fw_naive_dfs_execution_order(g)},
        {"RANK", find_fw_ranked_execution_order(g)},
        {"LINEAR", find_fw_linear_execution_order(g)},
        {"BRANCH", find_fw_branching_execution_order(g)},
        {"JOIN", find_fw_joining_execution_order(g)},
        {"FLAT_BJ", find_fw_flat_bj_execution_order(g)},
        {"FULL", find_fw_full_execution_order(g)}
    };
    
    AblationPeaks peaks;
    peaks.dfs = evaluate_peak(g, orders["DFS"]);
    peaks.rank = evaluate_peak(g, orders["RANK"]);
    peaks.linear = evaluate_peak(g, orders["LINEAR"]);
    peaks.branch = evaluate_peak(g, orders["BRANCH"]);
    peaks.join = evaluate_peak(g, orders["JOIN"]);
    peaks.flat_branch_join = evaluate_peak(g, orders["FLAT_BJ"]);
    peaks.full = evaluate_peak(g, orders["FULL"]);
    peaks.oracle = 0;
    peaks.oracle_complete = false;
    
    return peaks;
}

inline bool meaningfully_better(size_t better, size_t worse, double gap = 0.03) {
  return static_cast<double>(better) * (1.0 + gap) <= static_cast<double>(worse);
}

inline bool passes_cheap_separator(MotifTarget target, const AblationPeaks& p, double gap = 0.03) {
  switch (target) {
    case MotifTarget::Rank:
      return meaningfully_better(p.rank, p.dfs, gap);
    case MotifTarget::Linear:
      return meaningfully_better(p.linear, p.rank, gap);
    case MotifTarget::Branch:
      return meaningfully_better(p.branch, p.linear, gap) &&
             meaningfully_better(p.branch, p.join, gap);
    case MotifTarget::Join:
      return meaningfully_better(p.join, p.linear, gap) &&
             meaningfully_better(p.join, p.branch, gap);
    case MotifTarget::ForkJoin:
      return meaningfully_better(p.full, p.flat_branch_join, gap);
  }
  return false;
}

struct GeneratedGraph {
    Graph graph;
    uint32_t seed;
    AblationPeaks peaks;
};

inline GeneratedGraph generate_witness(MotifTarget target, uint32_t initial_seed) {
  constexpr uint32_t kMaxAttempts = 10000;
  constexpr size_t kDiscoveryStateLimit = 50000;
  constexpr size_t kMaximumOps = 16;

  for (uint32_t attempt = 0; attempt < kMaxAttempts; ++attempt) {
    const uint32_t seed = initial_seed + attempt;

    IRGraph ir = build_candidate(target, seed, false);
    randomize_ids_and_materialization(ir, seed);

    if (ir.ops.size() > kMaximumOps)
      continue;

    Graph graph = materialize(ir);
    AblationPeaks peaks = evaluate_ablations(graph);

    if (!passes_cheap_separator(target, peaks))
      continue;

    bool complete = false;
    auto oracle_order = find_fw_fork_join_execution_order(graph, kDiscoveryStateLimit, &complete);

    if (!complete)
      continue;

    peaks.oracle = evaluate_peak(graph, oracle_order);
    peaks.oracle_complete = true;

    // Strict validation
    bool valid = false;
    switch (target) {
        case MotifTarget::Rank: valid = peaks.rank == peaks.oracle; break;
        case MotifTarget::Linear: valid = peaks.linear == peaks.oracle; break;
        case MotifTarget::Branch: valid = peaks.branch == peaks.oracle; break;
        case MotifTarget::Join: valid = peaks.join == peaks.oracle; break;
        case MotifTarget::ForkJoin: valid = peaks.full == peaks.oracle; break;
    }
    
    if (valid) {
      return {std::move(graph), seed, peaks};
    }
  }

  throw std::runtime_error("Unable to generate motif witness within the attempt budget.");
}
