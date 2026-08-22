#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "allocator.h"
#include "dp_solver.h"
#include "graph.h"
#include "graph_executor.h"
#include "macro_solver.h"

struct Shape4D {
  size_t n, c, h, w;

  size_t bytes(size_t dtype_bytes = 4) const { return n * c * h * w * dtype_bytes; }

  bool operator==(const Shape4D& other) const {
    return n == other.n && c == other.c && h == other.h && w == other.w;
  }
  
  bool operator!=(const Shape4D& other) const {
    return !(*this == other);
  }
};

struct OpProfile {
  double workspace_ratio;
  double residual_ratio;
  bool cache_input;
};

const OpProfile kConv{0.40, 0.0, true};
const OpProfile kBatchNorm{0.1, 0.01, true};
const OpProfile kRelu{0.00, 0.0, false};
const OpProfile kAvgPool{0.00, 0.00, false};
const OpProfile kAdd{0.00, 0.00, false};
const OpProfile kConcat{0.05, 0.00, false};
const OpProfile kTranspose{0.25, 0.00, true};

struct IRTensor {
  std::string id;
  Shape4D shape;
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
  VisionBuilder(uint32_t seed)
      : rng_(seed) {}

  void set_context(int branch, int depth) {
    current_branch_ = branch;
    current_depth_ = depth;
  }

  std::string unique_id(const std::string& prefix) {
    std::string name = prefix;
    if (current_branch_ != -1) name += "_b" + std::to_string(current_branch_);
    if (current_depth_ != -1) name += "_d" + std::to_string(current_depth_);
    
    int count = name_counts_[name]++;
    if (count == 0) return name;
    return name + "_" + std::to_string(count);
  }

  uint32_t random_between(uint32_t low, uint32_t high) {
    std::uniform_int_distribution<uint32_t> dist(low, high);
    return dist(rng_);
  }

  VTensor add_input(const std::string& name, Shape4D shape) {
    int id = ir_.tensors.size();
    ir_.tensors.push_back({unique_id(name), shape});
    ir_.graph_inputs.push_back(id);
    return {id, shape};
  }

  void add_graph_output(const VTensor& tensor) { ir_.graph_outputs.push_back(tensor.id); }

  void set_outputs(const std::vector<VTensor>& tensors) {
    ir_.graph_outputs.clear();
    for (const auto& t : tensors) {
      ir_.graph_outputs.push_back(t.id);
    }
  }

  VTensor unary(const std::string& name, const VTensor& input, Shape4D output_shape,
                const OpProfile& profile) {
    int out_id = ir_.tensors.size();
    ir_.tensors.push_back({unique_id(name + "_out"), output_shape});

    const size_t workspace = static_cast<size_t>(profile.workspace_ratio * output_shape.bytes());
    const size_t residual = static_cast<size_t>(profile.residual_ratio * output_shape.bytes());

    IROp op;
    op.semantic_name = name;
    op.randomized_id = unique_id(name);
    op.profile = profile;
    op.inputs = {input.id};
    op.outputs = {out_id};
    op.workspace = workspace;
    op.residual = residual;
    ir_.ops.push_back(op);

    return {out_id, output_shape};
  }

  VTensor add_or_concat(const std::string& name, const std::vector<VTensor>& inputs,
                        bool is_add = true) {
    Shape4D out_shape = inputs[0].shape;
    if (!is_add) {
      out_shape.c = 0;
      for (const auto& t : inputs) out_shape.c += t.shape.c;
    }

    int out_id = ir_.tensors.size();
    ir_.tensors.push_back({unique_id(name + "_out"), out_shape});

    const OpProfile& profile = is_add ? kAdd : kConcat;
    const size_t workspace = static_cast<size_t>(profile.workspace_ratio * out_shape.bytes());
    const size_t residual = static_cast<size_t>(profile.residual_ratio * out_shape.bytes());

    IROp op;
    op.semantic_name = name;
    op.randomized_id = unique_id(name);
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
    op.randomized_id = unique_id(name);
    op.profile = OpProfile{0.05, 0.0, false};
    op.inputs = {input.id};
    op.workspace = static_cast<size_t>(0.05 * input.shape.bytes());
    op.residual = 0;

    for (size_t i = 0; i < branches; ++i) {
      const int output_id = static_cast<int>(ir_.tensors.size());
      ir_.tensors.push_back({unique_id(name + "_out_" + std::to_string(i)), input.shape});
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
  std::map<std::string, int> name_counts_;
  int current_branch_ = -1;
  int current_depth_ = -1;
};

// Vision Primitives
inline OpProfile sample_atomic_profile(VisionBuilder& b) {
  static const std::vector<double> workspace = {0.0, 0.125, 0.25, 0.5, 1.0};
  static const std::vector<double> residual = {0.0, 0.125, 0.25, 0.5};

  return {workspace[b.random_between(0, workspace.size() - 1)],
          residual[b.random_between(0, residual.size() - 1)], b.random_between(0, 1) == 1};
}

inline VTensor atomic_block(VisionBuilder& b, const VTensor& input) {
  return b.unary("atomic_block", input, input.shape, sample_atomic_profile(b));
}

// Grammars
enum class MotifTarget { Rank, Linear, Branch, Join, ForkJoin };

inline IRGraph build_rank_candidate(uint32_t seed) {
  VisionBuilder b(seed);
  Shape4D canonical = {1, 128, 28, 28};
  for (size_t i = 0; i < b.random_between(4, 6); ++i) {
    auto input = b.add_input("input", canonical);
    auto output = atomic_block(b, input);
    b.add_graph_output(output);
  }
  return b.take_ir();
}

inline IRGraph build_linear_candidate(uint32_t seed) {
  VisionBuilder b(seed);
  Shape4D canonical = {1, 64, 28, 28};
  auto x1 = b.add_input("stream", canonical);
  auto x2 = b.add_input("stream", canonical);
  auto x3 = b.add_input("stream", canonical);

  auto a1 = b.unary("expand", x1, canonical, kTranspose);
  auto a2 = b.unary("reduce", a1, canonical, kAvgPool);
  auto c1 = atomic_block(b, x2);
  auto c2 = atomic_block(b, c1);
  auto d1 = atomic_block(b, x3);
  auto d2 = atomic_block(b, d1);
  b.set_outputs({a2, c2, d2});
  return b.take_ir();
}

inline IRGraph build_branch_candidate(uint32_t seed) {
  VisionBuilder b(seed);
  Shape4D canonical = {1, 128, 28, 28};
  Shape4D reduced = {1, 128, 7, 7};
  auto input = b.add_input("input", canonical);
  auto distractor = b.add_input("distractor", canonical);

  auto fork = b.split("fork", input, 3);
  for (size_t i = 0; i < fork.size(); ++i) {
    b.set_context(i, -1);
    auto output = b.unary("avgpool_reduce", fork[i], reduced, kAvgPool);
    b.add_graph_output(output);
  }
  b.set_context(-1, -1);
  b.add_graph_output(atomic_block(b, distractor));
  return b.take_ir();
}

inline IRGraph build_join_candidate(uint32_t seed) {
  VisionBuilder b(seed);
  Shape4D canonical = {1, 128, 28, 28};
  Shape4D small = {1, 16, 28, 28};
  auto distractor = b.add_input("distractor", canonical);

  std::vector<VTensor> predecessors;
  for (size_t i = 0; i < 3; ++i) {
    b.set_context(i, -1);
    auto input = b.add_input("join_input", small);
    predecessors.push_back(b.unary("expand", input, canonical, kConv));
  }
  b.set_context(-1, -1);
  auto joined = b.add_or_concat("join", predecessors, true);
  auto output = b.unary("reduce", joined, small, kConv);
  b.set_outputs({output, atomic_block(b, distractor)});
  return b.take_ir();
}

inline IRGraph build_fork_join_candidate(uint32_t seed) {
  VisionBuilder b(seed);
  Shape4D canonical = {1, 64, 28, 28};
  auto input = b.add_input("input", canonical);

  auto outer = b.split("outer_fork", input, 2);
  auto inner = b.split("inner_fork", outer[0], 2);

  b.set_context(0, -1);
  auto i0 = atomic_block(b, inner[0]);
  b.set_context(1, -1);
  auto i1 = atomic_block(b, inner[1]);
  b.set_context(-1, -1);
  auto inner_join = b.add_or_concat("inner_join", {i0, i1}, true);

  b.set_context(1, -1);
  auto other = atomic_block(b, outer[1]);
  b.set_context(-1, -1);
  auto final_out = b.add_or_concat("outer_join", {inner_join, other}, true);

  auto distractor_in = b.add_input("distractor", canonical);
  auto distractor_out = atomic_block(b, distractor_in);

  b.set_outputs({final_out, distractor_out});

  return b.take_ir();
}

inline IRGraph build_candidate(MotifTarget target, uint32_t seed) {
  switch (target) {
    case MotifTarget::Rank:
      return build_rank_candidate(seed);
    case MotifTarget::Linear:
      return build_linear_candidate(seed);
    case MotifTarget::Branch:
      return build_branch_candidate(seed);
    case MotifTarget::Join:
      return build_join_candidate(seed);
    case MotifTarget::ForkJoin:
      return build_fork_join_candidate(seed);
  }
  return IRGraph();
}

inline Graph materialize(const IRGraph& ir) {
  Graph g;
  std::map<int, ActivationNode*> tensors;

  for (size_t i = 0; i < ir.tensors.size(); ++i) {
    tensors[i] = g.add_act(ir.tensors[i].id, ir.tensors[i].shape.bytes());
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

namespace vision_scale_detail {

class IRScaler {
public:
  IRScaler(uint32_t seed)
      : rng_(seed) {}

  void set_context(int branch, int depth) {
    current_branch_ = branch;
    current_depth_ = depth;
  }

  std::string unique_id(const std::string& prefix) {
    std::string name = prefix;
    if (current_branch_ != -1) name += "_b" + std::to_string(current_branch_);
    if (current_depth_ != -1) name += "_d" + std::to_string(current_depth_);
    
    int count = name_counts_[name]++;
    if (count == 0) return name;
    return name + "_" + std::to_string(count);
  }

  uint32_t random_between(uint32_t low, uint32_t high) {
    return std::uniform_int_distribution<uint32_t>(low, high)(rng_);
  }

  struct ConfiguredOp {
    std::string name;
    OpProfile profile;
    Shape4D out_shape;
    size_t workspace;
    size_t residual;
  };

  ConfiguredOp sample_vision_op(const Shape4D& in) {
    int op_type = random_between(0, 5);
    if (op_type == 0) { // Conv2D
      size_t oc_choices[] = {std::max<size_t>(16, in.c / 2), in.c, std::min<size_t>(1024, in.c * 2)};
      size_t oc = oc_choices[random_between(0, 2)];
      size_t k = (random_between(0, 1) == 0) ? 1 : 3;
      size_t s = (in.h >= 14 && random_between(0, 4) == 0) ? 2 : 1;
      size_t p = (k - 1) / 2;
      
      Shape4D out = {in.n, oc, (in.h + 2 * p - k) / s + 1, (in.w + 2 * p - k) / s + 1};
      size_t workspace = oc * in.c * k * k * 4; 
      size_t residual = out.bytes();
      
      return {"conv2d_oc" + std::to_string(oc) + "_k" + std::to_string(k) + "_s" + std::to_string(s),
              kConv, out, workspace, residual};
    } else if (op_type == 1) { // Conv2DTranspose
      size_t oc_choices[] = {std::max<size_t>(16, in.c / 2), in.c, std::min<size_t>(1024, in.c * 2)};
      size_t oc = oc_choices[random_between(0, 2)];
      size_t k = (random_between(0, 1) == 0) ? 2 : 4;
      size_t s = (in.h <= 56 && random_between(0, 4) == 0) ? 2 : 1;
      size_t p = (k == 4 && s == 2) ? 1 : 0;
      
      size_t out_h = (in.h - 1) * s - 2 * p + k;
      size_t out_w = (in.w - 1) * s - 2 * p + k;
      Shape4D out = {in.n, oc, out_h, out_w};
      size_t workspace = oc * in.c * k * k * 4; 
      size_t residual = out.bytes();
      
      return {"conv2dtranspose_oc" + std::to_string(oc) + "_k" + std::to_string(k) + "_s" + std::to_string(s),
              kConv, out, workspace, residual};
    } else if (op_type == 2) { // BatchNorm
      size_t workspace = in.c * 4 * 4; 
      size_t residual = in.bytes();
      return {"batchnorm_c" + std::to_string(in.c), kBatchNorm, in, workspace, residual};
    } else if (op_type == 3) { // ReLU
      return {"relu", kRelu, in, 0, in.bytes()};
    } else if (op_type == 4) { // AvgPool
      size_t k = (random_between(0, 1) == 0) ? 2 : 3;
      size_t s = (in.h >= 14 && random_between(0, 3) == 0) ? 2 : 1;
      size_t p = k / 2;
      Shape4D out = {in.n, in.c, (in.h + 2 * p - k) / s + 1, (in.w + 2 * p - k) / s + 1};
      return {"avgpool_k" + std::to_string(k) + "_s" + std::to_string(s), kAvgPool, out, 0, out.bytes()};
    } else { // Transpose
      return {"transpose", kTranspose, in, in.bytes(), in.bytes()};
    }
  }

  ConfiguredOp project_op(const Shape4D& in, const Shape4D& out) {
    size_t k = 1;
    size_t s = 1;
    if (in.h > out.h && out.h > 0) {
      s = in.h / out.h;
    }
    size_t workspace = out.c * in.c * k * k * 4;
    return {"project_to_c" + std::to_string(out.c) + "_h" + std::to_string(out.h) + "_s" + std::to_string(s), kConv, out, workspace, out.bytes()};
  }

  int add_tensor(IRGraph& ir, const std::string& name, Shape4D shape) {
    const int id = static_cast<int>(ir.tensors.size());
    ir.tensors.push_back({unique_id(name), shape});
    return id;
  }

  void add_unary(IRGraph& ir, int input, int output, const ConfiguredOp& conf) {
    ir.ops.push_back({conf.name,
                      unique_id(conf.name),
                      conf.profile,
                      {input},
                      {output},
                      conf.workspace,
                      conf.residual});
  }

  std::vector<int> add_split(IRGraph& ir, int input, size_t fanout, const std::string& name) {
    const Shape4D shape = ir.tensors[input].shape;
    std::vector<int> outputs;
    outputs.reserve(fanout);
    for (size_t i = 0; i < fanout; ++i) {
      outputs.push_back(add_tensor(ir, name + "_out_" + std::to_string(i), shape));
    }

    size_t workspace = shape.bytes() * 0.05; 
    ir.ops.push_back({name,
                      unique_id(name),
                      {0.05, 0.0, false},
                      {input},
                      outputs,
                      workspace,
                      0});
    return outputs;
  }

  void add_join(IRGraph& ir, const std::vector<int>& inputs, int output, const std::string& name) {
    const Shape4D shape = ir.tensors[output].shape;
    size_t workspace = shape.bytes() * kAdd.workspace_ratio;
    size_t residual = shape.bytes() * kAdd.residual_ratio;
    
    ir.ops.push_back({name,
                      unique_id(name),
                      kAdd,
                      inputs,
                      {output},
                      workspace,
                      residual});
  }

private:
  std::mt19937 rng_;
  std::map<std::string, int> name_counts_;
  int current_branch_ = -1;
  int current_depth_ = -1;
};

inline std::vector<bool> tensors_reachable_from_forks(const IRGraph& ir) {
  std::vector<bool> reachable(ir.tensors.size(), false);
  for (const auto& op : ir.ops) {
    if (op.outputs.size() > 1) {
      for (int output : op.outputs) reachable[output] = true;
    }
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto& op : ir.ops) {
      bool reached = false;
      for (int input : op.inputs) reached = reached || reachable[input];
      if (!reached) continue;
      for (int output : op.outputs) {
        if (!reachable[output]) {
          reachable[output] = true;
          changed = true;
        }
      }
    }
  }
  return reachable;
}

inline std::vector<bool> tensors_that_reach_joins(const IRGraph& ir) {
  std::vector<bool> reaches_join(ir.tensors.size(), false);
  for (const auto& op : ir.ops) {
    if (op.inputs.size() > 1) {
      for (int input : op.inputs) reaches_join[input] = true;
    }
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (auto op_it = ir.ops.rbegin(); op_it != ir.ops.rend(); ++op_it) {
      bool reaches = false;
      for (int output : op_it->outputs) reaches = reaches || reaches_join[output];
      if (!reaches) continue;
      for (int input : op_it->inputs) {
        if (!reaches_join[input]) {
          reaches_join[input] = true;
          changed = true;
        }
      }
    }
  }
  return reaches_join;
}

inline bool graph_input_reaches_join(const IRGraph& ir, int graph_input) {
  return tensors_that_reach_joins(ir)[graph_input];
}

inline IRGraph scale_rank(const IRGraph& orig, uint32_t seed) {
  IRGraph scaled;
  IRScaler scaler(seed);
  constexpr size_t kWidthMultiplier = 3;

  for (const auto& op : orig.ops) {
    if (op.inputs.size() != 1 || op.outputs.size() != 1) {
      throw std::invalid_argument("Rank scaling requires independent unary operators.");
    }

    for (size_t replica = 0; replica < kWidthMultiplier; ++replica) {
      scaler.set_context(replica, -1);
      const Shape4D orig_in = orig.tensors[op.inputs[0]].shape;
      const Shape4D orig_out = orig.tensors[op.outputs[0]].shape;
      const int input = scaler.add_tensor(scaled, "rank_input", orig_in);
      
      auto sampled = scaler.sample_vision_op(orig_in);
      if (replica == 0) sampled.profile = op.profile;

      if (sampled.out_shape != orig_out) {
        const int inter = scaler.add_tensor(scaled, "rank_inter", sampled.out_shape);
        scaler.add_unary(scaled, input, inter, sampled);
        auto proj = scaler.project_op(sampled.out_shape, orig_out);
        const int output = scaler.add_tensor(scaled, "rank_output", orig_out);
        scaler.add_unary(scaled, inter, output, proj);
        scaled.graph_inputs.push_back(input);
        scaled.graph_outputs.push_back(output);
      } else {
        const int output = scaler.add_tensor(scaled, "rank_output", orig_out);
        scaler.add_unary(scaled, input, output, sampled);
        scaled.graph_inputs.push_back(input);
        scaled.graph_outputs.push_back(output);
      }
    }
  }
  return scaled;
}

inline IRGraph scale_linear(const IRGraph& orig, uint32_t seed) {
  IRGraph scaled;
  scaled.tensors = orig.tensors;
  scaled.graph_inputs = orig.graph_inputs;
  scaled.graph_outputs = orig.graph_outputs;
  IRScaler scaler(seed);
  constexpr size_t kLengthMultiplier = 3;

  for (const auto& op : orig.ops) {
    if (op.inputs.size() != 1 || op.outputs.size() != 1) {
      scaled.ops.push_back(op);
      continue;
    }

    int current = op.inputs[0];
    const Shape4D target_shape = scaled.tensors[op.outputs[0]].shape;
    for (size_t part = 0; part < kLengthMultiplier; ++part) {
      scaler.set_context(-1, part);
      const bool last = part + 1 == kLengthMultiplier;
      auto sampled = scaler.sample_vision_op(scaled.tensors[current].shape);
      if (part == 0) sampled.profile = op.profile;

      if (last) {
        if (sampled.out_shape != target_shape) {
          const int inter = scaler.add_tensor(scaled, "linear_inter", sampled.out_shape);
          scaler.add_unary(scaled, current, inter, sampled);
          auto proj = scaler.project_op(sampled.out_shape, target_shape);
          scaler.add_unary(scaled, inter, op.outputs[0], proj);
        } else {
          scaler.add_unary(scaled, current, op.outputs[0], sampled);
        }
      } else {
        const int output = scaler.add_tensor(scaled, "linear_t", sampled.out_shape);
        scaler.add_unary(scaled, current, output, sampled);
        current = output;
      }
    }
  }
  return scaled;
}

inline void append_branch_tree(IRGraph& ir, IRScaler& scaler, int input, size_t depth,
                               std::vector<int>& leaves, const Shape4D& target_shape, int branch = -1) {
  scaler.set_context(branch, depth);
  if (depth == 0) {
    auto sampled = scaler.sample_vision_op(ir.tensors[input].shape);
    if (sampled.out_shape != target_shape) {
      const int inter = scaler.add_tensor(ir, "nested_branch_inter", sampled.out_shape);
      scaler.add_unary(ir, input, inter, sampled);
      auto proj = scaler.project_op(sampled.out_shape, target_shape);
      const int leaf = scaler.add_tensor(ir, "nested_branch_leaf", target_shape);
      scaler.add_unary(ir, inter, leaf, proj);
      leaves.push_back(leaf);
    } else {
      const int leaf = scaler.add_tensor(ir, "nested_branch_leaf", target_shape);
      scaler.add_unary(ir, input, leaf, sampled);
      leaves.push_back(leaf);
    }
    return;
  }

  const size_t fanout = scaler.random_between(2, 3);
  const auto children = scaler.add_split(ir, input, fanout, "nested_branch_fork");
  for (size_t i = 0; i < children.size(); ++i) {
    append_branch_tree(ir, scaler, children[i], depth - 1, leaves, target_shape, i);
  }
}

inline IRGraph scale_branch(const IRGraph& orig, uint32_t seed) {
  IRGraph scaled = orig;
  IRScaler scaler(seed);
  constexpr size_t kAdditionalBranchDepth = 2;

  const auto reachable_from_fork = tensors_reachable_from_forks(orig);
  std::vector<int> retained_outputs;
  std::vector<int> expandable_leaves;
  for (int output : orig.graph_outputs) {
    if (reachable_from_fork[output]) {
      expandable_leaves.push_back(output);
    } else {
      retained_outputs.push_back(output);
    }
  }

  if (expandable_leaves.empty()) {
    throw std::invalid_argument("Branch scaling requires at least one fork-derived leaf.");
  }

  scaled.graph_outputs = retained_outputs;
  for (int leaf : expandable_leaves) {
    const Shape4D target = orig.tensors[leaf].shape;
    append_branch_tree(scaled, scaler, leaf, kAdditionalBranchDepth, scaled.graph_outputs, target);
  }
  return scaled;
}

inline void append_join_tree(IRGraph& ir, IRScaler& scaler, int output, size_t depth, int branch = -1) {
  scaler.set_context(branch, depth);
  const Shape4D shape = ir.tensors[output].shape;
  if (depth == 0) {
    const int input = scaler.add_tensor(ir, "nested_join_input", shape);
    ir.graph_inputs.push_back(input);
    auto sampled = scaler.sample_vision_op(shape);
    
    if (sampled.out_shape != shape) {
      const int inter = scaler.add_tensor(ir, "nested_join_inter", sampled.out_shape);
      scaler.add_unary(ir, input, inter, sampled);
      auto proj = scaler.project_op(sampled.out_shape, shape);
      scaler.add_unary(ir, inter, output, proj);
    } else {
      scaler.add_unary(ir, input, output, sampled);
    }
    return;
  }

  const size_t fanin = scaler.random_between(2, 3);
  std::vector<int> predecessors;
  predecessors.reserve(fanin);
  for (size_t i = 0; i < fanin; ++i) {
    const int predecessor = scaler.add_tensor(ir, "nested_join_predecessor", shape);
    append_join_tree(ir, scaler, predecessor, depth - 1, i);
    predecessors.push_back(predecessor);
  }
  scaler.set_context(branch, depth);
  scaler.add_join(ir, predecessors, output, "nested_join");
}

inline IRGraph scale_join(const IRGraph& orig, uint32_t seed) {
  IRGraph scaled = orig;
  IRScaler scaler(seed);
  constexpr size_t kAdditionalJoinDepth = 2;

  std::vector<int> retained_inputs;
  std::vector<int> expandable_inputs;
  for (int input : orig.graph_inputs) {
    if (graph_input_reaches_join(orig, input)) {
      expandable_inputs.push_back(input);
    } else {
      retained_inputs.push_back(input);
    }
  }

  if (expandable_inputs.empty()) {
    throw std::invalid_argument("Join scaling requires at least one join-feeding input.");
  }

  scaled.graph_inputs = retained_inputs;
  for (int input : expandable_inputs) {
    // The old graph-input tensor becomes the root output of a newly inserted
    // join tree. Existing consumers therefore need no rewiring.
    append_join_tree(scaled, scaler, input, kAdditionalJoinDepth);
  }
  return scaled;
}

inline void append_fork_join(IRGraph& ir, IRScaler& scaler, int input, int output,
                             size_t nested_depth, int branch = -1) {
  scaler.set_context(branch, nested_depth);
  const Shape4D shape = ir.tensors[output].shape;
  const size_t fanout = scaler.random_between(2, 3);
  const auto fork_outputs = scaler.add_split(ir, input, fanout, "nested_fj_fork");

  std::vector<int> branch_outputs;
  branch_outputs.reserve(fanout);
  const size_t recursive_branch = scaler.random_between(0, fanout - 1);
  for (size_t i = 0; i < fanout; ++i) {
    const int branch_output = scaler.add_tensor(ir, "nested_fj_branch", shape);
    if (nested_depth > 1 && i == recursive_branch) {
      append_fork_join(ir, scaler, fork_outputs[i], branch_output, nested_depth - 1, i);
    } else {
      scaler.set_context(i, nested_depth);
      auto sampled = scaler.sample_vision_op(ir.tensors[fork_outputs[i]].shape);
      if (sampled.out_shape != shape) {
         const int inter = scaler.add_tensor(ir, "nested_fj_inter", sampled.out_shape);
         scaler.add_unary(ir, fork_outputs[i], inter, sampled);
         auto proj = scaler.project_op(sampled.out_shape, shape);
         scaler.add_unary(ir, inter, branch_output, proj);
      } else {
         scaler.add_unary(ir, fork_outputs[i], branch_output, sampled);
      }
    }
    branch_outputs.push_back(branch_output);
  }
  scaler.set_context(branch, nested_depth);
  scaler.add_join(ir, branch_outputs, output, "nested_fj_join");
}

inline IRGraph scale_fork_join(const IRGraph& orig, uint32_t seed) {
  IRGraph scaled;
  scaled.tensors = orig.tensors;
  scaled.graph_inputs = orig.graph_inputs;
  scaled.graph_outputs = orig.graph_outputs;
  IRScaler scaler(seed);
  constexpr size_t kNestedForkJoinDepth = 2;

  const auto reachable_from_fork = tensors_reachable_from_forks(orig);
  const auto reaches_join = tensors_that_reach_joins(orig);
  bool replaced = false;

  for (const auto& op : orig.ops) {
    const bool unary = op.inputs.size() == 1 && op.outputs.size() == 1;
    const bool inside_fork_join =
        unary && reachable_from_fork[op.inputs[0]] && reaches_join[op.outputs[0]];

    if (inside_fork_join) {
      // Substitute a branch leaf with a complete fork-join block. Recursing
      // inside one child creates true hierarchical fork-joins rather than a
      // larger flat fork followed by one flat join.
      append_fork_join(scaled, scaler, op.inputs[0], op.outputs[0], kNestedForkJoinDepth);
      replaced = true;
    } else {
      scaled.ops.push_back(op);
    }
  }

  if (!replaced) {
    throw std::invalid_argument(
        "Fork-join scaling requires a unary branch body between a fork and a join.");
  }
  return scaled;
}

inline MotifTarget infer_target(const IRGraph& ir) {
  bool has_fork = false;
  bool has_join = false;
  for (const auto& op : ir.ops) {
    has_fork = has_fork || op.outputs.size() > 1;
    has_join = has_join || op.inputs.size() > 1;
  }

  if (has_fork && has_join) return MotifTarget::ForkJoin;
  if (has_fork) return MotifTarget::Branch;
  if (has_join) return MotifTarget::Join;

  std::vector<bool> produced(ir.tensors.size(), false);
  for (const auto& op : ir.ops) {
    for (int output : op.outputs) produced[output] = true;
  }
  for (const auto& op : ir.ops) {
    for (int input : op.inputs) {
      if (produced[input]) return MotifTarget::Linear;
    }
  }
  return MotifTarget::Rank;
}

}  // namespace vision_scale_detail

inline IRGraph scale_ir(MotifTarget target, const IRGraph& orig, uint32_t seed) {
  using namespace vision_scale_detail;
  switch (target) {
    case MotifTarget::Rank:
      return scale_rank(orig, seed);
    case MotifTarget::Linear:
      return scale_linear(orig, seed);
    case MotifTarget::Branch:
      return scale_branch(orig, seed);
    case MotifTarget::Join:
      return scale_join(orig, seed);
    case MotifTarget::ForkJoin:
      return scale_fork_join(orig, seed);
  }
  throw std::invalid_argument("Unknown motif target.");
}

// Backward-compatible overload used by the current simulation driver.
inline IRGraph scale_ir(const IRGraph& orig, uint32_t seed) {
  return scale_ir(vision_scale_detail::infer_target(orig), orig, seed);
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
extern std::vector<std::string> find_fw_ranked_execution_order(Graph& graph,
                                                               std::ostream* os = nullptr);
extern std::vector<std::string> find_fw_linear_execution_order(Graph& graph,
                                                               std::ostream* os = nullptr);
extern std::vector<std::string> find_fw_branching_execution_order(Graph& graph,
                                                                  std::ostream* os = nullptr);
extern std::vector<std::string> find_fw_joining_execution_order(Graph& graph,
                                                                std::ostream* os = nullptr);

inline std::vector<std::string> find_fw_flat_bj_execution_order(Graph& graph,
                                                                std::ostream* os = nullptr) {
  FlatBJSolver solver(graph, os);
  return solver.find_forward_order();
}

inline std::vector<std::string> find_fw_full_execution_order(Graph& graph,
                                                             std::ostream* os = nullptr) {
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
      {"FULL", find_fw_full_execution_order(g)}};

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

    IRGraph ir = build_candidate(target, seed);
    randomize_ids_and_materialization(ir, seed);

    if (ir.ops.size() > kMaximumOps) continue;

    Graph graph = materialize(ir);
    AblationPeaks peaks = evaluate_ablations(graph);

    if (!passes_cheap_separator(target, peaks)) continue;

    bool complete = false;
    auto oracle_order = find_fw_fork_join_execution_order(graph, kDiscoveryStateLimit, &complete);

    if (!complete) continue;

    peaks.oracle = evaluate_peak(graph, oracle_order);
    peaks.oracle_complete = true;

    // Strict validation
    bool valid = false;
    switch (target) {
      case MotifTarget::Rank:
        valid = peaks.rank == peaks.oracle;
        break;
      case MotifTarget::Linear:
        valid = peaks.linear == peaks.oracle;
        break;
      case MotifTarget::Branch:
        valid = peaks.branch == peaks.oracle;
        break;
      case MotifTarget::Join:
        valid = peaks.join == peaks.oracle;
        break;
      case MotifTarget::ForkJoin:
        valid = peaks.full == peaks.oracle;
        break;
    }

    if (valid) {
      return {std::move(graph), seed, peaks};
    }
  }

  throw std::runtime_error("Unable to generate motif witness within the attempt budget.");
}
