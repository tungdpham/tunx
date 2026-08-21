#pragma once

#include <cstddef>

#include "graph.h"

inline size_t random_act_size() {
  int rand_val = rand() % 100;
  if (rand_val < 30) return 1280;
  if (rand_val < 60) return 2560;
  if (rand_val < 80) return 5120;
  return 10240;
}

inline size_t random_res_size() {
  int rand_val = rand() % 100;
  if (rand_val < 30) return 512;
  if (rand_val < 60) return 1024;
  if (rand_val < 80) return 2048;
  return 4096;
}

inline size_t random_ws_size() {
  int rand_val = rand() % 100;
  if (rand_val < 30) return 256;
  if (rand_val < 60) return 512;
  if (rand_val < 80) return 1024;
  return 2048;
}

inline ActivationNode* build_random_fork_join_dag(Graph& g, ActivationNode* input, int depth,
                                                  int& node_counter) {
  if (depth == 0) {
    return input;
  }

  std::string prefix = "d" + std::to_string(depth) + "_" + std::to_string(node_counter++);

  int structure_type = rand() % 3;

  if (structure_type == 0) {
    // Sequential
    auto act = g.add_act(prefix + "_seq", random_act_size());
    g.add_op(prefix + "_seq_conv", random_ws_size(), {input}, {act}, {}, random_res_size());
    return build_random_fork_join_dag(g, act, depth - 1, node_counter);
  } else if (structure_type == 1) {
    // Fork_join
    auto left_act = g.add_act(prefix + "_left", random_act_size());
    g.add_op(prefix + "_left_conv", random_ws_size(), {input}, {left_act}, {}, random_res_size());
    auto left_out = build_random_fork_join_dag(g, left_act, depth - 1, node_counter);

    auto right_act = g.add_act(prefix + "_right", random_act_size());
    g.add_op(prefix + "_right_conv", random_ws_size(), {input}, {right_act}, {}, random_res_size());
    auto right_out = build_random_fork_join_dag(g, right_act, depth - 1, node_counter);

    auto merge_act = g.add_act(prefix + "_merge", random_act_size());
    g.add_op(prefix + "_add", random_ws_size(), {left_out, right_out}, {merge_act}, {},
             random_res_size());

    return merge_act;
  } else {
    // Residual (skip connection)
    auto main_act = g.add_act(prefix + "_main", random_act_size());
    g.add_op(prefix + "_main_conv", random_ws_size(), {input}, {main_act}, {}, random_res_size());
    auto main_out = build_random_fork_join_dag(g, main_act, depth - 1, node_counter);

    auto merge_act = g.add_act(prefix + "_res_add", random_act_size());
    g.add_op(prefix + "_res_add", random_ws_size(), {main_out, input}, {merge_act}, {},
             random_res_size());

    return merge_act;
  }
}

inline Graph random_fork_join_graph(int depth) {
  Graph g;
  int node_counter = 0;
  auto input = g.add_act("input", 10000);
  g.set_inputs({input});
  auto output = build_random_fork_join_dag(g, input, depth, node_counter);
  g.set_outputs({output});
  return g;
}

inline ActivationNode* build_random_order_invariant_fork_join_dag(Graph& g, ActivationNode* input,
                                                                  int depth, int& node_counter) {
  if (depth == 0) {
    return input;
  }

  std::string prefix = "sd" + std::to_string(depth) + "_" + std::to_string(node_counter++);
  int structure_type = rand() % 3;

  if (structure_type == 0) {
    auto act = g.add_act(prefix + "_seq", random_act_size());
    g.add_op(prefix + "_seq_conv", random_ws_size(), {input}, {act}, {}, random_res_size());
    return build_random_order_invariant_fork_join_dag(g, act, depth - 1, node_counter);
  }

  if (structure_type == 1) {
    auto left_input = g.add_act(prefix + "_left_input", random_act_size());
    auto right_input = g.add_act(prefix + "_right_input", random_act_size());
    g.add_op(prefix + "_split", random_ws_size(), {input}, {left_input, right_input}, {},
             random_res_size());

    auto left_out =
        build_random_order_invariant_fork_join_dag(g, left_input, depth - 1, node_counter);
    auto right_out =
        build_random_order_invariant_fork_join_dag(g, right_input, depth - 1, node_counter);

    auto merge_act = g.add_act(prefix + "_merge", random_act_size());
    g.add_op(prefix + "_add", random_ws_size(), {left_out, right_out}, {merge_act}, {},
             random_res_size());
    return merge_act;
  }

  auto main_input = g.add_act(prefix + "_main_input", random_act_size());
  auto skip_input = g.add_act(prefix + "_skip_input", random_act_size());
  g.add_op(prefix + "_split", random_ws_size(), {input}, {main_input, skip_input}, {},
           random_res_size());

  auto main_out =
      build_random_order_invariant_fork_join_dag(g, main_input, depth - 1, node_counter);
  auto merge_act = g.add_act(prefix + "_res_merge", random_act_size());
  g.add_op(prefix + "_res_add", random_ws_size(), {main_out, skip_input}, {merge_act}, {},
           random_res_size());
  return merge_act;
}

inline Graph random_order_invariant_fork_join_graph(int depth) {
  Graph g;
  int node_counter = 0;
  auto input = g.add_act("input", random_act_size());
  g.set_inputs({input});
  auto output = build_random_order_invariant_fork_join_dag(g, input, depth, node_counter);
  g.set_outputs({output});
  return g;
}

inline Graph random_m_sequences_graph(int m, int length) {
  Graph g;
  int node_counter = 0;
  for (int i = 0; i < m; ++i) {
    auto input = g.add_act("input_" + std::to_string(i), random_act_size());
    auto current_inputs = g.inputs();
    current_inputs.push_back(input);
    g.set_inputs(current_inputs);

    ActivationNode* curr = input;
    for (int j = 0; j < length; ++j) {
      std::string prefix = "seq_" + std::to_string(i) + "_" + std::to_string(j) + "_" +
                           std::to_string(node_counter++);
      auto next_act = g.add_act(prefix, random_act_size());
      g.add_op(prefix + "_op", random_ws_size(), {curr}, {next_act}, {}, random_res_size());
      curr = next_act;
    }
    auto current_outputs = g.outputs();
    current_outputs.push_back(curr);
    g.set_outputs(current_outputs);
  }
  return g;
}

inline void build_random_branching_dag(Graph& g, ActivationNode* input, int depth,
                                       int& node_counter) {
  if (depth == 0) {
    auto outputs = g.outputs();
    outputs.push_back(input);
    g.set_outputs(outputs);
    return;
  }

  std::string prefix = "b" + std::to_string(depth) + "_" + std::to_string(node_counter++);
  bool is_seq = (rand() % 3 > 0);

  if (is_seq) {
    auto next_act = g.add_act(prefix + "_seq", random_act_size());
    g.add_op(prefix + "_seq_conv", random_ws_size(), {input}, {next_act}, {}, random_res_size());
    build_random_branching_dag(g, next_act, depth - 1, node_counter);
  } else {
    int num_branches = rand() % 2 + 2;
    for (int i = 0; i < num_branches; ++i) {
      auto next_act = g.add_act(prefix + "_branch" + std::to_string(i), random_act_size());
      g.add_op(prefix + "_conv" + std::to_string(i), random_ws_size(), {input}, {next_act}, {},
               random_res_size());
      build_random_branching_dag(g, next_act, depth - 1, node_counter);
    }
  }
}

inline Graph random_branching_graph(int depth) {
  Graph g;
  int node_counter = 0;
  auto input = g.add_act("input", random_act_size());
  g.set_inputs({input});
  build_random_branching_dag(g, input, depth, node_counter);
  return g;
}

inline void build_random_order_invariant_branching_dag(Graph& g, ActivationNode* input, int depth,
                                                       int& node_counter) {
  if (depth == 0) {
    auto outputs = g.outputs();
    outputs.push_back(input);
    g.set_outputs(outputs);
    return;
  }

  std::string prefix = "sb" + std::to_string(depth) + "_" + std::to_string(node_counter++);
  if (rand() % 3 > 0) {
    auto next_act = g.add_act(prefix + "_seq", random_act_size());
    g.add_op(prefix + "_seq_conv", random_ws_size(), {input}, {next_act}, {}, random_res_size());
    build_random_order_invariant_branching_dag(g, next_act, depth - 1, node_counter);
    return;
  }

  const int num_branches = rand() % 2 + 2;
  std::vector<ActivationNode*> branch_inputs;
  branch_inputs.reserve(num_branches);
  for (int i = 0; i < num_branches; ++i) {
    branch_inputs.push_back(
        g.add_act(prefix + "_branch_input" + std::to_string(i), random_act_size()));
  }
  g.add_op(prefix + "_split", random_ws_size(), {input}, branch_inputs);

  for (auto* branch_input : branch_inputs) {
    build_random_order_invariant_branching_dag(g, branch_input, depth - 1, node_counter);
  }
}

inline Graph random_order_invariant_branching_graph(int depth) {
  Graph g;
  int node_counter = 0;
  auto input = g.add_act("input", random_act_size());
  g.set_inputs({input});
  build_random_order_invariant_branching_dag(g, input, depth, node_counter);
  return g;
}

inline ActivationNode* build_random_joining_dag(Graph& g, int depth, int& node_counter) {
  if (depth == 0) {
    std::string prefix = "j" + std::to_string(depth) + "_" + std::to_string(node_counter++);
    auto input = g.add_act(prefix + "_input", random_act_size());
    auto current_inputs = g.inputs();
    current_inputs.push_back(input);
    g.set_inputs(current_inputs);
    return input;
  }

  std::string prefix = "j" + std::to_string(depth) + "_" + std::to_string(node_counter++);
  bool is_seq = (rand() % 3 > 0);

  if (is_seq) {
    auto input = build_random_joining_dag(g, depth - 1, node_counter);
    auto output = g.add_act(prefix + "_seq_out", random_act_size());
    g.add_op(prefix + "_seq_conv", random_ws_size(), {input}, {output}, {}, random_res_size());
    return output;
  } else {
    int num_joining = rand() % 2 + 2;
    std::vector<ActivationNode*> inputs;
    for (int i = 0; i < num_joining; ++i) {
      inputs.push_back(build_random_joining_dag(g, depth - 1, node_counter));
    }
    auto output = g.add_act(prefix + "_merge", random_act_size());
    g.add_op(prefix + "_add", random_ws_size(), inputs, {output});
    return output;
  }
}

inline Graph random_joining_graph(int depth) {
  Graph g;
  int node_counter = 0;
  auto output = build_random_joining_dag(g, depth, node_counter);
  g.set_outputs({output});
  return g;
}

inline Graph sample_branch_graph() {
  Graph g;
  ActivationNode* input = g.add_act("input", 10240);

  size_t branch_length = 5;
  ActivationNode* prev = input;
  ActivationNode* b1_tail = nullptr;
  // branch 1
  for (size_t i = 0; i < branch_length; i++) {
    b1_tail = g.add_act("b1_act" + std::to_string(i), random_act_size());
    g.add_op("b1_op" + std::to_string(i), random_ws_size(), {prev}, {b1_tail}, {},
             random_res_size());
    prev = b1_tail;
  }

  prev = input;
  ActivationNode* b2_tail = nullptr;
  // branch 2
  for (size_t i = 0; i < branch_length; i++) {
    b2_tail = g.add_act("b2_act" + std::to_string(i), random_act_size());
    g.add_op("b2_op" + std::to_string(i), random_ws_size(), {prev}, {b2_tail}, {},
             random_res_size());
    prev = b2_tail;
  }

  prev = input;
  ActivationNode* b3_tail = nullptr;
  // branch 3
  for (size_t i = 0; i < branch_length; i++) {
    b3_tail = g.add_act("b3_act" + std::to_string(i), random_act_size());
    g.add_op("b3_op" + std::to_string(i), random_ws_size(), {prev}, {b3_tail}, {},
             random_res_size());
    prev = b3_tail;
  }

  // join
  ActivationNode* output = g.add_act("output", 10240);
  g.add_op("join_op", random_ws_size(), {b1_tail, b2_tail, b3_tail}, {output}, {},
           random_res_size());

  g.set_inputs({input});
  g.set_outputs({output});

  return g;
}

// Tunx V1 graph with exact topology and profiled memory values.
// Architecture: conv1 -> bn1 -> pool1 -> asymmetric_block(256ch) -> pool2 -> avgpool -> flatten ->
// output Batch=64, BF16, ImageNet-100 (224x224x3).
// Activation sizes, workspace, and residual_mem derived from real edge profile traces.
inline Graph tunx_v1_graph() {
  Graph g;

  // --- Activation sizes (BF16 = 2 bytes per element, batch=64) ---
  constexpr size_t input_size = 19267584;     // 64 * 224 * 224 * 3 * 2
  constexpr size_t stem_112 = 102760448;      // 64 * 112 * 112 * 64 * 2
  constexpr size_t stem_56 = 25690112;        // 64 * 56 * 56 * 64 * 2
  constexpr size_t asym_256 = 102760448;      // 64 * 56 * 56 * 256 * 2
  constexpr size_t asym_512 = 205520896;      // 64 * 56 * 56 * 512 * 2
  constexpr size_t asym_2048 = 822083584;     // 64 * 56 * 56 * 2048 * 2
  constexpr size_t concat_size = 1130364928;  // 64 * 56 * 56 * 2816 * 2
  constexpr size_t pool2_size = 282591232;    // 64 * 28 * 28 * 2816 * 2
  constexpr size_t global_avg = 360448;       // 64 * 1 * 1 * 2816 * 2
  constexpr size_t output_size = 12800;       // 64 * 100 * 2

  // --- Activations ---
  auto* a_input = g.add_act("input", input_size);

  // Stem
  auto* a_conv1 = g.add_act("conv1_out", stem_112);
  auto* a_bn1 = g.add_act("bn1_out", stem_112);
  auto* a_pool1 = g.add_act("pool1_out", stem_56);

  // Branch 1: 1x1 conv -> fork(3x3 main, 3x3 shortcut) -> add
  auto* a_b1c1 = g.add_act("b1_conv_1_out", asym_512);
  auto* a_b1c2m = g.add_act("b1_conv_2_main_out", asym_2048);
  auto* a_b1c2s = g.add_act("b1_conv2_shortcut_out", asym_2048);
  auto* a_add = g.add_act("add_out", asym_2048);

  // Branch 2: conv1->bn1->conv1->bn1
  auto* a_b2c1a = g.add_act("b2_conv1_a_out", asym_256);
  auto* a_b2bn1a = g.add_act("b2_bn1_a_out", asym_256);
  auto* a_b2c1b = g.add_act("b2_conv1_b_out", asym_256);
  auto* a_b2bn1b = g.add_act("b2_bn1_b_out", asym_256);

  // Branch 3: conv1->bn1->conv2->bn2
  auto* a_b3c1 = g.add_act("b3_conv1_out", asym_256);
  auto* a_b3bn1 = g.add_act("b3_bn1_out", asym_256);
  auto* a_b3c2 = g.add_act("b3_conv2_out", asym_256);
  auto* a_b3bn2 = g.add_act("b3_bn2_out", asym_256);

  // Branch 4: maxpool->conv->bn
  auto* a_b4pool = g.add_act("b4_pool_out", stem_56);
  auto* a_b4conv = g.add_act("b4_conv_out", asym_256);
  auto* a_b4bn = g.add_act("b4_bn_out", asym_256);

  // Tail: concat -> relu -> pool2 -> avgpool -> flatten -> output
  auto* a_concat = g.add_act("concat_out", concat_size);
  auto* a_relu = g.add_act("relu_out", concat_size);
  auto* a_pool2 = g.add_act("pool2_out", pool2_size);
  auto* a_avgpool = g.add_act("avgpool_out", global_avg);
  auto* a_flatten = g.add_act("flatten_out", global_avg);
  auto* a_output = g.add_act("output_out", output_size);

  // --- Operations (name, workspace, inputs, outputs, cache, residual_mem) ---
  // Stem
  g.add_op("conv1", 51430656, {a_input}, {a_conv1}, {a_input}, 19267584);
  g.add_op("bn1", 51393792, {a_conv1}, {a_bn1}, {a_conv1}, 154141184);
  g.add_op("pool1", 51380224, {a_bn1}, {a_pool1}, {}, 51380224);

  // Branch 1
  g.add_op("asym1_b1_conv_1", 0, {a_pool1}, {a_b1c1}, {a_pool1});
  g.add_op("asym1_b1_conv_2_main", 0, {a_b1c1}, {a_b1c2m}, {a_b1c1});
  g.add_op("asym1_b1_conv2_shortcut", 0, {a_b1c1}, {a_b1c2s}, {a_b1c1});
  g.add_op("add", 0, {a_b1c2m, a_b1c2s}, {a_add});

  // Branch 2
  g.add_op("asym1_b2_conv1", 0, {a_pool1}, {a_b2c1a}, {a_pool1});
  g.add_op("asym1_b2_bn1", 51433984, {a_b2c1a}, {a_b2bn1a}, {a_b2c1a}, 154142720);
  g.add_op("asym1_b2_conv1_b", 0, {a_b2bn1a}, {a_b2c1b}, {a_b2bn1a}, 102760448);
  g.add_op("asym1_b2_bn1_b", 51433984, {a_b2c1b}, {a_b2bn1b}, {a_b2c1b}, 154142720);

  // Branch 3
  g.add_op("asym1_b3_conv1", 0, {a_pool1}, {a_b3c1}, {a_pool1});
  g.add_op("asym1_b3_bn1", 51433984, {a_b3c1}, {a_b3bn1}, {a_b3c1}, 154142720);
  g.add_op("asym1_b3_conv2", 0, {a_b3bn1}, {a_b3c2}, {a_b3bn1}, 102760448);
  g.add_op("asym1_b3_bn2", 51433984, {a_b3c2}, {a_b3bn2}, {a_b3c2}, 154142720);

  // Branch 4
  g.add_op("asym1_b4_pool", 51380224, {a_pool1}, {a_b4pool}, {}, 51380224);
  g.add_op("asym1_b4_conv", 0, {a_b4pool}, {a_b4conv}, {a_b4pool}, 25690112);
  g.add_op("asym1_b4_bn", 51433984, {a_b4conv}, {a_b4bn}, {a_b4conv}, 154142720);

  // Tail
  g.add_op("asym1_concat", 256, {a_add, a_b2bn1b, a_b3bn2, a_b4bn}, {a_concat});
  g.add_op("asym1_relu", 0, {a_concat}, {a_relu});
  g.add_op("pool2", 565182464, {a_relu}, {a_pool2}, {}, 1695547392);
  g.add_op("avgpool", 0, {a_pool2}, {a_avgpool});
  g.add_op("flatten", 0, {a_avgpool}, {a_flatten});
  g.add_op("output", 66048, {a_flatten}, {a_output}, {a_flatten}, 360448);

  g.set_inputs({a_input});
  g.set_outputs({a_output});

  return g;
}

inline Graph sample_failure_graph() {
  Graph g;

  auto* input = g.add_act("input", 1280);
  auto* b3_0_seq = g.add_act("b3_0_seq", 1280);
  auto* b2_1_branch0 = g.add_act("b2_1_branch0", 5120);
  auto* b2_1_branch1 = g.add_act("b2_1_branch1", 2560);
  auto* b1_2_branch0 = g.add_act("b1_2_branch0", 5120);
  auto* b1_2_branch1 = g.add_act("b1_2_branch1", 2560);
  auto* b1_2_branch2 = g.add_act("b1_2_branch2", 2560);
  auto* b1_3_seq = g.add_act("b1_3_seq", 1280);

  g.add_op("b3_0_seq_conv", 2048, {input}, {b3_0_seq}, {}, 1024);
  
  g.add_op("b2_1_conv0", 256, {b3_0_seq}, {b2_1_branch0}, {}, 4096);
  g.add_op("b2_1_conv1", 2048, {b3_0_seq}, {b2_1_branch1}, {}, 1024);

  g.add_op("b1_2_conv0", 512, {b2_1_branch0}, {b1_2_branch0}, {}, 4096);
  g.add_op("b1_2_conv1", 1024, {b2_1_branch0}, {b1_2_branch1}, {}, 1024);
  g.add_op("b1_2_conv2", 512, {b2_1_branch0}, {b1_2_branch2}, {}, 1024);

  g.add_op("b1_3_seq_conv", 512, {b2_1_branch1}, {b1_3_seq}, {}, 512);

  g.set_inputs({input});
  g.set_outputs({b1_2_branch0, b1_2_branch1, b1_2_branch2, b1_3_seq});

  return g;
}
inline Graph sample_failure_graph_2() {
  Graph g;

  auto* input = g.add_act("input", 5120);
  auto* b3_0_seq = g.add_act("b3_0_seq", 5120);
  auto* b2_1_branch0 = g.add_act("b2_1_branch0", 1280);
  auto* b2_1_branch1 = g.add_act("b2_1_branch1", 1280);
  auto* b2_1_branch2 = g.add_act("b2_1_branch2", 10240);
  auto* b1_2_seq = g.add_act("b1_2_seq", 5120);
  auto* b1_3_seq = g.add_act("b1_3_seq", 5120);
  auto* b1_4_seq = g.add_act("b1_4_seq", 5120);

  g.add_op("b3_0_seq_conv", 2048, {input}, {b3_0_seq}, {}, 1024);
  
  g.add_op("b2_1_conv0", 256, {b3_0_seq}, {b2_1_branch0}, {}, 512);
  g.add_op("b2_1_conv1", 1024, {b3_0_seq}, {b2_1_branch1}, {}, 2048);
  g.add_op("b2_1_conv2", 512, {b3_0_seq}, {b2_1_branch2}, {}, 512);

  g.add_op("b1_2_seq_conv", 1024, {b2_1_branch0}, {b1_2_seq}, {}, 512);
  g.add_op("b1_3_seq_conv", 2048, {b2_1_branch1}, {b1_3_seq}, {}, 1024);
  g.add_op("b1_4_seq_conv", 256, {b2_1_branch2}, {b1_4_seq}, {}, 1024);

  g.set_inputs({input});
  g.set_outputs({b1_2_seq, b1_3_seq, b1_4_seq});

  return g;
}

