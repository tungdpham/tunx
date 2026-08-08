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

inline size_t random_ws_size() {
  int rand_val = rand() % 100;
  if (rand_val < 30) return 512;
  if (rand_val < 60) return 1024;
  if (rand_val < 80) return 2048;
  return 4096;
}

inline ActivationNode* build_random_diamond_dag(Graph& g, ActivationNode* input, int depth,
                                                int& node_counter) {
  if (depth == 0) {
    return input;
  }

  std::string prefix = "d" + std::to_string(depth) + "_" + std::to_string(node_counter++);

  int structure_type = rand() % 3;

  if (structure_type == 0) {
    // Sequential
    auto act = g.add_act(prefix + "_seq", random_act_size());
    g.add_op(prefix + "_seq_conv", random_ws_size(), {input}, {act});
    return build_random_diamond_dag(g, act, depth - 1, node_counter);
  } else if (structure_type == 1) {
    // Diamond
    auto left_act = g.add_act(prefix + "_left", random_act_size());
    g.add_op(prefix + "_left_conv", random_ws_size(), {input}, {left_act});
    auto left_out = build_random_diamond_dag(g, left_act, depth - 1, node_counter);

    auto right_act = g.add_act(prefix + "_right", random_act_size());
    g.add_op(prefix + "_right_conv", random_ws_size(), {input}, {right_act});
    auto right_out = build_random_diamond_dag(g, right_act, depth - 1, node_counter);

    auto merge_act = g.add_act(prefix + "_merge", random_act_size());
    g.add_op(prefix + "_add", random_ws_size(), {left_out, right_out}, {merge_act});

    return merge_act;
  } else {
    // Residual (skip connection)
    auto main_act = g.add_act(prefix + "_main", random_act_size());
    g.add_op(prefix + "_main_conv", random_ws_size(), {input}, {main_act});
    auto main_out = build_random_diamond_dag(g, main_act, depth - 1, node_counter);

    auto merge_act = g.add_act(prefix + "_res_add", random_act_size());
    g.add_op(prefix + "_res_add", random_ws_size(), {main_out, input}, {merge_act});

    return merge_act;
  }
}

inline Graph random_diamond_graph(int depth) {
  Graph g;
  int node_counter = 0;
  auto input = g.add_act("input", 10000);
  g.set_inputs({input});
  auto output = build_random_diamond_dag(g, input, depth, node_counter);
  g.set_outputs({output});
  return g;
}

inline ActivationNode* build_random_static_diamond_dag(Graph& g, ActivationNode* input, int depth,
                                                       int& node_counter) {
  if (depth == 0) {
    return input;
  }

  std::string prefix = "sd" + std::to_string(depth) + "_" + std::to_string(node_counter++);
  int structure_type = rand() % 3;

  if (structure_type == 0) {
    auto act = g.add_act(prefix + "_seq", random_act_size());
    g.add_op(prefix + "_seq_conv", random_ws_size(), {input}, {act});
    return build_random_static_diamond_dag(g, act, depth - 1, node_counter);
  }

  if (structure_type == 1) {
    auto left_input = g.add_act(prefix + "_left_input", random_act_size());
    auto right_input = g.add_act(prefix + "_right_input", random_act_size());
    g.add_op(prefix + "_split", random_ws_size(), {input}, {left_input, right_input});

    auto left_out = build_random_static_diamond_dag(g, left_input, depth - 1, node_counter);
    auto right_out = build_random_static_diamond_dag(g, right_input, depth - 1, node_counter);

    auto merge_act = g.add_act(prefix + "_merge", random_act_size());
    g.add_op(prefix + "_add", random_ws_size(), {left_out, right_out}, {merge_act});
    return merge_act;
  }

  auto main_input = g.add_act(prefix + "_main_input", random_act_size());
  auto skip_input = g.add_act(prefix + "_skip_input", random_act_size());
  g.add_op(prefix + "_split", random_ws_size(), {input}, {main_input, skip_input});

  auto main_out = build_random_static_diamond_dag(g, main_input, depth - 1, node_counter);
  auto merge_act = g.add_act(prefix + "_res_merge", random_act_size());
  g.add_op(prefix + "_res_add", random_ws_size(), {main_out, skip_input}, {merge_act});
  return merge_act;
}

inline Graph random_static_diamond_graph(int depth) {
  Graph g;
  int node_counter = 0;
  auto input = g.add_act("input", random_act_size());
  g.set_inputs({input});
  auto output = build_random_static_diamond_dag(g, input, depth, node_counter);
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
      g.add_op(prefix + "_op", random_ws_size(), {curr}, {next_act});
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
    g.add_op(prefix + "_seq_conv", random_ws_size(), {input}, {next_act});
    build_random_branching_dag(g, next_act, depth - 1, node_counter);
  } else {
    int num_branches = rand() % 2 + 2;
    for (int i = 0; i < num_branches; ++i) {
      auto next_act = g.add_act(prefix + "_branch" + std::to_string(i), random_act_size());
      g.add_op(prefix + "_conv" + std::to_string(i), random_ws_size(), {input}, {next_act});
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

inline void build_random_static_branching_dag(Graph& g, ActivationNode* input, int depth,
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
    g.add_op(prefix + "_seq_conv", random_ws_size(), {input}, {next_act});
    build_random_static_branching_dag(g, next_act, depth - 1, node_counter);
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
    build_random_static_branching_dag(g, branch_input, depth - 1, node_counter);
  }
}

inline Graph random_static_branching_graph(int depth) {
  Graph g;
  int node_counter = 0;
  auto input = g.add_act("input", random_act_size());
  g.set_inputs({input});
  build_random_static_branching_dag(g, input, depth, node_counter);
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
    g.add_op(prefix + "_seq_conv", random_ws_size(), {input}, {output});
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
    g.add_op("b1_op" + std::to_string(i), random_ws_size(), {prev}, {b1_tail});
    prev = b1_tail;
  }

  prev = input;
  ActivationNode* b2_tail = nullptr;
  // branch 2
  for (size_t i = 0; i < branch_length; i++) {
    b2_tail = g.add_act("b2_act" + std::to_string(i), random_act_size());
    g.add_op("b2_op" + std::to_string(i), random_ws_size(), {prev}, {b2_tail});
    prev = b2_tail;
  }

  prev = input;
  ActivationNode* b3_tail = nullptr;
  // branch 3
  for (size_t i = 0; i < branch_length; i++) {
    b3_tail = g.add_act("b3_act" + std::to_string(i), random_act_size());
    g.add_op("b3_op" + std::to_string(i), random_ws_size(), {prev}, {b3_tail});
    prev = b3_tail;
  }

  // join
  ActivationNode* output = g.add_act("output", 10240);
  g.add_op("join_op", random_ws_size(), {b1_tail, b2_tail, b3_tail}, {output});

  g.set_inputs({input});
  g.set_outputs({output});

  return g;
}
