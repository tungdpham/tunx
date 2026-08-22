/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/example_graphs.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <string>

#include "device/iallocator.hpp"
#include "nn/graph.hpp"
#include "nn/graph_builder.hpp"
#include "nn/layer_factory.hpp"
#include "type/type.hpp"

namespace tunx {

namespace {

using Shape = Vec<size_t>;

void finalize_graph(Graph &graph, IAllocator &allocator, const Node &output, GraphOpts opts) {
  output->set_uid("output");
  graph.set_output(output);
  graph.compile(allocator, opts);
}

Graph create_mnist_graph(IAllocator &allocator, GraphOpts opts) {
  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 28, 28, 1};

  Node x = conv2d(input, shape, 8, 5, 1, 0, false, "conv1");
  x = batchnorm(x, shape, true, "bn1");
  x = maxpool2d(x, shape, 3, 3, 0, "pool1");
  x = conv2d(x, shape, 16, 1, 1, 0, false, "conv2_1x1");
  x = batchnorm(x, shape, true, "bn2_1x1");
  x = relu(x, shape, "relu2");
  x = conv2d(x, shape, 48, 5, 1, 0, false, "conv3");
  x = batchnorm(x, shape, true, "bn3");
  x = maxpool2d(x, shape, 2, 2, 0, "pool2");
  x = flatten(x, shape, 1, -1, "flatten");
  Node output = dense(x, shape, 10, false, "output");
  finalize_graph(graph, allocator, output, opts);
  return graph;
}

Graph create_cifar10_resnet9_graph(IAllocator &allocator, GraphOpts opts) {
  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 32, 32, 3};

  Node x = conv2d(input, shape, 64, 3, 1, 1, false, "conv1");
  x = batchnorm(x, shape, true, "bn1");
  x = conv2d(x, shape, 128, 3, 1, 1, false, "conv2");
  x = batchnorm(x, shape, true, "bn2");
  x = maxpool2d(x, shape, 2, 2, 0, "pool1");
  x = basic_residual_block(x, shape, 128, 1, "res_block1");
  x = basic_residual_block(x, shape, 128, 1, "res_block2");
  x = conv2d(x, shape, 256, 3, 1, 1, false, "conv3");
  x = batchnorm(x, shape, true, "bn3");
  x = maxpool2d(x, shape, 2, 2, 0, "pool2");
  x = basic_residual_block(x, shape, 256, 1, "res_block3");
  x = basic_residual_block(x, shape, 256, 1, "res_block4");
  x = conv2d(x, shape, 512, 3, 1, 1, false, "conv4");
  x = batchnorm(x, shape, true, "bn4");
  x = maxpool2d(x, shape, 2, 2, 0, "pool3");
  x = basic_residual_block(x, shape, 512, 1, "res_block5");
  x = avgpool2d(x, shape, 4, 1, 0, "avgpool");
  x = flatten(x, shape, 1, -1, "flatten");
  Node output = dense(x, shape, 10, true, "output");
  finalize_graph(graph, allocator, output, opts);
  return graph;
}

Graph create_cifar100_resnet18_graph(IAllocator &allocator, GraphOpts opts) {
  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 32, 32, 3};

  Node x = conv2d(input, shape, 32, 3, 1, 1, false, "conv1");
  x = batchnorm(x, shape, true, "bn1");
  x = maxpool2d(x, shape, 2, 2, 0, "maxpool");
  x = basic_residual_block(x, shape, 64, 1, "layer1_block1");
  x = basic_residual_block(x, shape, 64, 1, "layer1_block2");
  x = basic_residual_block(x, shape, 128, 2, "layer2_block1");
  x = basic_residual_block(x, shape, 128, 1, "layer2_block2");
  x = basic_residual_block(x, shape, 256, 2, "layer3_block1");
  x = basic_residual_block(x, shape, 256, 1, "layer3_block2");
  x = basic_residual_block(x, shape, 512, 2, "layer4_block1");
  x = basic_residual_block(x, shape, 512, 1, "layer4_block2");
  x = avgpool2d(x, shape, 2, 1, 0, "avgpool");
  x = flatten(x, shape, 1, -1, "flatten");
  Node output = dense(x, shape, 100, true, "fc");
  finalize_graph(graph, allocator, output, opts);
  return graph;
}

Graph create_cifar100_wrn16_8_graph(IAllocator &allocator, GraphOpts opts) {
  constexpr size_t width_factor = 8;
  constexpr float dropout_rate = 0.3f;
  constexpr size_t c1 = 16 * width_factor;
  constexpr size_t c2 = 32 * width_factor;
  constexpr size_t c3 = 64 * width_factor;

  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 32, 32, 3};

  Node x = conv2d(input, shape, 16, 3, 1, 1, true, "conv1");
  x = wide_residual_block(x, shape, c1, 1, dropout_rate, "group1_block1");
  x = wide_residual_block(x, shape, c1, 1, dropout_rate, "group1_block2");
  x = wide_residual_block(x, shape, c2, 2, dropout_rate, "group2_block1");
  x = wide_residual_block(x, shape, c2, 1, dropout_rate, "group2_block2");
  x = wide_residual_block(x, shape, c3, 2, dropout_rate, "group3_block1");
  x = wide_residual_block(x, shape, c3, 1, dropout_rate, "group3_block2");
  x = batchnorm(x, shape, true, "bn_final");
  x = avgpool2d(x, shape, 8, 1, 0, "avgpool");
  x = flatten(x, shape, 1, -1, "flatten");
  Node output = dense(x, shape, 100, true, "fc");
  finalize_graph(graph, allocator, output, opts);
  return graph;
}

Graph create_tiny_imagenet_wrn16_8_graph(IAllocator &allocator, GraphOpts opts) {
  constexpr size_t width_factor = 8;
  constexpr float dropout_rate = 0.3f;
  constexpr size_t c1 = 16 * width_factor;
  constexpr size_t c2 = 32 * width_factor;
  constexpr size_t c3 = 64 * width_factor;

  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 64, 64, 3};

  Node x = conv2d(input, shape, 16, 3, 1, 1, true, "conv1");
  x = wide_residual_block(x, shape, c1, 1, dropout_rate, "group1_block1");
  x = wide_residual_block(x, shape, c1, 1, dropout_rate, "group1_block2");
  x = wide_residual_block(x, shape, c2, 2, dropout_rate, "group2_block1");
  x = wide_residual_block(x, shape, c2, 1, dropout_rate, "group2_block2");
  x = wide_residual_block(x, shape, c3, 2, dropout_rate, "group3_block1");
  x = wide_residual_block(x, shape, c3, 1, dropout_rate, "group3_block2");
  x = batchnorm(x, shape, true, "bn_final");
  x = avgpool2d(x, shape, 8, 1, 0, "avgpool");
  x = flatten(x, shape, 1, -1, "flatten");
  Node output = dense(x, shape, 200, true, "fc");
  finalize_graph(graph, allocator, output, opts);
  return graph;
}

Graph create_imagenet100_resnet50_graph(IAllocator &allocator, GraphOpts opts) {
  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 224, 224, 3};

  Node x = conv2d(input, shape, 64, 7, 2, 3, true, "conv1");
  x = batchnorm(x, shape, true, "bn1");
  x = maxpool2d(x, shape, 3, 2, 1, "maxpool");
  x = bottleneck_residual_block(x, shape, 64, 256, 1, "layer1_block1");
  x = bottleneck_residual_block(x, shape, 64, 256, 1, "layer1_block2");
  x = bottleneck_residual_block(x, shape, 64, 256, 1, "layer1_block3");
  x = bottleneck_residual_block(x, shape, 128, 512, 2, "layer2_block1");
  x = bottleneck_residual_block(x, shape, 128, 512, 1, "layer2_block2");
  x = bottleneck_residual_block(x, shape, 128, 512, 1, "layer2_block3");
  x = bottleneck_residual_block(x, shape, 128, 512, 1, "layer2_block4");
  x = bottleneck_residual_block(x, shape, 256, 1024, 2, "layer3_block1");
  x = bottleneck_residual_block(x, shape, 256, 1024, 1, "layer3_block2");
  x = bottleneck_residual_block(x, shape, 256, 1024, 1, "layer3_block3");
  x = bottleneck_residual_block(x, shape, 256, 1024, 1, "layer3_block4");
  x = bottleneck_residual_block(x, shape, 256, 1024, 1, "layer3_block5");
  x = bottleneck_residual_block(x, shape, 256, 1024, 1, "layer3_block6");
  x = bottleneck_residual_block(x, shape, 512, 2048, 2, "layer4_block1");
  x = bottleneck_residual_block(x, shape, 512, 2048, 1, "layer4_block2");
  x = bottleneck_residual_block(x, shape, 512, 2048, 1, "layer4_block3");
  x = avgpool2d(x, shape, 7, 1, 0, "avgpool");
  x = flatten(x, shape, 1, -1, "flatten");
  Node output = dense(x, shape, 100, true, "fc");
  finalize_graph(graph, allocator, output, opts);
  return graph;
}

Graph create_gpt2_graph(IAllocator &allocator, size_t embed_dim, size_t num_heads,
                        size_t num_layers, const std::string &name, GraphOpts opts) {
  constexpr size_t seq_len = 1024;
  constexpr size_t vocab_size = 50257;
  constexpr float dropout_rate = 0.1f;

  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, seq_len};

  Node x = embedding(input, shape, vocab_size, embed_dim, "token_embed");
  x = positional_embedding(x, shape, embed_dim, seq_len, "pos_embed");
  x = dropout(x, shape, dropout_rate, "dropout");

  for (size_t i = 0; i < num_layers; ++i) {
    x = gpt_block(x, shape, embed_dim, num_heads, embed_dim * 4, dropout_rate, true,
                  name + "_block_" + std::to_string(i));
  }

  x = layernorm(x, shape, 1e-5f, true, "ln_f");
  Node output = dense(x, shape, vocab_size, true, "head");
  finalize_graph(graph, allocator, output, opts);
  return graph;
}

Graph create_inception_v1_graph(IAllocator &allocator, GraphOpts opts) {
  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 224, 224, 3};

  Node x = conv2d(input, shape, 64, 7, 2, 3, false, "conv1");
  x = batchnorm(x, shape, true, "bn1");
  x = maxpool2d(x, shape, 3, 2, 1, "pool1");

  x = inception_block(x, shape, 32, "inc1");
  x = inception_block(x, shape, 64, "inc2");
  x = maxpool2d(x, shape, 3, 2, 1, "pool2");

  x = inception_block(x, shape, 128, "inc3");
  x = inception_block(x, shape, 128, "inc4");

  x = avgpool2d(x, shape, 28, 1, 0, "avgpool");
  x = flatten(x, shape, 1, -1, "flatten");
  Node output = dense(x, shape, 100, true, "output");

  finalize_graph(graph, allocator, output, opts);
  return graph;
}

Graph create_tunx_v1_graph(IAllocator &allocator, GraphOpts opts) {
  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 224, 224, 3};

  Node x = conv2d(input, shape, 64, 7, 2, 3, false, "conv1");  // -> {112, 112, 64}
  x = batchnorm(x, shape, true, "bn1");
  x = maxpool2d(x, shape, 3, 2, 1, "pool1");  // -> {56, 56, 64}

  x = v1_residual_block(x, shape, 256, "asym1");  // -> {28, 28, 256}
  x = maxpool2d(x, shape, 2, 2, 0, "pool2");      // -> {14, 14, 256}

  x = avgpool2d(x, shape, 14, 1, 0, "avgpool");
  x = flatten(x, shape, 1, -1, "flatten");
  Node output = dense(x, shape, 100, true, "output");

  finalize_graph(graph, allocator, output, opts);
  return graph;
}

Graph create_tunx_v2_graph(IAllocator &allocator, GraphOpts opts) {
  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 224, 224, 3};

  /*
   * Construct a compact common seed:
   *
   * 224x224x3 -> 14x14x3 -> 14x14x8
   */
  Node seed = avgpool2d(input, shape, 16, 16, 0, "seed_pool");

  seed = conv2d(seed, shape, 8, 1, 1, 0, false, "seed_projection");

  const Shape seed_shape = shape;

  /*
   * Outer group 1: relatively small inner fork.
   *
   * Expanded resolutions:
   *   28x28, 56x56, 84x84
   */
  Shape g1_shape = seed_shape;
  Node g1 = v2_nested_group(seed, g1_shape, 64, 2, 4, 6, "outer_g1");

  /*
   * Outer group 2: medium inner fork.
   *
   * Expanded resolutions:
   *   56x56, 84x84, 112x112
   */
  Shape g2_shape = seed_shape;
  Node g2 = v2_nested_group(seed, g2_shape, 64, 4, 6, 8, "outer_g2");

  /*
   * Outer group 3: largest inner fork.
   *
   * Expanded resolutions:
   *   84x84, 112x112, 140x140
   */
  Shape g3_shape = seed_shape;
  Node g3 = v2_nested_group(seed, g3_shape, 64, 6, 8, 10, "outer_g3");

  /*
   * Each group output is:
   *
   * 28x28x(3 * 64) = 28x28x192
   *
   * Add is used for the outer join so the channel count remains 192.
   */
  Shape merged_shape = g1_shape;
  Node merged = add({g1, g2, g3}, merged_shape, "outer_merge");

  shape = merged_shape;

  // 28x28x192 -> 1x1x192
  Node x = avgpool2d(merged, shape, 28, 1, 0, "global_avgpool");

  x = flatten(x, shape, 1, -1, "flatten");
  Node output = dense(x, shape, 100, true, "output");

  finalize_graph(graph, allocator, output, opts);
  return graph;
}

Graph create_tunx_v3_graph(IAllocator &allocator, GraphOpts opts) {
  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 224, 224, 3};

  Node x = avgpool2d(input, shape, 16, 16, 0, "pool1");  // -> {14, 14, 3}
  x = conv2d(x, shape, 8, 1, 1, 0, false, "conv1");      // -> {14, 14, 8}

  const Shape initial_shape = shape;

  Shape b1_shape = initial_shape;
  Node b1 = convtranspose2d(x, b1_shape, 32, 2, 2, 0, false, "b1_up1");  // -> {28, 28, 32}
  b1 = convtranspose2d(b1, b1_shape, 64, 2, 2, 0, false, "b1_up2");      // -> {56, 56, 64}
  b1 = conv2d(b1, b1_shape, 128, 3, 1, 1, false, "b1_conv");             // -> {112, 112, 56}
  b1 = transpose(b1, b1_shape, 1, 2, "b1_transpose");                    // -> {56, 56, 128}
  b1 = conv2d(b1, b1_shape, 64, 3, 1, 1, false, "b1_down");              // -> {56, 56, 64}
  b1 = avgpool2d(b1, b1_shape, 2, 2, 0, "b1_pool");                      // -> {28, 28, 64}

  Shape b2_shape = initial_shape;
  Node b2 = convtranspose2d(x, b2_shape, 64, 2, 2, 0, false, "b2_trans");  // -> {28, 28, 64}
  b2 = convtranspose2d(b2, b2_shape, 64, 2, 2, 0, false, "b2_up2");        // -> {56, 56, 64}
  b2 = conv2d(b2, b2_shape, 64, 3, 1, 1, false, "b2_conv");                // -> {56, 56, 64}
  b2 = conv2d(b2, b2_shape, 64, 3, 1, 1, false, "b2_conv2");               // -> {56, 56, 64}
  b2 = maxpool2d(b2, b2_shape, 2, 2, 0, "b2_pool");                        // -> {28, 28, 64}

  Shape b3_shape = initial_shape;
  Node b3 = convtranspose2d(x, b3_shape, 32, 2, 2, 0, false, "b3_up1");  // -> {28,28,16}
  b3 = convtranspose2d(b3, b3_shape, 128, 4, 4, 0, false, "b3_up2");     // -> {112,112,128}
  b3 = transpose(b3, b3_shape, 1, 2, "b3_transpose");                    // -> {112,112,128}
  b3 = b3 * -1;                                                          // -> {112,112,128}
  b3 = maxpool2d(b3, b3_shape, 4, 4, 0, "b3_pool");                      // -> {28, 28, 128}
  b3 = conv2d(b3, b3_shape, 64, 3, 1, 1, false, "b3_down1");             // -> {28, 28, 64}

  Shape y_shape = b1_shape;
  Node y = add({b1, b2, b3}, y_shape, "merge_b1_b2_b3");
  y = conv2d(y, y_shape, 8, 1, 1, 0, false, "out_conv");
  y = relu(y, y_shape, "out_relu");

  Node output = dense(y, y_shape, 100, true, "output");

  finalize_graph(graph, allocator, output, opts);
  return graph;
}
}  // namespace

std::unordered_map<std::string, std::function<Graph(IAllocator &, GraphOpts)>>
    ExampleGraphs::creators_;

void ExampleGraphs::register_defaults() {
  register_graph("mnist_cnn", create_mnist_graph);

  register_graph("inception_v1", create_inception_v1_graph);

  register_graph("cifar10_resnet9", create_cifar10_resnet9_graph);

  register_graph("cifar100_resnet18", create_cifar100_resnet18_graph);
  register_graph("cifar100_wrn16_8", create_cifar100_wrn16_8_graph);

  register_graph("tiny_imagenet_wrn16_8", create_tiny_imagenet_wrn16_8_graph);

  register_graph("imagenet100_resnet50", create_imagenet100_resnet50_graph);

  register_graph("gpt2_small", [](IAllocator &allocator, GraphOpts opts) {
    return create_gpt2_graph(allocator, 768, 12, 12, "gpt2_small", opts);
  });
  register_graph("gpt2_medium", [](IAllocator &allocator, GraphOpts opts) {
    return create_gpt2_graph(allocator, 1024, 16, 24, "gpt2_medium", opts);
  });
  register_graph("gpt2_large", [](IAllocator &allocator, GraphOpts opts) {
    return create_gpt2_graph(allocator, 1280, 20, 36, "gpt2_large", opts);
  });

  // graphs for benchmarking
  register_graph("tunx_v1", create_tunx_v1_graph);
  register_graph("tunx_v2", create_tunx_v2_graph);
  register_graph("tunx_v3", create_tunx_v3_graph);
}
}  // namespace tunx