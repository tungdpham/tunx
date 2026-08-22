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

  Node x = conv2d(input, shape, 16, 7, 2, 3, false, "conv1");  // -> {112, 112, 16}
  x = batchnorm(x, shape, true, "bn1");                        // -> {112, 112, 16}
  x = maxpool2d(x, shape, 4, 4, 0, "pool1");                   // -> {28, 28, 16}

  const Shape initial_shape = shape;

  Shape b1_shape = initial_shape;
  Node b1 = convtranspose2d(x, b1_shape, 32, 2, 2, 0, false, "b1_up1");  // -> {56, 56, 32}
  b1 = batchnorm(b1, b1_shape, true, "b1_bn1");                          // -> {56, 56, 32}
  b1 = convtranspose2d(b1, b1_shape, 64, 4, 4, 0, false, "b1_up2");      // -> {224, 224, 64}
  b1 = batchnorm(b1, b1_shape, true, "b1_bn2");                          // -> {224, 224, 64}
  b1 = conv2d(b1, b1_shape, 64, 3, 1, 1, false, "b1_down");              // -> {224, 224, 64}
  b1 = maxpool2d(b1, b1_shape, 4, 4, 0, "b1_pool");                      // -> {56, 56, 64}

  Shape b2_shape = initial_shape;
  Node b2 = convtranspose2d(x, b2_shape, 64, 2, 2, 0, false, "b2_trans");  // -> {56, 56, 64}
  b2 = batchnorm(b2, b2_shape, true, "b2_bn1");                            // -> {56, 56, 64}
  b2 = convtranspose2d(b2, b2_shape, 64, 2, 2, 0, false, "b2_up2");        // -> {112, 112, 64}
  b2 = batchnorm(b2, b2_shape, true, "b2_bn2");                            // -> {112, 112, 64}
  b2 = conv2d(b2, b2_shape, 64, 3, 1, 1, false, "b2_conv");                // -> {112, 112, 64}
  b2 = batchnorm(b2, b2_shape, true, "b2_bn2");                            // -> {112, 112, 64}
  b2 = conv2d(b2, b2_shape, 64, 3, 1, 1, false, "b2_conv2");               // -> {112, 112, 64}
  b2 = batchnorm(b2, b2_shape, true, "b2_bn2");                            // -> {112, 112, 64}
  b2 = b2 * -1;                                                            // -> {112, 112, 64}
  b2 = maxpool2d(b2, b2_shape, 2, 2, 0, "b2_pool");                        // -> {56, 56, 64}

  Shape b3_shape = initial_shape;
  Node b3 = convtranspose2d(x, b3_shape, 32, 2, 2, 0, false, "b3_up1");  // -> {56, 56, 32}
  b3 = convtranspose2d(b3, b3_shape, 128, 4, 4, 0, false, "b3_up2");     // -> {224, 224, 128}
  b3 = avgpool2d(b3, b3_shape, 4, 4, 0, "b3_pool");                      // -> {56, 56, 128}
  b3 = conv2d(b3, b3_shape, 64, 3, 1, 1, false, "b3_down1");             // -> {56, 56, 64}
  b3 = batchnorm(b3, b3_shape, true, "b3_bn2");                          // -> {56, 56, 64}

  Shape y_shape = b1_shape;
  Node y = add({b1, b2, b3}, y_shape, "merge_b1_b2_b3");
  y = relu(y, y_shape, "out_relu");
  y = flatten(y, y_shape, 1, -1, "flatten");

  Node output = dense(y, y_shape, 100, true, "output");

  finalize_graph(graph, allocator, output, opts);
  return graph;
}

std::pair<Node, Shape> v2_block(Node x, Shape shape, const std::string &name) {
  Shape b1_shape = shape;
  Node b1 = convtranspose2d(x, b1_shape, 256, 2, 2, 0, false, name + "_up1");  // -> {224, 224, 128}
  b1 = maxpool2d(b1, b1_shape, 3, 1, 1, name + "_bn1");                        // -> {112, 112, 128}

  Shape b2_shape = shape;
  Node b2 = convtranspose2d(x, b2_shape, 256, 4, 4, 0, false, name + "_up2");  // -> {224, 224, 128}
  Node b2_left = maxpool2d(b2 * -1, b2_shape, 2, 2, 0, name + "_left_pool");   // -> {112, 112, 128}
  Node b2_right = avgpool2d(b2, b2_shape, 2, 2, 0, name + "_right_pool");      // -> {112, 112, 128}
  Node c = add({b1, b2_left, b2_right}, b1_shape, name + "_add");

  Shape out_shape = b1_shape;
  auto out = conv2d(c, out_shape, 128, 3, 1, 1, false, name + "_conv");
  return {out, out_shape};
}

Graph create_tunx_v2_graph(IAllocator &allocator, GraphOpts opts) {
  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 224, 224, 3};

  Node x = conv2d(input, shape, 16, 7, 2, 3, false, "conv1");  // -> {112, 112, 16}
  x = batchnorm(x, shape, true, "bn1");                        // -> {112, 112, 16}
  x = maxpool2d(x, shape, 4, 4, 0, "pool1");                   // -> {56, 56, 16}

  const Shape initial_shape = shape;

  Shape b1_shape = initial_shape;
  Node b1 = convtranspose2d(x, b1_shape, 64, 2, 2, 0, false, "b1_up1");  // -> {112, 112, 64}
  b1 = batchnorm(b1, b1_shape, true, "b1_bn1");                          // -> {112, 112, 64}
  b1 = convtranspose2d(b1, b1_shape, 128, 2, 2, 0, false, "b1_up2");     // -> {224, 224, 128}
  b1 = batchnorm(b1, b1_shape, true, "b1_bn2");                          // -> {224,224, 128}
  b1 = conv2d(b1, b1_shape, 128, 3, 1, 1, false, "b1_down");             // -> {224, 224, 128}
  b1 = maxpool2d(b1, b1_shape, 2, 2, 0, "b1_pool");                      // -> {112, 112, 128}

  Shape b2_shape = initial_shape;
  Node b2 = convtranspose2d(x, b2_shape, 128, 2, 2, 0, false, "b2_trans");  // -> {56, 56, 128}
  b2 = batchnorm(b2, b2_shape, true, "b2_bn1");
  b2 = convtranspose2d(b2, b2_shape, 128, 2, 2, 0, false, "b2_up2");  // -> {112, 112, 128}
  b2 = batchnorm(b2, b2_shape, true, "b2_bn2");
  b2 = conv2d(b2, b2_shape, 128, 3, 1, 1, false, "b2_conv");   // -> {112, 112, 128}
  b2 = conv2d(b2, b2_shape, 128, 3, 1, 1, false, "b2_conv2");  // -> {112, 112, 128}
  b2 = b2 * -1;                                                // -> {112, 112, 128}
  b2 = maxpool2d(b2, b2_shape, 2, 2, 0, "b2_pool");            // -> {56, 56, 128}

  Shape b3_shape = initial_shape;
  Node b3 = conv2d(x, b3_shape, 64, 3, 1, 1, false, "b3_conv");       // -> {56, 56, 64}
  b3 = convtranspose2d(b3, b3_shape, 128, 5, 1, 2, false, "b3_up1");  // {56, 56, 128}
  std::tie(b3, b3_shape) = v2_block(b3, b3_shape, "b3_block");        // -> {112, 112, 128}

  Shape y_shape = b1_shape;
  Node y = add({b1, b2, b3}, y_shape, "merge_b1_b2_b3");
  y = relu(y, y_shape, "out_relu");
  y = flatten(y, y_shape, 1, -1, "flatten");

  Node output = dense(y, y_shape, 100, true, "output");

  finalize_graph(graph, allocator, output, opts);
  return graph;
}

std::pair<Node, Shape> v3_block(Node x, Shape shape, const std::string &name) {
  Shape b1_shape = shape;
  Node b1 = convtranspose2d(x, b1_shape, 128, 2, 2, 0, false, name + "_up1");  // -> {56, 56, 64}
  b1 = maxpool2d(b1, b1_shape, 4, 4, 0, name + "_pool1");                      // -> {28, 28, 64}

  Shape b2_shape = shape;
  Node b2 = convtranspose2d(x, b2_shape, 128, 2, 2, 0, false, name + "_up2");  // -> {56, 56, 128}
  Shape b2_left_shape = b2_shape;
  Node b2_left =
      maxpool2d(b2 * -1, b2_left_shape, 4, 4, 0, name + "_left_pool");  // -> {28, 28, 64}
  Shape b2_right_shape = b2_shape;
  Node b2_right = avgpool2d(b2, b2_right_shape, 4, 4, 0, name + "_right_pool");  // -> {28, 28, 64}
  auto [c, c_shape] = concat({b1, b2_left, b2_right}, {b1_shape, b2_left_shape, b2_right_shape}, 3,
                             name + "_concat");
  c = conv2d(c, c_shape, 64, 3, 1, 1, false, name + "_conv");
  return {c, c_shape};
}

Graph create_tunx_v3_graph(IAllocator &allocator, GraphOpts opts) {
  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 224, 224, 3};

  Node x = conv2d(input, shape, 16, 7, 2, 3, false, "conv1");  // -> {112, 112, 16}
  x = batchnorm(x, shape, true, "bn1");                        // -> {112, 112, 16}
  x = maxpool2d(x, shape, 4, 4, 0, "pool1");                   // -> {28, 28, 16}

  const Shape initial_shape = shape;

  Shape b1_shape = initial_shape;
  Node b1 = convtranspose2d(x, b1_shape, 64, 2, 2, 0, false, "b1_up1");  // -> {56, 56, 32}
  b1 = batchnorm(b1, b1_shape, true, "b1_bn1");
  b1 = convtranspose2d(b1, b1_shape, 128, 2, 2, 0, false, "b1_up2");  // -> {112, 112, 128}
  b1 = batchnorm(b1, b1_shape, true, "b1_bn2");
  b1 = conv2d(b1, b1_shape, 64, 3, 1, 1, false, "b1_down");  // -> {112, 112, 64}
  b1 = maxpool2d(b1, b1_shape, 2, 2, 0, "b1_pool");          // -> {56, 56, 64}

  Shape b2_shape = initial_shape;
  Node b2 = convtranspose2d(x, b2_shape, 64, 2, 2, 0, false, "b2_trans");  // -> {56, 56, 64}
  b2 = batchnorm(b2, b2_shape, true, "b2_bn1");
  b2 = convtranspose2d(b2, b2_shape, 64, 2, 2, 0, false, "b2_up2");  // -> {112, 112, 64}
  b2 = batchnorm(b2, b2_shape, true, "b2_bn2");
  b2 = conv2d(b2, b2_shape, 64, 3, 1, 1, false, "b2_conv");   // -> {112, 112, 64}
  b2 = conv2d(b2, b2_shape, 64, 3, 1, 1, false, "b2_conv2");  // -> {112, 112, 64}
  b2 = b2 * -1;                                               // -> {112, 112, 64}
  b2 = maxpool2d(b2, b2_shape, 2, 2, 0, "b2_pool");           // -> {56, 56, 64}

  Shape b3_shape = initial_shape;
  Node b3 = conv2d(x, b3_shape, 32, 3, 1, 1, false, "b3_conv");      // -> {28, 28, 32}
  b3 = convtranspose2d(b3, b3_shape, 64, 2, 2, 0, false, "b3_up1");  // {56, 56, 64}
  b3 = convtranspose2d(b3, b3_shape, 64, 2, 2, 0, false, "b3_up2");  // {112, 112, 64}
  std::tie(b3, b3_shape) = v3_block(b3, b3_shape, "b3_block");       // -> {56, 56, 64}

  Shape y_shape = b1_shape;
  Node y = add({b1, b2, b3}, y_shape, "merge_b1_b2_b3");
  y = relu(y, y_shape, "out_relu");
  y = flatten(y, y_shape, 1, -1, "flatten");

  Node output = dense(y, y_shape, 100, true, "output");

  finalize_graph(graph, allocator, output, opts);
  return graph;
}

std::pair<Node, Shape> recursive_wide_fork_join(Node x, Shape shape, size_t depth,
                                                const std::string &name) {
  constexpr size_t kOutputChannels = 64;

  // Depth-dependent channel sizes create asymmetric (alpha, beta) descriptors
  // at each recursion level. Inner levels have large join tensors; outer levels
  // have large local tensors. This forces the Full Solver's coordinated
  // fork-join resolution to diverge from the greedy Branch-only / Join-only
  // orderings.
  const size_t kLocalChannels = 64 + depth * 256;
  const size_t kJoinChannels = 832 - depth * 256;

  if (depth == 0) {
    Shape expanded_shape = shape;

    Node expanded =
        convtranspose2d(x, expanded_shape, kJoinChannels, 2, 2, 0, false, name + "_expand");

    Node negative = expanded * -1;
    negative->set_uid(name + "_neg");

    Shape left_shape = expanded_shape;
    Node left = maxpool2d(negative, left_shape, 2, 2, 0, name + "_left_pool");

    Shape right_shape = expanded_shape;
    Node right = avgpool2d(expanded, right_shape, 2, 2, 0, name + "_right_pool");

    Shape mid_shape = expanded_shape;
    Node mid = conv2d(expanded, mid_shape, kJoinChannels, 3, 2, 1, false, name + "_mid_conv");

    Shape joined_shape = left_shape;
    Node joined = add({left, right, mid}, joined_shape, name + "_join");

    Node output = conv2d(joined, joined_shape, kOutputChannels, 1, 1, 0, false, name + "_compress");

    return {output, joined_shape};
  }

  // ── Left branch: local (expand → pool → widen) ──
  Shape local_shape = shape;

  Node local =
      convtranspose2d(x, local_shape, kLocalChannels, 2, 2, 0, false, name + "_local_expand");

  local = maxpool2d(local, local_shape, 2, 2, 0, name + "_local_pool");

  local = conv2d(local, local_shape, kJoinChannels, 1, 1, 0, false, name + "_local_widen");

  // ── Right branch: complete recursively nested fork–join ──
  Shape nested_shape = shape;

  Node nested = conv2d(x, nested_shape, kOutputChannels, 1, 1, 0, false, name + "_nested_seed");

  std::tie(nested, nested_shape) = recursive_wide_fork_join(
      nested, nested_shape, depth - 1, name + "_d" + std::to_string(depth - 1));

  nested = conv2d(nested, nested_shape, kJoinChannels, 1, 1, 0, false, name + "_nested_widen");

  // ── Binary parent join ──
  Shape joined_shape = local_shape;

  Node joined = add({local, nested}, joined_shape, name + "_join");

  Node output = conv2d(joined, joined_shape, kOutputChannels, 1, 1, 0, false, name + "_compress");

  return {output, joined_shape};
}

Graph create_tunx_v4_graph(IAllocator &allocator, GraphOpts opts) {
  Graph graph;

  Node input = graph.input("input");
  Shape shape = {1, 224, 224, 3};

  Node x = conv2d(input, shape, 32, 7, 2, 3, false, "stem_conv");
  // [32, 112, 112, 32]

  x = batchnorm(x, shape, true, "stem_bn");

  x = maxpool2d(x, shape, 2, 2, 0, "stem_pool");
  // [32, 56, 56, 32]

  x = conv2d(x, shape, 64, 1, 1, 0, false, "stem_project");
  // [32, 56, 56, 64]

  /*
   * Three parent fork–joins plus one base fork–join:
   *
   *   rfj_d3
   *      └── rfj_d2
   *             └── rfj_d1
   *                    └── rfj_d0
   */
  std::tie(x, shape) = recursive_wide_fork_join(x, shape, 3, "rfj_d3");

  x = relu(x, shape, "out_relu");

  x = avgpool2d(x, shape, 4, 4, 0, "out_pool");
  // [32, 14, 14, 64]

  x = flatten(x, shape, 1, -1, "flatten");

  Node output = dense(x, shape, 100, true, "output");

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
  register_graph("tunx_v4", create_tunx_v4_graph);
}
}  // namespace tunx