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

#include "nn/graph.hpp"
#include "nn/layer_factory.hpp"
#include "nn/layers_impl/concat.hpp"
#include "type/type.hpp"

namespace tunx {

namespace {

using Shape = Vec<size_t>;

size_t channels(const Shape &shape) {
  if (shape.empty()) {
    throw std::runtime_error("Shape is empty");
  }
  return shape.back();
}

Node conv2d(Node input, Shape &shape, size_t out_channels, size_t kernel_h, size_t kernel_w,
            size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, bool use_bias,
            const std::string &name) {
  auto layer = Conv2D(channels(shape), out_channels, kernel_h, kernel_w, stride_h, stride_w, pad_h,
                      pad_w, use_bias, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

Node batchnorm(Node input, Shape &shape, bool use_relu, const std::string &name) {
  auto layer = BatchNorm(channels(shape), dtype_eps(DType_t::FP32), 0.1f, true, use_relu, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

Node maxpool2d(Node input, Shape &shape, size_t pool_h, size_t pool_w, size_t stride_h,
               size_t stride_w, size_t pad_h, size_t pad_w, const std::string &name) {
  auto layer = MaxPool2D(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

Node avgpool2d(Node input, Shape &shape, size_t pool_h, size_t pool_w, size_t stride_h,
               size_t stride_w, size_t pad_h, size_t pad_w, const std::string &name) {
  auto layer = AvgPool2D(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

Node flatten(Node input, Shape &shape, int start_dim, int end_dim, const std::string &name) {
  auto layer = Flatten(start_dim, end_dim, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

Node dense(Node input, Shape &shape, size_t output_features, bool use_bias,
           const std::string &name) {
  auto layer = Dense(channels(shape), output_features, use_bias, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

Node relu(Node input, Shape &shape, const std::string &name) {
  auto layer = ReLU(name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

Node gelu(Node input, Shape &shape, const std::string &name) {
  auto layer = GELU(name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

Node layernorm(Node input, Shape &shape, float epsilon, bool affine, const std::string &name) {
  auto layer = LayerNorm(channels(shape), epsilon, affine, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

Node dropout(Node input, Shape &shape, float dropout_rate, const std::string &name) {
  auto layer = Dropout(dropout_rate, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

Node embedding(Node input, Shape &shape, size_t vocab_size, size_t embed_dim,
               const std::string &name) {
  auto layer = Embedding(vocab_size, embed_dim, static_cast<size_t>(-1), name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

Node positional_embedding(Node input, Shape &shape, size_t embed_dim, size_t seq_len,
                          const std::string &name) {
  auto layer = PositionalEmbedding(embed_dim, seq_len, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

Node class_token(Node input, Shape &shape, size_t embed_dim, const std::string &name) {
  auto layer = ClassToken(embed_dim, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

Node slice(Node input, Shape &shape, size_t axis, size_t start, size_t length,
           const std::string &name) {
  auto layer = Slice(axis, start, length, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

Node flash_attention(Node input, Shape &shape, size_t embed_dim, size_t num_heads, bool is_causal,
                     const std::string &name) {
  auto layer = FlashAttentionBlock(embed_dim, num_heads, is_causal, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

Node basic_residual_block(Node input, Shape &shape, size_t out_channels, size_t stride,
                          const std::string &name) {
  Shape main_shape = shape;
  Node main =
      conv2d(input, main_shape, out_channels, 3, 3, stride, stride, 1, 1, false, name + "_conv1");
  main = batchnorm(main, main_shape, true, name + "_bn0");
  main = conv2d(main, main_shape, out_channels, 3, 3, 1, 1, 1, 1, false, name + "_conv2");
  main = batchnorm(main, main_shape, false, name + "_bn1");

  Shape shortcut_shape = shape;
  Node shortcut = input;
  if (stride != 1 || channels(shape) != out_channels) {
    shortcut = conv2d(shortcut, shortcut_shape, out_channels, 1, 1, stride, stride, 0, 0, false,
                      name + "_conv0");
    shortcut = batchnorm(shortcut, shortcut_shape, false, name + "_shortcut_bn");
  }

  shape = main_shape;

  auto output = main + shortcut;
  output = relu(output, shape, name + "_relu");
  return output;
}

Node wide_residual_block(Node input, Shape &shape, size_t out_channels, size_t stride,
                         float dropout_rate, const std::string &name) {
  Shape main_shape = shape;
  Node main = batchnorm(input, main_shape, true, name + "_bn1");
  main = conv2d(main, main_shape, out_channels, 3, 3, stride, stride, 1, 1, true, name + "_conv1");
  main = batchnorm(main, main_shape, true, name + "_bn2");
  if (dropout_rate > 0.0f) {
    main = dropout(main, main_shape, dropout_rate, name + "_dropout");
  }
  main = conv2d(main, main_shape, out_channels, 3, 3, 1, 1, 1, 1, true, name + "_conv2");

  Shape shortcut_shape = shape;
  Node shortcut = input;
  if (stride != 1 || channels(shape) != out_channels) {
    shortcut = conv2d(shortcut, shortcut_shape, out_channels, 1, 1, stride, stride, 0, 0, false,
                      name + "_shortcut_conv");
  }

  shape = main_shape;

  return main + shortcut;
}

Node bottleneck_residual_block(Node input, Shape &shape, size_t mid_channels, size_t out_channels,
                               size_t stride, const std::string &name) {
  Shape main_shape = shape;
  Node main = conv2d(input, main_shape, mid_channels, 1, 1, 1, 1, 0, 0, false, name + "_conv1");
  main = batchnorm(main, main_shape, true, name + "_bn0");
  main = conv2d(main, main_shape, mid_channels, 3, 3, stride, stride, 1, 1, false, name + "_conv2");
  main = batchnorm(main, main_shape, true, name + "_bn1");
  main = conv2d(main, main_shape, out_channels, 1, 1, 1, 1, 0, 0, false, name + "_conv3");
  main = batchnorm(main, main_shape, true, name + "_bn2");

  Shape shortcut_shape = shape;
  Node shortcut = input;
  if (stride != 1 || channels(shape) != out_channels) {
    shortcut = conv2d(shortcut, shortcut_shape, out_channels, 1, 1, stride, stride, 0, 0, false,
                      name + "_conv0");
    shortcut = batchnorm(shortcut, shortcut_shape, false, name + "_bn3");
  }

  shape = main_shape;
  return main + shortcut;
}

Node gpt_block(Node input, Shape &shape, size_t embed_dim, size_t num_heads, size_t ffn_dim,
               float dropout_rate, bool is_causal, const std::string &name) {
  Shape attn_shape = shape;
  Node attn = layernorm(input, attn_shape, 1e-5f, true, name + "_ln_1");
  attn = flash_attention(attn, attn_shape, embed_dim, num_heads, is_causal, name + "_attn");
  attn = dropout(attn, attn_shape, dropout_rate, name + "_attn_dropout");
  Node x = input + attn;

  Shape ffn_shape = shape;
  Node ffn = layernorm(x, ffn_shape, 1e-5f, true, name + "_ln_2");
  ffn = dense(ffn, ffn_shape, ffn_dim, true, name + "_mlp_fc1");
  ffn = gelu(ffn, ffn_shape, name + "_mlp_activation");
  ffn = dense(ffn, ffn_shape, embed_dim, true, name + "_mlp_fc2");
  ffn = dropout(ffn, ffn_shape, dropout_rate, name + "_mlp_dropout");

  shape = ffn_shape;
  return x + ffn;
}

Node concat(const Vec<Node> &inputs, size_t axis, const std::string &name) {
  auto layer = Concat(axis, name);
  return layer(inputs[0], inputs[1], inputs[2],
               inputs[3]);  // Currently supporting 4 inputs since inception uses 4
}

Node inception_block(Node input, Shape &shape, size_t out_channels, const std::string &name) {
  Shape b1_shape = shape;
  Node b1 = conv2d(input, b1_shape, out_channels, 1, 1, 1, 1, 0, 0, false, name + "_b1_conv");
  b1 = batchnorm(b1, b1_shape, true, name + "_b1_bn");

  Shape b2_shape = shape;
  Node b2 = conv2d(input, b2_shape, out_channels, 1, 1, 1, 1, 0, 0, false, name + "_b2_conv1");
  b2 = batchnorm(b2, b2_shape, true, name + "_b2_bn1");
  b2 = conv2d(b2, b2_shape, out_channels, 3, 3, 1, 1, 1, 1, false, name + "_b2_conv2");
  b2 = batchnorm(b2, b2_shape, true, name + "_b2_bn2");

  Shape b3_shape = shape;
  Node b3 = conv2d(input, b3_shape, out_channels, 1, 1, 1, 1, 0, 0, false, name + "_b3_conv1");
  b3 = batchnorm(b3, b3_shape, true, name + "_b3_bn1");
  b3 = conv2d(b3, b3_shape, out_channels, 5, 5, 1, 1, 2, 2, false, name + "_b3_conv2");
  b3 = batchnorm(b3, b3_shape, true, name + "_b3_bn2");

  Shape b4_shape = shape;
  Node b4 = maxpool2d(input, b4_shape, 3, 3, 1, 1, 1, 1, name + "_b4_pool");
  b4 = conv2d(b4, b4_shape, out_channels, 1, 1, 1, 1, 0, 0, false, name + "_b4_conv");
  b4 = batchnorm(b4, b4_shape, true, name + "_b4_bn");

  shape = b1_shape;
  shape[3] = b1_shape[3] + b2_shape[3] + b3_shape[3] + b4_shape[3];
  std::cout << "Inception block " + name + " shape: ";
  for (size_t i = 0; i < shape.size(); i++) {
    std::cout << shape[i] << " ";
  }
  std::cout << std::endl;

  Node out = concat({b1, b2, b3, b4}, 3, name + "_concat");
  // Node out = b1 + b2 + b3 + b4;
  return relu(out, shape, name + "_relu");
}

void finalize_graph(Graph &graph, IAllocator &allocator, const Node &output, GraphOpts opts) {
  output->set_uid("output");
  graph.set_output(output);
  graph.compile(allocator, opts);
}

Graph create_mnist_graph(IAllocator &allocator, GraphOpts opts) {
  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 28, 28, 1};

  Node x = conv2d(input, shape, 8, 5, 5, 1, 1, 0, 0, false, "conv1");
  x = batchnorm(x, shape, true, "bn1");
  x = maxpool2d(x, shape, 3, 3, 3, 3, 0, 0, "pool1");
  x = conv2d(x, shape, 16, 1, 1, 1, 1, 0, 0, false, "conv2_1x1");
  x = batchnorm(x, shape, true, "bn2_1x1");
  x = relu(x, shape, "relu2");
  x = conv2d(x, shape, 48, 5, 5, 1, 1, 0, 0, false, "conv3");
  x = batchnorm(x, shape, true, "bn3");
  x = maxpool2d(x, shape, 2, 2, 2, 2, 0, 0, "pool2");
  x = flatten(x, shape, 1, -1, "flatten");
  Node output = dense(x, shape, 10, false, "output");
  finalize_graph(graph, allocator, output, opts);
  return graph;
}

Graph create_cifar10_vgg_graph(IAllocator &allocator, GraphOpts opts) {
  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 32, 32, 3};

  Node x = conv2d(input, shape, 64, 3, 3, 1, 1, 1, 1, false, "conv0");
  x = batchnorm(x, shape, true, "bn0");
  x = conv2d(x, shape, 64, 3, 3, 1, 1, 1, 1, false, "conv1");
  x = batchnorm(x, shape, true, "bn1");
  x = maxpool2d(x, shape, 2, 2, 2, 2, 0, 0, "pool0");
  x = conv2d(x, shape, 128, 3, 3, 1, 1, 1, 1, false, "conv2");
  x = batchnorm(x, shape, true, "bn2");
  x = conv2d(x, shape, 128, 3, 3, 1, 1, 1, 1, false, "conv3");
  x = batchnorm(x, shape, true, "bn3");
  x = maxpool2d(x, shape, 2, 2, 2, 2, 0, 0, "pool1");
  x = conv2d(x, shape, 256, 3, 3, 1, 1, 1, 1, false, "conv4");
  x = batchnorm(x, shape, true, "bn4");
  x = conv2d(x, shape, 256, 3, 3, 1, 1, 1, 1, false, "conv5");
  x = relu(x, shape, "relu5");
  x = conv2d(x, shape, 256, 3, 3, 1, 1, 1, 1, false, "conv6");
  x = batchnorm(x, shape, true, "bn6");
  x = maxpool2d(x, shape, 2, 2, 2, 2, 0, 0, "pool2");
  x = conv2d(x, shape, 512, 3, 3, 1, 1, 1, 1, false, "conv7");
  x = batchnorm(x, shape, true, "bn7");
  x = conv2d(x, shape, 512, 3, 3, 1, 1, 1, 1, false, "conv8");
  x = batchnorm(x, shape, true, "bn8");
  x = conv2d(x, shape, 512, 3, 3, 1, 1, 1, 1, false, "conv9");
  x = batchnorm(x, shape, true, "bn9");
  x = maxpool2d(x, shape, 2, 2, 2, 2, 0, 0, "pool3");
  x = flatten(x, shape, 1, -1, "flatten");
  x = dense(x, shape, 512, true, "fc0");
  x = relu(x, shape, "relu10");
  Node output = dense(x, shape, 10, true, "fc1");
  finalize_graph(graph, allocator, output, opts);
  return graph;
}

Graph create_cifar10_test_graph(IAllocator &allocator, GraphOpts opts) {
  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 32, 32, 3};

  Node x = conv2d(input, shape, 64, 3, 3, 1, 1, 1, 1, false, "conv1");
  x = batchnorm(x, shape, true, "bn1");
  x = conv2d(x, shape, 128, 3, 3, 1, 1, 1, 1, false, "conv2");
  x = batchnorm(x, shape, true, "bn2");
  x = maxpool2d(x, shape, 2, 2, 2, 2, 0, 0, "pool1");
  x = basic_residual_block(x, shape, 256, 1, "res_block1");
  x = maxpool2d(x, shape, 2, 2, 2, 2, 0, 0, "pool2");
  x = basic_residual_block(x, shape, 512, 1, "res_block2");
  x = maxpool2d(x, shape, 2, 2, 2, 2, 0, 0, "pool3");
  x = avgpool2d(x, shape, 4, 4, 1, 1, 0, 0, "avgpool");
  x = flatten(x, shape, 1, -1, "flatten");
  Node output = dense(x, shape, 10, true, "output");
  finalize_graph(graph, allocator, output, opts);
  return graph;
}

Graph create_cifar10_resnet9_graph(IAllocator &allocator, GraphOpts opts) {
  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 32, 32, 3};

  Node x = conv2d(input, shape, 64, 3, 3, 1, 1, 1, 1, false, "conv1");
  x = batchnorm(x, shape, true, "bn1");
  x = conv2d(x, shape, 128, 3, 3, 1, 1, 1, 1, false, "conv2");
  x = batchnorm(x, shape, true, "bn2");
  x = maxpool2d(x, shape, 2, 2, 2, 2, 0, 0, "pool1");
  x = basic_residual_block(x, shape, 128, 1, "res_block1");
  x = basic_residual_block(x, shape, 128, 1, "res_block2");
  x = conv2d(x, shape, 256, 3, 3, 1, 1, 1, 1, false, "conv3");
  x = batchnorm(x, shape, true, "bn3");
  x = maxpool2d(x, shape, 2, 2, 2, 2, 0, 0, "pool2");
  x = basic_residual_block(x, shape, 256, 1, "res_block3");
  x = basic_residual_block(x, shape, 256, 1, "res_block4");
  x = conv2d(x, shape, 512, 3, 3, 1, 1, 1, 1, false, "conv4");
  x = batchnorm(x, shape, true, "bn4");
  x = maxpool2d(x, shape, 2, 2, 2, 2, 0, 0, "pool3");
  x = basic_residual_block(x, shape, 512, 1, "res_block5");
  x = avgpool2d(x, shape, 4, 4, 1, 1, 0, 0, "avgpool");
  x = flatten(x, shape, 1, -1, "flatten");
  Node output = dense(x, shape, 10, true, "output");
  finalize_graph(graph, allocator, output, opts);
  return graph;
}

Graph create_cifar100_resnet18_graph(IAllocator &allocator, GraphOpts opts) {
  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 32, 32, 3};

  Node x = conv2d(input, shape, 32, 3, 3, 1, 1, 1, 1, false, "conv1");
  x = batchnorm(x, shape, true, "bn1");
  x = maxpool2d(x, shape, 2, 2, 2, 2, 0, 0, "maxpool");
  x = basic_residual_block(x, shape, 64, 1, "layer1_block1");
  x = basic_residual_block(x, shape, 64, 1, "layer1_block2");
  x = basic_residual_block(x, shape, 128, 2, "layer2_block1");
  x = basic_residual_block(x, shape, 128, 1, "layer2_block2");
  x = basic_residual_block(x, shape, 256, 2, "layer3_block1");
  x = basic_residual_block(x, shape, 256, 1, "layer3_block2");
  x = basic_residual_block(x, shape, 512, 2, "layer4_block1");
  x = basic_residual_block(x, shape, 512, 1, "layer4_block2");
  x = avgpool2d(x, shape, 2, 2, 1, 1, 0, 0, "avgpool");
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

  Node x = conv2d(input, shape, 16, 3, 3, 1, 1, 1, 1, true, "conv1");
  x = wide_residual_block(x, shape, c1, 1, dropout_rate, "group1_block1");
  x = wide_residual_block(x, shape, c1, 1, dropout_rate, "group1_block2");
  x = wide_residual_block(x, shape, c2, 2, dropout_rate, "group2_block1");
  x = wide_residual_block(x, shape, c2, 1, dropout_rate, "group2_block2");
  x = wide_residual_block(x, shape, c3, 2, dropout_rate, "group3_block1");
  x = wide_residual_block(x, shape, c3, 1, dropout_rate, "group3_block2");
  x = batchnorm(x, shape, true, "bn_final");
  x = avgpool2d(x, shape, 8, 8, 1, 1, 0, 0, "avgpool");
  x = flatten(x, shape, 1, -1, "flatten");
  Node output = dense(x, shape, 100, true, "fc");
  finalize_graph(graph, allocator, output, opts);
  return graph;
}

Graph create_tiny_imagenet_resnet18_graph(IAllocator &allocator, GraphOpts opts) {
  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 64, 64, 3};

  Node x = conv2d(input, shape, 32, 3, 3, 1, 1, 1, 1, false, "conv1");
  x = batchnorm(x, shape, true, "bn1");
  x = maxpool2d(x, shape, 2, 2, 2, 2, 0, 0, "maxpool");
  x = basic_residual_block(x, shape, 64, 1, "layer1_block1");
  x = basic_residual_block(x, shape, 64, 1, "layer1_block2");
  x = basic_residual_block(x, shape, 128, 2, "layer2_block1");
  x = basic_residual_block(x, shape, 128, 1, "layer2_block2");
  x = basic_residual_block(x, shape, 256, 2, "layer3_block1");
  x = basic_residual_block(x, shape, 256, 1, "layer3_block2");
  x = basic_residual_block(x, shape, 512, 2, "layer4_block1");
  x = basic_residual_block(x, shape, 512, 1, "layer4_block2");
  x = avgpool2d(x, shape, 4, 4, 1, 1, 0, 0, "avgpool");
  x = flatten(x, shape, 1, -1, "flatten");
  Node output = dense(x, shape, 200, true, "fc");
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

  Node x = conv2d(input, shape, 16, 3, 3, 1, 1, 1, 1, true, "conv1");
  x = wide_residual_block(x, shape, c1, 1, dropout_rate, "group1_block1");
  x = wide_residual_block(x, shape, c1, 1, dropout_rate, "group1_block2");
  x = wide_residual_block(x, shape, c2, 2, dropout_rate, "group2_block1");
  x = wide_residual_block(x, shape, c2, 1, dropout_rate, "group2_block2");
  x = wide_residual_block(x, shape, c3, 2, dropout_rate, "group3_block1");
  x = wide_residual_block(x, shape, c3, 1, dropout_rate, "group3_block2");
  x = batchnorm(x, shape, true, "bn_final");
  x = avgpool2d(x, shape, 8, 8, 1, 1, 0, 0, "avgpool");
  x = flatten(x, shape, 1, -1, "flatten");
  Node output = dense(x, shape, 200, true, "fc");
  finalize_graph(graph, allocator, output, opts);
  return graph;
}

Graph create_tiny_imagenet_resnet50_graph(IAllocator &allocator, GraphOpts opts) {
  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 64, 64, 3};

  Node x = conv2d(input, shape, 64, 3, 3, 1, 1, 1, 1, true, "conv1");
  x = batchnorm(x, shape, true, "bn1");
  x = maxpool2d(x, shape, 3, 3, 2, 2, 1, 1, "maxpool");
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
  x = avgpool2d(x, shape, 4, 4, 1, 1, 0, 0, "avgpool");
  x = flatten(x, shape, 1, -1, "flatten");
  Node output = dense(x, shape, 200, true, "fc");
  finalize_graph(graph, allocator, output, opts);
  return graph;
}

Graph create_resnet50_imagenet_graph(IAllocator &allocator, GraphOpts opts) {
  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 224, 224, 3};

  Node x = conv2d(input, shape, 64, 7, 7, 2, 2, 3, 3, true, "conv1");
  x = batchnorm(x, shape, true, "bn1");
  x = maxpool2d(x, shape, 3, 3, 2, 2, 1, 1, "maxpool");
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
  x = avgpool2d(x, shape, 7, 7, 1, 1, 0, 0, "avgpool");
  x = flatten(x, shape, 1, -1, "flatten");
  Node output = dense(x, shape, 1000, true, "fc");
  finalize_graph(graph, allocator, output, opts);
  return graph;
}

Graph create_imagenet100_resnet50_graph(IAllocator &allocator, GraphOpts opts) {
  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 224, 224, 3};

  Node x = conv2d(input, shape, 64, 7, 7, 2, 2, 3, 3, true, "conv1");
  x = batchnorm(x, shape, true, "bn1");
  x = maxpool2d(x, shape, 3, 3, 2, 2, 1, 1, "maxpool");
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
  x = avgpool2d(x, shape, 7, 7, 1, 1, 0, 0, "avgpool");
  x = flatten(x, shape, 1, -1, "flatten");
  Node output = dense(x, shape, 100, true, "fc");
  finalize_graph(graph, allocator, output, opts);
  return graph;
}

Graph create_tiny_imagenet_vit_graph(IAllocator &allocator, GraphOpts opts) {
  constexpr size_t patch_size = 4;
  constexpr size_t embed_dim = 256;
  constexpr size_t num_heads = 4;
  constexpr size_t mlp_ratio = 4;
  constexpr size_t depth = 4;
  constexpr size_t num_classes = 200;
  constexpr size_t num_patches = (64 / patch_size) * (64 / patch_size);
  constexpr size_t seq_len = num_patches + 1;

  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 64, 64, 3};

  Node x = conv2d(input, shape, embed_dim, patch_size, patch_size, patch_size, patch_size, 0, 0,
                  true, "patch_embed");
  x = flatten(x, shape, 1, 2, "flatten_patches");
  x = class_token(x, shape, embed_dim, "class_token");
  x = positional_embedding(x, shape, embed_dim, seq_len, "positional_embedding");
  x = dropout(x, shape, 0.1f, "dropout");

  for (size_t i = 0; i < depth; ++i) {
    x = gpt_block(x, shape, embed_dim, num_heads, embed_dim * mlp_ratio, 0.1f, false,
                  "encoder_" + std::to_string(i));
  }

  x = layernorm(x, shape, dtype_eps(DType_t::FP32), true, "ln_final");
  x = slice(x, shape, 1, 0, 1, "extract_cls");
  x = flatten(x, shape, 1, -1, "flatten_cls");
  Node output = dense(x, shape, num_classes, true, "head");
  finalize_graph(graph, allocator, output, opts);
  return graph;
}

Graph create_tiny_imagenet_flash_vit_graph(IAllocator &allocator, GraphOpts opts) {
  constexpr size_t patch_size = 4;
  constexpr size_t embed_dim = 256;
  constexpr size_t num_heads = 4;
  constexpr size_t mlp_ratio = 4;
  constexpr size_t depth = 4;
  constexpr size_t num_classes = 200;
  constexpr size_t num_patches = (64 / patch_size) * (64 / patch_size);
  constexpr size_t seq_len = num_patches + 1;

  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 64, 64, 3};

  Node x = conv2d(input, shape, embed_dim, patch_size, patch_size, patch_size, patch_size, 0, 0,
                  true, "patch_embed");
  x = flatten(x, shape, 1, 2, "flatten_patches");
  x = class_token(x, shape, embed_dim, "class_token");
  x = positional_embedding(x, shape, embed_dim, seq_len, "positional_embedding");
  x = dropout(x, shape, 0.1f, "dropout");

  for (size_t i = 0; i < depth; ++i) {
    x = gpt_block(x, shape, embed_dim, num_heads, embed_dim * mlp_ratio, 0.1f, false,
                  "encoder_" + std::to_string(i));
  }

  x = layernorm(x, shape, dtype_eps(DType_t::FP32), true, "ln_final");
  // x = slice(x, shape, 1, 0, 1, "extract_cls");
  x = flatten(x, shape, 1, -1, "flatten_cls");
  Node output = dense(x, shape, num_classes, true, "head");
  finalize_graph(graph, allocator, output, opts);
  return graph;
}

Graph create_gpt2_graph(IAllocator &allocator, size_t embed_dim, size_t num_heads,
                        size_t num_layers, const std::string &name,
                        GraphOpts opts) {
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

  Node x = conv2d(input, shape, 64, 7, 7, 2, 2, 3, 3, false, "conv1");
  x = batchnorm(x, shape, true, "bn1");
  x = maxpool2d(x, shape, 3, 3, 2, 2, 1, 1, "pool1");

  x = inception_block(x, shape, 32, "inc1");
  x = inception_block(x, shape, 64, "inc2");
  x = maxpool2d(x, shape, 3, 3, 2, 2, 1, 1, "pool2");

  x = inception_block(x, shape, 128, "inc3");
  x = inception_block(x, shape, 128, "inc4");

  x = avgpool2d(x, shape, 28, 28, 1, 1, 0, 0, "avgpool");
  x = flatten(x, shape, 1, -1, "flatten");
  Node output = dense(x, shape, 100, true, "output");

  finalize_graph(graph, allocator, output, opts);
  return graph;
}

Graph create_unet_topology_graph(IAllocator &allocator, GraphOpts opts) {
  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 32, 32, 3};

  // Encoder
  Shape e1_shape = shape;
  Node e1 = conv2d(input, e1_shape, 64, 3, 3, 1, 1, 1, 1, false, "e1_conv1");
  e1 = batchnorm(e1, e1_shape, true, "e1_bn1");
  e1 = conv2d(e1, e1_shape, 64, 3, 3, 1, 1, 1, 1, false, "e1_conv2");
  e1 = batchnorm(e1, e1_shape, true, "e1_bn2");

  Shape e2_shape = e1_shape;
  Node e2 = conv2d(e1, e2_shape, 128, 3, 3, 1, 1, 1, 1, false, "e2_conv1");
  e2 = batchnorm(e2, e2_shape, true, "e2_bn1");
  e2 = conv2d(e2, e2_shape, 128, 3, 3, 1, 1, 1, 1, false, "e2_conv2");
  e2 = batchnorm(e2, e2_shape, true, "e2_bn2");

  Shape e3_shape = e2_shape;
  Node e3 = conv2d(e2, e3_shape, 256, 3, 3, 1, 1, 1, 1, false, "e3_conv1");
  e3 = batchnorm(e3, e3_shape, true, "e3_bn1");
  e3 = conv2d(e3, e3_shape, 256, 3, 3, 1, 1, 1, 1, false, "e3_conv2");
  e3 = batchnorm(e3, e3_shape, true, "e3_bn2");

  // Bottleneck
  Shape b_shape = e3_shape;
  Node b = conv2d(e3, b_shape, 256, 3, 3, 1, 1, 1, 1, false, "b_conv1");
  b = batchnorm(b, b_shape, true, "b_bn1");
  b = conv2d(b, b_shape, 256, 3, 3, 1, 1, 1, 1, false, "b_conv2");
  b = batchnorm(b, b_shape, true, "b_bn2");

  // Decoder
  Shape d3_shape = b_shape;
  Node d3 = b + e3;
  d3 = conv2d(d3, d3_shape, 128, 3, 3, 1, 1, 1, 1, false, "d3_conv1");
  d3 = batchnorm(d3, d3_shape, true, "d3_bn1");
  d3 = conv2d(d3, d3_shape, 128, 3, 3, 1, 1, 1, 1, false, "d3_conv2");
  d3 = batchnorm(d3, d3_shape, true, "d3_bn2");

  Shape d2_shape = d3_shape;
  Node d2 = d3 + e2;
  d2 = conv2d(d2, d2_shape, 64, 3, 3, 1, 1, 1, 1, false, "d2_conv1");
  d2 = batchnorm(d2, d2_shape, true, "d2_bn1");
  d2 = conv2d(d2, d2_shape, 64, 3, 3, 1, 1, 1, 1, false, "d2_conv2");
  d2 = batchnorm(d2, d2_shape, true, "d2_bn2");

  Shape d1_shape = d2_shape;
  Node d1 = d2 + e1;
  d1 = conv2d(d1, d1_shape, 32, 3, 3, 1, 1, 1, 1, false, "d1_conv1");
  d1 = batchnorm(d1, d1_shape, true, "d1_bn1");
  d1 = conv2d(d1, d1_shape, 32, 3, 3, 1, 1, 1, 1, false, "d1_conv2");
  d1 = batchnorm(d1, d1_shape, true, "d1_bn2");

  shape = d1_shape;
  Node x = avgpool2d(d1, shape, 32, 32, 1, 1, 0, 0, "avgpool");
  x = flatten(x, shape, 1, -1, "flatten");
  Node output = dense(x, shape, 10, true, "output");

  finalize_graph(graph, allocator, output, opts);
  return graph;
}

Graph create_fpn_topology_graph(IAllocator &allocator, GraphOpts opts) {
  Graph graph;
  Node input = graph.input("input");
  Shape shape = {1, 32, 32, 3};

  Shape c1_shape = shape;
  Node c1 = conv2d(input, c1_shape, 64, 3, 3, 1, 1, 1, 1, false, "c1_conv");
  c1 = batchnorm(c1, c1_shape, true, "c1_bn");

  Shape c2_shape = c1_shape;
  Node c2 = conv2d(c1, c2_shape, 128, 3, 3, 1, 1, 1, 1, false, "c2_conv");
  c2 = batchnorm(c2, c2_shape, true, "c2_bn");

  Shape c3_shape = c2_shape;
  Node c3 = conv2d(c2, c3_shape, 256, 3, 3, 1, 1, 1, 1, false, "c3_conv");
  c3 = batchnorm(c3, c3_shape, true, "c3_bn");

  Shape c4_shape = c3_shape;
  Node c4 = conv2d(c3, c4_shape, 512, 3, 3, 1, 1, 1, 1, false, "c4_conv");
  c4 = batchnorm(c4, c4_shape, true, "c4_bn");

  Shape p4_shape = c4_shape;
  Node p4 = conv2d(c4, p4_shape, 256, 1, 1, 1, 1, 0, 0, false, "p4_lat");

  Shape p3_shape = c3_shape;
  Node p3_lat = conv2d(c3, p3_shape, 256, 1, 1, 1, 1, 0, 0, false, "p3_lat");
  Node p3 = p4 + p3_lat;
  p3 = conv2d(p3, p3_shape, 256, 3, 3, 1, 1, 1, 1, false, "p3_out");

  Shape p2_shape = c2_shape;
  Node p2_lat = conv2d(c2, p2_shape, 256, 1, 1, 1, 1, 0, 0, false, "p2_lat");
  Node p2 = p3 + p2_lat;
  p2 = conv2d(p2, p2_shape, 256, 3, 3, 1, 1, 1, 1, false, "p2_out");

  Shape p1_shape = c1_shape;
  Node p1_lat = conv2d(c1, p1_shape, 256, 1, 1, 1, 1, 0, 0, false, "p1_lat");
  Node p1 = p2 + p1_lat;
  p1 = conv2d(p1, p1_shape, 256, 3, 3, 1, 1, 1, 1, false, "p1_out");

  Node out = p1 + p2;
  out = out + p3;
  out = out + p4;
  shape = p1_shape;

  out = avgpool2d(out, shape, 32, 32, 1, 1, 0, 0, "avgpool");
  out = flatten(out, shape, 1, -1, "flatten");
  Node output = dense(out, shape, 10, true, "output");

  finalize_graph(graph, allocator, output, opts);
  return graph;
}

}  // namespace

std::unordered_map<std::string, std::function<Graph(IAllocator &, GraphOpts)>>
    ExampleGraphs::creators_;

void ExampleGraphs::register_defaults() {
  register_graph("mnist_cnn", create_mnist_graph);

  register_graph("inception_v1", create_inception_v1_graph);
  register_graph("unet_topology", create_unet_topology_graph);
  register_graph("fpn_topology", create_fpn_topology_graph);

  register_graph("cifar10_vgg", create_cifar10_vgg_graph);
  register_graph("cifar10_test", create_cifar10_test_graph);
  register_graph("cifar10_resnet9", create_cifar10_resnet9_graph);

  register_graph("cifar100_resnet18", create_cifar100_resnet18_graph);
  register_graph("cifar100_wrn16_8", create_cifar100_wrn16_8_graph);

  register_graph("tiny_imagenet_resnet18", create_tiny_imagenet_resnet18_graph);
  register_graph("tiny_imagenet_wrn16_8", create_tiny_imagenet_wrn16_8_graph);
  register_graph("tiny_imagenet_resnet50", create_tiny_imagenet_resnet50_graph);
  register_graph("tiny_imagenet_vit", create_tiny_imagenet_vit_graph);
  register_graph("tiny_imagenet_flash_vit", create_tiny_imagenet_flash_vit_graph);

  register_graph("imagenet_resnet50", create_resnet50_imagenet_graph);
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
}

}  // namespace tunx