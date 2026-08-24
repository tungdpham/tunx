#pragma once

#include <string>

#include "nn/layer_factory.hpp"
#include "nn/layers_impl/concat.hpp"
#include "nn/layers_impl/conv2d_transpose.hpp"
#include "nn/layers_impl/transpose.hpp"
#include "type/type.hpp"

namespace tunx {

using Shape = Vec<size_t>;

inline size_t channels(const Shape &shape) {
  if (shape.empty()) {
    throw std::runtime_error("Shape is empty");
  }
  return shape.back();
}

inline Node add(const Vec<Node> &inputs, Shape &shape, const std::string &name) {
  auto layer = Add(name);
  shape = layer.output_shapes(Vec<Shape>(inputs.size(), shape))[0];
  return layer(inputs);
}

inline Node conv2d(Node input, Shape &shape, size_t out_channels, size_t kernel, size_t stride,
                   size_t pad, bool use_bias, const std::string &name) {
  auto layer = Conv2D(channels(shape), out_channels, kernel, kernel, stride, stride, pad, pad,
                      use_bias, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

inline Node batchnorm(Node input, Shape &shape, bool use_relu, const std::string &name) {
  auto layer = BatchNorm(channels(shape), dtype_eps(DType_t::FP32), 0.1f, true, use_relu, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

inline Node maxpool2d(Node input, Shape &shape, size_t pool, size_t stride, size_t pad,
                      const std::string &name) {
  auto layer = MaxPool2D(pool, pool, stride, stride, pad, pad, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

inline Node avgpool2d(Node input, Shape &shape, size_t pool, size_t stride, size_t pad,
                      const std::string &name) {
  auto layer = AvgPool2D(pool, pool, stride, stride, pad, pad, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

inline Node flatten(Node input, Shape &shape, int start_dim, int end_dim, const std::string &name) {
  auto layer = Flatten(start_dim, end_dim, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

inline Node transpose(Node input, Shape &shape, int dim0, int dim1, const std::string &name) {
  auto layer = Transpose(dim0, dim1, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

inline Node convtranspose2d(Node input, Shape &shape, size_t out_channels, size_t kernel,
                            size_t stride, size_t pad, bool use_bias, const std::string &name) {
  auto layer = ConvTranspose2D(channels(shape), out_channels, kernel, kernel, stride, stride, pad,
                               pad, use_bias, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

inline Node dense(Node input, Shape &shape, size_t output_features, bool use_bias,
                  const std::string &name) {
  auto layer = Dense(channels(shape), output_features, use_bias, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

inline Node relu(Node input, Shape &shape, const std::string &name) {
  auto layer = ReLU(name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

inline Node gelu(Node input, Shape &shape, const std::string &name) {
  auto layer = GELU(name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

inline Node layernorm(Node input, Shape &shape, float epsilon, bool affine,
                      const std::string &name) {
  auto layer = LayerNorm(channels(shape), epsilon, affine, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

inline Node dropout(Node input, Shape &shape, float dropout_rate, const std::string &name) {
  auto layer = Dropout(dropout_rate, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

inline Node embedding(Node input, Shape &shape, size_t vocab_size, size_t embed_dim,
                      const std::string &name) {
  auto layer = Embedding(vocab_size, embed_dim, static_cast<size_t>(-1), name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

inline Node positional_embedding(Node input, Shape &shape, size_t embed_dim, size_t seq_len,
                                 const std::string &name) {
  auto layer = PositionalEmbedding(embed_dim, seq_len, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

inline Node class_token(Node input, Shape &shape, size_t embed_dim, const std::string &name) {
  auto layer = ClassToken(embed_dim, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

inline Node slice(Node input, Shape &shape, size_t axis, size_t start, size_t length,
                  const std::string &name) {
  auto layer = Slice(axis, start, length, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

inline Node attention(Node input, Shape &shape, size_t embed_dim, size_t num_heads, bool is_causal,
                      const std::string &name) {
  auto layer = FlashAttentionBlock(embed_dim, num_heads, is_causal, name);
  shape = layer.output_shapes({shape})[0];
  return layer(input);
}

inline std::pair<Node, Shape> concat(const Vec<Node> &inputs, const Vec<Shape> &shapes, size_t axis,
                                     const std::string &name) {
  auto layer = Concat(axis, name);
  Shape out_shape = layer.output_shapes(shapes)[0];
  return {layer(inputs), out_shape};
}

inline Node basic_residual_block(Node input, Shape &shape, size_t out_channels, size_t stride,
                                 const std::string &name) {
  Shape main_shape = shape;
  Node main = conv2d(input, main_shape, out_channels, 3, stride, 1, false, name + "_conv1");
  main = batchnorm(main, main_shape, true, name + "_bn0");
  main = conv2d(main, main_shape, out_channels, 3, 1, 1, false, name + "_conv2");
  main = batchnorm(main, main_shape, false, name + "_bn1");

  Shape shortcut_shape = shape;
  Node shortcut = input;
  if (stride != 1 || channels(shape) != out_channels) {
    shortcut = conv2d(shortcut, shortcut_shape, out_channels, 1, stride, 0, false, name + "_conv0");
    shortcut = batchnorm(shortcut, shortcut_shape, false, name + "_shortcut_bn");
  }

  shape = main_shape;

  auto output = main + shortcut;
  output = relu(output, shape, name + "_relu");
  return output;
}

inline Node wide_residual_block(Node input, Shape &shape, size_t out_channels, size_t stride,
                                float dropout_rate, const std::string &name) {
  Shape main_shape = shape;
  Node main = batchnorm(input, main_shape, true, name + "_bn1");
  main = conv2d(main, main_shape, out_channels, 3, stride, 1, true, name + "_conv1");
  main = batchnorm(main, main_shape, true, name + "_bn2");
  if (dropout_rate > 0.0f) {
    main = dropout(main, main_shape, dropout_rate, name + "_dropout");
  }
  main = conv2d(main, main_shape, out_channels, 3, 1, 1, true, name + "_conv2");

  Shape shortcut_shape = shape;
  Node shortcut = input;
  if (stride != 1 || channels(shape) != out_channels) {
    shortcut = conv2d(shortcut, shortcut_shape, out_channels, 1, stride, 0, false,
                      name + "_shortcut_conv");
  }

  shape = main_shape;

  return main + shortcut;
}

inline Node bottleneck_residual_block(Node input, Shape &shape, size_t mid_channels,
                                      size_t out_channels, size_t stride, const std::string &name) {
  Shape main_shape = shape;
  Node main = conv2d(input, main_shape, mid_channels, 1, 1, 0, false, name + "_conv1");
  main = batchnorm(main, main_shape, true, name + "_bn0");
  main = conv2d(main, main_shape, mid_channels, 3, stride, 1, false, name + "_conv2");
  main = batchnorm(main, main_shape, true, name + "_bn1");
  main = conv2d(main, main_shape, out_channels, 1, 1, 0, false, name + "_conv3");
  main = batchnorm(main, main_shape, true, name + "_bn2");

  Shape shortcut_shape = shape;
  Node shortcut = input;
  if (stride != 1 || channels(shape) != out_channels) {
    shortcut = conv2d(shortcut, shortcut_shape, out_channels, 1, stride, 0, false, name + "_conv0");
    shortcut = batchnorm(shortcut, shortcut_shape, false, name + "_bn3");
  }

  shape = main_shape;
  return main + shortcut;
}

inline Node gpt_block(Node input, Shape &shape, size_t embed_dim, size_t num_heads, size_t ffn_dim,
                      float dropout_rate, bool is_causal, const std::string &name) {
  Shape attn_shape = shape;
  Node attn = layernorm(input, attn_shape, 1e-5f, true, name + "_ln_1");
  attn = attention(attn, attn_shape, embed_dim, num_heads, is_causal, name + "_attn");
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

inline Node inception_block(Node input, Shape &shape, size_t out_channels,
                            const std::string &name) {
  Shape b1_shape = shape;
  Node b1 = conv2d(input, b1_shape, out_channels, 1, 1, 0, false, name + "_b1_conv");
  b1 = batchnorm(b1, b1_shape, true, name + "_b1_bn");

  Shape b2_shape = shape;
  Node b2 = conv2d(input, b2_shape, out_channels, 1, 1, 0, false, name + "_b2_conv1");
  b2 = batchnorm(b2, b2_shape, true, name + "_b2_bn1");
  b2 = conv2d(b2, b2_shape, out_channels, 3, 1, 1, false, name + "_b2_conv2");
  b2 = batchnorm(b2, b2_shape, true, name + "_b2_bn2");

  Shape b3_shape = shape;
  Node b3 = conv2d(input, b3_shape, out_channels, 1, 1, 0, false, name + "_b3_conv1");
  b3 = batchnorm(b3, b3_shape, true, name + "_b3_bn1");
  b3 = conv2d(b3, b3_shape, out_channels, 5, 1, 2, false, name + "_b3_conv2");
  b3 = batchnorm(b3, b3_shape, true, name + "_b3_bn2");

  Shape b4_shape = shape;
  Node b4 = maxpool2d(input, b4_shape, 3, 1, 1, name + "_b4_pool");
  b4 = conv2d(b4, b4_shape, out_channels, 1, 1, 0, false, name + "_b4_conv");
  b4 = batchnorm(b4, b4_shape, true, name + "_b4_bn");

  auto [out, out_shape] =
      concat({b1, b2, b3, b4}, {b1_shape, b2_shape, b3_shape, b4_shape}, 3, name + "_concat");
  shape = out_shape;
  return relu(out, shape, name + "_relu");
}

inline Node v1_residual_block(Node input, Shape &shape, size_t out_channels,
                              const std::string &name) {
  Shape b1_shape = shape;
  Node b1 = avgpool2d(input, b1_shape, 3, 1, 1, name + "_b1_avg_pool");
  b1 = conv2d(b1, b1_shape, out_channels, 1, 1, 0, false, name + "_b1_conv");
  b1 = batchnorm(b1, b1_shape, true, name + "_b1_bn");
  b1 = maxpool2d(b1, b1_shape, 2, 2, 0, name + "_b1_pool");

  Shape b2_shape = shape;
  Node b2 = conv2d(input, b2_shape, out_channels, 3, 1, 1, false, name + "_b2_conv1");
  b2 = batchnorm(b2, b2_shape, true, name + "_b2_bn1");
  b2 = conv2d(b2, b2_shape, out_channels, 3, 1, 1, false, name + "_b2_conv2");
  b2 = batchnorm(b2, b2_shape, true, name + "_b2_bn2");
  b2 = maxpool2d(b2, b2_shape, 2, 2, 0, name + "_b2_pool");

  Shape b3_shape = shape;
  Node b3 = conv2d(input, b3_shape, out_channels, 3, 1, 1, false, name + "_b3_conv1");
  b3 = batchnorm(b3, b3_shape, true, name + "_b3_bn1");
  b3 = conv2d(b3, b3_shape, out_channels, 5, 1, 2, false, name + "_b3_conv2");
  b3 = batchnorm(b3, b3_shape, true, name + "_b3_bn2");
  b3 = maxpool2d(b3, b3_shape, 2, 2, 0, name + "_b3_pool");

  Shape b4_shape = shape;
  Node b4 = conv2d(input, b4_shape, out_channels * 2, 3, 1, 1, false, name + "_b4_conv_1");
  b4 = batchnorm(b4, b4_shape, true, name + "_b4_bn1");

  Shape b4_main_shape = b4_shape;
  Node b4_main =
      conv2d(b4, b4_main_shape, out_channels * 4, 3, 1, 1, false, name + "_b4_conv_2_main");

  Shape b4_shortcut_shape = b4_shape;
  Node b4_shortcut =
      conv2d(b4, b4_shortcut_shape, out_channels * 4, 3, 1, 1, false, name + "_b4_conv2_shortcut");

  b4 = b4_main + b4_shortcut;
  b4_shape = b4_main_shape;
  b4 = maxpool2d(b4, b4_shape, 2, 2, 0, name + "_b4_pool");

  auto [out, out_shape] =
      concat({b1, b2, b3, b4}, {b1_shape, b2_shape, b3_shape, b4_shape}, 3, name + "_concat");
  shape = out_shape;
  return relu(out, shape, name + "_relu");
}

inline std::pair<Node, Shape> v2_expand_reduce_branch(Node input, const Shape &input_shape,
                                                      size_t out_channels, size_t scale,
                                                      const std::string &name) {
  if (scale < 2 || scale % 2 != 0) {
    throw std::invalid_argument("V2 expansion scale must be an even integer >= 2");
  }

  Shape shape = input_shape;
  Node x = convtranspose2d(input, shape, out_channels, scale, scale, 0, false, name + "_expand");

  if (scale > 2) {
    const size_t reduction = scale / 2;
    x = avgpool2d(x, shape, reduction, reduction, 0, name + "_reduce");
  }

  return {x, shape};
}

inline Node v2_nested_group(Node input, Shape &shape, size_t out_channels, size_t scale1,
                            size_t scale2, size_t scale3, const std::string &name) {
  const Shape input_shape = shape;

  auto [b1, b1_shape] =
      v2_expand_reduce_branch(input, input_shape, out_channels, scale1, name + "_b1");

  auto [b2, b2_shape] =
      v2_expand_reduce_branch(input, input_shape, out_channels, scale2, name + "_b2");

  auto [b3, b3_shape] =
      v2_expand_reduce_branch(input, input_shape, out_channels, scale3, name + "_b3");

  auto [output, output_shape] =
      concat({b1, b2, b3}, {b1_shape, b2_shape, b3_shape}, 3, name + "_concat");

  shape = output_shape;
  return output;
}

}  // namespace tunx
