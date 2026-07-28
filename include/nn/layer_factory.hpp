#pragma once

#include "blocks_impl/flash_attention_block.hpp"
#include "blocks_impl/residual_block.hpp"
#include "layers_impl/activation.hpp"
#include "layers_impl/avgpool2d.hpp"
#include "layers_impl/batchnorm.hpp"
#include "layers_impl/class_token.hpp"
#include "layers_impl/conv2d.hpp"
#include "layers_impl/dense.hpp"
#include "layers_impl/dropout.hpp"
#include "layers_impl/elu.hpp"
#include "layers_impl/embedding.hpp"
#include "layers_impl/flatten.hpp"
#include "layers_impl/gelu.hpp"
#include "layers_impl/layer_norm.hpp"
#include "layers_impl/leaky_relu.hpp"
#include "layers_impl/linear.hpp"
#include "layers_impl/maxpool2d.hpp"
#include "layers_impl/positional_embedding.hpp"
#include "layers_impl/relu.hpp"
#include "layers_impl/sdpa.hpp"
#include "layers_impl/sigmoid.hpp"
#include "layers_impl/slice.hpp"
#include "layers_impl/tanh.hpp"
#include "layers_impl/transpose.hpp"
#include "nn/blocks_impl/flash_attention_block.hpp"
#include "nn/blocks_impl/sequential.hpp"
#include "nn/graph.hpp"  // IWYU pragma: export
#include "nn/layer.hpp"
#include "nn/layers_impl/add.hpp"
#include "nn/layers_impl/div.hpp"
#include "nn/layers_impl/mul.hpp"
#include "nn/layers_impl/sub.hpp"
#include "nn/param.hpp"
#include "tensor/ops.hpp"

namespace tunx {
inline Node operator+(const Node &a, const Node &b) {
  Add add_layer;
  return add_layer(a, b);
}
inline Node operator-(const Node &a, const Node &b) {
  Sub sub_layer;
  return sub_layer(a, b);
}
inline Node operator*(const Node &a, const Node &b) {
  Mul mul_layer;
  return mul_layer(a, b);
}
inline Node operator/(const Node &a, const Node &b) {
  Div div_layer;
  return div_layer(a, b);
}
}  // namespace tunx

namespace tunx {

// Concept to ensure LayerType has TYPE_NAME and create_from_config
template <typename T>
concept HasLayerTypeName = requires {
  { T::TYPE_NAME } -> std::convertible_to<const char *>;
  { T::create_from_config(std::declval<const LayerConfig &>()) } -> std::convertible_to<Layer>;
};

class LayerFactory {
private:
  static std::unordered_map<std::string, std::function<Layer(const LayerConfig &)>> creators_;

public:
  static void register_layer(const std::string &type,
                             std::function<Layer(const LayerConfig &)> creator) {
    creators_[type] = creator;
  }

  template <HasLayerTypeName LayerType>
  static void register_layer_type() {
    register_layer(LayerType::TYPE_NAME, [](const LayerConfig &config) -> Layer {
      return LayerType::create_from_config(config);
    });
  }

  static Layer create(const std::string &type, const LayerConfig &config) {
    auto it = creators_.find(type);
    if (it != creators_.end()) {
      return it->second(config);
    }
    throw std::invalid_argument("Unknown layer type: " + type);
  }

  static Layer create(const LayerConfig &config) { return create(config.type, config); }

  static void register_defaults() {
    register_layer_type<Dense>();
    register_layer_type<Activation>();
    register_layer_type<ReLU>();
    register_layer_type<ELU>();
    register_layer_type<GELU>();
    register_layer_type<LeakyReLU>();
    register_layer_type<Linear>();
    register_layer_type<Sigmoid>();
    register_layer_type<Tanh>();
    register_layer_type<Conv2D>();
    register_layer_type<MaxPool2D>();
    register_layer_type<AvgPool2D>();
    register_layer_type<BatchNorm>();
    register_layer_type<Dropout>();
    register_layer_type<LayerNorm>();
    register_layer_type<Flatten>();
    register_layer_type<ClassToken>();
    register_layer_type<PositionalEmbedding>();
    register_layer_type<Slice>();
    register_layer_type<Embedding>();
    register_layer_type<SDPA>();
    register_layer_type<ResidualBlock>();
    register_layer_type<Sequential>();
    register_layer_type<Transpose>();

    register_layer_type<Add>();
    register_layer_type<Sub>();
    register_layer_type<Mul>();
    register_layer_type<Div>();
  }

  static Vec<std::string> available_types() {
    Vec<std::string> types;
    for (const auto &pair : creators_) {
      types.push_back(pair.first);
    }
    return types;
  }
};

template <typename LayerType>
std::unique_ptr<LayerType> load_config(std::istream &file) {
  size_t j_size;
  file.read(reinterpret_cast<char *>(&j_size), sizeof(size_t));
  std::string j_str(j_size, '\0');
  file.read(&j_str[0], j_size);
  nlohmann::json j = nlohmann::json::parse(j_str);
  LayerConfig config = LayerConfig::from_json(j);
  LayerFactory::register_defaults();
  Layer base_layer = LayerFactory::create(config);
  LayerType *raw_ptr = dynamic_cast<LayerType *>(base_layer.release());
  if (!raw_ptr) {
    throw std::runtime_error("Failed to cast layer to requested type");
  }
  std::unique_ptr<LayerType> layer(raw_ptr);
  return layer;
}

inline void load_params(std::istream &in, LayerImpl &layer) {
  Vec<Param> params = layer.params();
  for (auto &param : params) {
    load(param.data(), in);
  }
}

inline std::unordered_map<std::string, std::function<Layer(const LayerConfig &)>>
    LayerFactory::creators_;

}  // namespace tunx
