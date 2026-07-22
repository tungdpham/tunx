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
#include "tensor/tensor_ops.hpp"

namespace tunx {
inline Node operator+(const Node &a, const Node &b) {
  auto add_layer = make_layer<internal::AddImpl>();
  return add_layer(a, b);
}
inline Node operator-(const Node &a, const Node &b) {
  auto sub_layer = make_layer<internal::SubImpl>();
  return sub_layer(a, b);
}
inline Node operator*(const Node &a, const Node &b) {
  auto mul_layer = make_layer<internal::MulImpl>();
  return mul_layer(a, b);
}
inline Node operator/(const Node &a, const Node &b) {
  auto div_layer = make_layer<internal::DivImpl>();
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
    register_layer_type<internal::DenseImpl>();
    register_layer_type<internal::ActivationImpl>();
    register_layer_type<internal::ReLUImpl>();
    register_layer_type<internal::ELUImpl>();
    register_layer_type<internal::GELUImpl>();
    register_layer_type<internal::LeakyReLUImpl>();
    register_layer_type<internal::LinearImpl>();
    register_layer_type<internal::SigmoidImpl>();
    register_layer_type<internal::TanhImpl>();
    register_layer_type<internal::Conv2DImpl>();
    register_layer_type<internal::MaxPool2DImpl>();
    register_layer_type<internal::AvgPool2DImpl>();
    register_layer_type<internal::BatchNormImpl>();
    register_layer_type<internal::DropoutImpl>();
    register_layer_type<internal::LayerNormImpl>();
    register_layer_type<internal::FlattenImpl>();
    register_layer_type<internal::ClassTokenImpl>();
    register_layer_type<internal::PositionalEmbeddingImpl>();
    register_layer_type<internal::SliceImpl>();
    register_layer_type<internal::EmbeddingImpl>();
    register_layer_type<internal::SDPAImpl>();
    register_layer_type<ResidualBlockImpl>();
    register_layer_type<FlashAttentionBlockImpl>();
    register_layer_type<internal::TransposeImpl>();
    register_layer_type<SequentialImpl>();

    register_layer_type<internal::AddImpl>();
    register_layer_type<internal::SubImpl>();
    register_layer_type<internal::MulImpl>();
    register_layer_type<internal::DivImpl>();
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
