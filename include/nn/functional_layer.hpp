/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "device/iallocator.hpp"
#include "nn/graph.hpp"
#include "nn/layer.hpp"
#include "nn/param.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace tunx {

struct OpContext {
  LayerImpl* layer;
  Engine engine;
  engine_handle handle;
  Residuals residuals;
  bool is_training;
  DType_t io_dtype;
  DType_t param_dtype;
  DType_t compute_dtype;
  bool use_seed;
  unsigned long long srand_seed;
  IAllocator* ws_allocator;

  Tensor make_tensor(const Vec<size_t>& shape, DType_t dtype) const {
    return layer->make_tensor(shape, dtype);
  }

  Param make_param(const Vec<size_t>& shape, DType_t dtype = DType_t::UNKNOWN) const {
    if (dtype == DType_t::UNKNOWN) dtype = param_dtype;
    return layer->make_param(shape, dtype);
  }
};

template <typename T>
struct function_traits;

template <typename R, typename... Args>
struct function_traits<R (*)(Args...)> {
  static constexpr size_t arity = sizeof...(Args);
  using return_type = R;
  using args_tuple = std::tuple<Args...>;
};

template <typename Target, typename Tuple, size_t... Is>
constexpr size_t count_type_before_impl(std::index_sequence<Is...>) {
  return (0 + ... +
          (std::is_same_v<Target, std::decay_t<std::tuple_element_t<Is, Tuple>>> ? 1 : 0));
}

template <typename Target, typename Tuple, size_t N>
constexpr size_t count_type_before() {
  return count_type_before_impl<Target, Tuple>(std::make_index_sequence<N>{});
}

template <typename Op>
class FunctionalLayerImpl : public LayerImpl {
public:
  using ForwardSig = decltype(&Op::forward);
  using BackwardSig = decltype(&Op::backward);
  using ForwardTraits = function_traits<ForwardSig>;
  using BackwardTraits = function_traits<BackwardSig>;
  using Config = typename Op::Config;

  struct ParamDef {
    std::string name;
    Vec<size_t> shape;
    std::function<void(Param&, OpContext&)> init_fn;
    DType_t dtype;
  };

  Config config_;
  std::vector<ParamDef> param_defs_;

  FunctionalLayerImpl(Config config, const std::string& name = Op::TYPE_NAME)
      : LayerImpl(name),
        config_(std::move(config)) {}

  void register_param(const std::string& name, const Vec<size_t>& shape,
                      std::function<void(Param&, OpContext&)> init_fn,
                      DType_t dtype = DType_t::UNKNOWN) {
    param_defs_.push_back({name, shape, std::move(init_fn), dtype});
  }

  OpContext make_context(Residuals residuals = Residuals{}) {
    return OpContext{this,
                     this->engine_,
                     this->engine_handle_,
                     residuals,
                     this->is_training_,
                     this->io_dtype_,
                     this->param_dtype_,
                     this->compute_dtype_,
                     this->use_seed_,
                     this->srand_seed_,
                     this->ws_allocator_};
  }

  template <typename... Args,
            typename = std::enable_if_t<(std::is_same_v<std::decay_t<Args>, Node> && ...)>>
  Node operator()(Args&&... args) {
    std::vector<Node> inputs = {std::forward<Args>(args)...};
    return operator()(inputs);
  }

  Node operator()(const std::vector<Node>& inputs) {
    Graph* graph = nullptr;
    for (const auto& node : inputs) {
      if (!node)
        throw std::runtime_error(std::string(Op::TYPE_NAME) + ": input nodes cannot be null");
      if (!graph)
        graph = node->graph();
      else if (graph != node->graph()) {
        throw std::runtime_error(std::string(Op::TYPE_NAME) +
                                 ": all input nodes must belong to the same graph");
      }
    }

    if (!graph)
      throw std::runtime_error(std::string(Op::TYPE_NAME) + ": graph is null, cannot add node");

    Node output = graph->make_node();
    std::shared_ptr<LayerImpl> self = this->shared_from_this();
    graph->add_edge(self, inputs, {output});
    return output;
  }

protected:
  void init_impl() override {
    OpContext ctx = make_context();
    if constexpr (requires { Op::init(ctx, config_); }) {
      Op::init(ctx, config_);
    }
    for (const auto& def : param_defs_) {
      DType_t t = def.dtype == DType_t::UNKNOWN ? this->param_dtype_ : def.dtype;
      Param p = this->make_param(def.shape, t);
      if (def.init_fn) {
        def.init_fn(p, ctx);
      }
    }
  }

  template <typename T>
  struct always_false : std::false_type {};
  template <typename T>
  static constexpr bool always_false_v = always_false<T>::value;

  template <size_t I, typename TraitsType>
  decltype(auto) get_arg(const Vec<Tensor>& inputs, Residuals& residuals, OpContext& ctx) {
    using ArgType = std::tuple_element_t<I, typename TraitsType::args_tuple>;
    using BaseType = std::decay_t<ArgType>;

    if constexpr (std::is_same_v<BaseType, Tensor>) {
      constexpr size_t idx = count_type_before<Tensor, typename TraitsType::args_tuple, I>();
      if (idx >= inputs.size()) throw std::runtime_error("Not enough inputs provided to layer.");
      return inputs[idx];
    } else if constexpr (std::is_same_v<BaseType, Vec<Tensor>>) {
      return inputs;
    } else if constexpr (std::is_same_v<BaseType, Layer>) {
      constexpr size_t idx = count_type_before<Layer, typename TraitsType::args_tuple, I>();
      if (idx >= this->registered_layers_.size())
        throw std::runtime_error("Not enough layers registered in layer.");
      return Layer(this->registered_layers_[idx]);
    } else if constexpr (std::is_same_v<BaseType, Vec<Layer>>) {
      Vec<Layer> res;
      for (const auto& l : this->registered_layers_) {
        res.push_back(Layer(l));
      }
      return res;
    } else if constexpr (std::is_same_v<BaseType, Param>) {
      constexpr size_t idx = count_type_before<Param, typename TraitsType::args_tuple, I>();
      if (idx >= this->params_.size())
        throw std::runtime_error("Not enough params registered in layer.");
      return this->params_[idx];
    } else if constexpr (std::is_same_v<BaseType, Config>) {
      return config_;
    } else if constexpr (std::is_same_v<BaseType, OpContext>) {
      return ctx;
    } else {
      static_assert(always_false_v<BaseType>, "Unsupported argument type in Op signature");
    }
  }

  template <size_t... Is>
  Vec<Tensor> dispatch_forward(const Vec<Tensor>& inputs, Residuals& residuals, OpContext& ctx,
                               std::index_sequence<Is...>) {
    if constexpr (std::is_same_v<typename ForwardTraits::return_type, Vec<Tensor>>) {
      return Op::forward(get_arg<Is, ForwardTraits>(inputs, residuals, ctx)...);
    } else {
      return {Op::forward(get_arg<Is, ForwardTraits>(inputs, residuals, ctx)...)};
    }
  }

  template <size_t... Is>
  Vec<Tensor> dispatch_backward(const Vec<Tensor>& grad_outputs, Residuals& residuals,
                                OpContext& ctx, std::index_sequence<Is...>) {
    if constexpr (std::is_same_v<typename BackwardTraits::return_type, Vec<Tensor>>) {
      return Op::backward(get_arg<Is, BackwardTraits>(grad_outputs, residuals, ctx)...);
    } else {
      return {Op::backward(get_arg<Is, BackwardTraits>(grad_outputs, residuals, ctx)...)};
    }
  }

  Vec<Tensor> forward_impl(const Vec<Tensor>& inputs, Residuals& residuals) override {
    OpContext ctx = make_context(residuals);
    return dispatch_forward(inputs, residuals, ctx,
                            std::make_index_sequence<ForwardTraits::arity>{});
  }

  Vec<Tensor> backward_impl(const Vec<Tensor>& grad_outputs, Residuals& residuals) override {
    OpContext ctx = make_context(residuals);
    return dispatch_backward(grad_outputs, residuals, ctx,
                             std::make_index_sequence<BackwardTraits::arity>{});
  }

public:
  std::string type() const override { return Op::TYPE_NAME; }

  LayerConfig get_config() const override {
    if constexpr (requires { Op::get_config(config_, this->name_, this->registered_layers_); }) {
      return Op::get_config(config_, this->name_, this->registered_layers_);
    } else {
      return Op::get_config(config_, this->name_);
    }
  }

  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>>& input_shapes) const override {
    if constexpr (requires {
                    Op::output_shapes(input_shapes, config_, this->registered_layers_);
                  }) {
      return Op::output_shapes(input_shapes, config_, this->registered_layers_);
    } else {
      return Op::output_shapes(input_shapes, config_);
    }
  }
};

template <typename Op>
class FunctionalLayer : public LayerRef<FunctionalLayerImpl<Op>> {
public:
  template <typename C = typename Op::Config,
            typename = std::enable_if_t<std::is_default_constructible_v<C>>>
  FunctionalLayer(const std::string& name = Op::TYPE_NAME)
      : LayerRef<FunctionalLayerImpl<Op>>(C{}, name) {}

  template <typename C = typename Op::Config,
            typename = std::enable_if_t<std::is_default_constructible_v<C>>>
  FunctionalLayer(const char* name)
      : LayerRef<FunctionalLayerImpl<Op>>(C{}, std::string(name)) {}

  FunctionalLayer(typename Op::Config config, const std::string& name = Op::TYPE_NAME)
      : LayerRef<FunctionalLayerImpl<Op>>(std::move(config), name) {}

  FunctionalLayer(typename Op::Config config, const char* name)
      : LayerRef<FunctionalLayerImpl<Op>>(std::move(config), std::string(name)) {}

  static constexpr const char* TYPE_NAME = Op::TYPE_NAME;

  static Layer create_from_config(const LayerConfig& config) {
    return FunctionalLayer(Op::parse_config(config),
                           config.name.empty() ? Op::TYPE_NAME : config.name);
  }

  template <typename... Args,
            typename = std::enable_if_t<(std::is_same_v<std::decay_t<Args>, Node> && ...)>>
  Node operator()(Args&&... args) {
    return (*this->impl_)(std::forward<Args>(args)...);
  }

  Node operator()(const std::vector<Node>& inputs) { return (*this->impl_)(inputs); }
};

}  // namespace tunx
