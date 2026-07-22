#pragma once

#include "nn/layer.hpp"
#include "nn/param.hpp"

namespace tunx {
class Block : public LayerImpl {
public:
  Block(const std::string &name = "block")
      : LayerImpl(name) {}

  virtual const Vec<Layer> layers() const = 0;

  Vec<Layer> layers() {
    const Vec<Layer> &const_layers = static_cast<const Block &>(*this).layers();
    return const_cast<Vec<Layer> &>(const_layers);
  }

  Vec<Param> params() override {
    Vec<Param> params;
    for (Layer &layer : this->layers()) {
      auto layer_params = layer.params();
      params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
  }

  const Vec<Param> params() const override {
    Vec<Param> params;
    for (const Layer &layer : this->layers()) {
      const auto layer_params = layer.params();
      params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
  }

protected:
  void init_impl() override {
    Vec<Layer> layers = this->layers();
    InitOptions opts{
        .ws_allocator = ws_allocator_,
        .engine = engine_,
        .handle = engine_handle_,
        .seed = srand_seed_,
        .io_dtype = io_dtype_,
        .param_dtype = param_dtype_,
        .compute_dtype = compute_dtype_,
    };

    for (Layer &layer : layers) {
      layer.init(*param_allocator_, opts);
    }
  }

  void on_set_training(bool training) override {
    Vec<Layer> layers = this->layers();
    for (Layer &layer : layers) {
      layer.set_training(training);
    }
  }
};
}  // namespace tunx