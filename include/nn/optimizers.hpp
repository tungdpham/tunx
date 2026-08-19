#pragma once

#include <memory>
#include <string>

#include "common/config.hpp"
#include "device/stream.hpp"
#include "nn/graph.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

using OptimizerConfig = TConfig;

class Optimizer {
public:
  explicit Optimizer(float learning_rate)
      : learning_rate_(learning_rate) {}
  virtual ~Optimizer() = default;

  void attach(Graph &graph);

  // update will dispatch the work on the graph's current stream
  void update();

  // will dispatch the work on the graph's current stream
  void zero_grads();

  void set_lr(float lr) { learning_rate_ = lr; }
  float get_lr() const { return learning_rate_; }

  virtual std::string name() const = 0;
  virtual OptimizerConfig get_config() const = 0;
  virtual std::unique_ptr<Optimizer> clone() const = 0;
  virtual Vec<Tensor> states() const = 0;

protected:
  float learning_rate_;
  Graph *graph_;
  Vec<Param> params_;

  virtual void on_attach() {}

  virtual void update_impl(stream s) = 0;
};

class SGD : public Optimizer {
public:
  SGD(float learning_rate = 0.01f, float momentum = 0.0f)
      : Optimizer(learning_rate),
        momentum_(momentum) {}

  std::string name() const override { return "SGD"; }

  OptimizerConfig get_config() const override;

  std::unique_ptr<Optimizer> clone() const override {
    return std::make_unique<SGD>(this->learning_rate_, momentum_);
  }

  Vec<Tensor> states() const override { return velocities_; }

protected:
  void on_attach() override;
  void update_impl(stream s) override;

private:
  float momentum_;
  Vec<Tensor> velocities_;

  template <typename T>
  void update_impl_T(Tensor &param, const Tensor &grad, Tensor &velocity, stream s);
};

class Adam : public Optimizer {
public:
  Adam(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f,
       float epsilon = 1e-8f, float weight_decay = 0.0f, bool decouple_weight_decay = false)
      : Optimizer(learning_rate),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        weight_decay_(weight_decay),
        decouple_weight_decay_(decouple_weight_decay),
        t_(0) {}

  std::string name() const override { return decouple_weight_decay_ ? "AdamW" : "Adam"; }

  OptimizerConfig get_config() const override;

  std::unique_ptr<Optimizer> clone() const override {
    return std::make_unique<Adam>(this->learning_rate_, beta1_, beta2_, epsilon_, weight_decay_,
                                  decouple_weight_decay_);
  }

  Vec<Tensor> states() const override {
    Vec<Tensor> states = m_;
    states.insert(states.end(), v_.begin(), v_.end());
    return states;
  }

protected:
  void on_attach() override;
  void update_impl(stream s) override;

private:
  float beta1_;
  float beta2_;
  float epsilon_;
  float weight_decay_;
  bool decouple_weight_decay_;
  unsigned long t_;
  Vec<Tensor> m_;
  Vec<Tensor> v_;

  template <typename T>
  void update_impl_T(Tensor &param, const Tensor &grad, Tensor &m, Tensor &v,
                     float bias_correction1, float bias_correction2, stream s);
};

class OptimizerFactory {
public:
  static std::unique_ptr<Optimizer> create_from_config(const OptimizerConfig &config);
  static std::unique_ptr<Optimizer> create_sgd(float learning_rate = 0.01f, float momentum = 0.0f);
  static std::unique_ptr<Optimizer> create_adam(float learning_rate = 0.001f, float beta1 = 0.9f,
                                                float beta2 = 0.999f, float epsilon = 1e-8f,
                                                float weight_decay = 0.0f,
                                                bool decouple_weight_decay = false);
};

}  // namespace tunx
