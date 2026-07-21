/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cmath>
#include <cstddef>
#include <device/stream.hpp>
#include <iostream>
#include <memory>
#include <string>

#include "common/config.hpp"
#include "device/pool_allocator.hpp"
#include "device/task.hpp"
#include "nn/graph.hpp"
#include "optimizers_impl/cpu/adam_kernels.hpp"
#include "optimizers_impl/cpu/sgd_kernels.hpp"
#ifdef TUNX_USE_CUDA
#include "optimizers_impl/cuda/adam_kernels.hpp"
#include "optimizers_impl/cuda/sgd_kernels.hpp"
#endif
#include "tensor/tensor.hpp"
#include "tensor/tensor_ops.hpp"

namespace tunx {

using OptimizerConfig = TConfig;

class Optimizer {
public:
  explicit Optimizer(float learning_rate)
      : learning_rate_(learning_rate) {}
  virtual ~Optimizer() = default;

  void attach(Graph &graph) {
    graph_ = &graph;
    params_ = graph.params();
    on_attach();
    std::cout << "Optimizer attached to " << params_.size() << " parameters" << std::endl;
  }

  // update will dispatch the work on the graph's current stream
  void update() {
    if (!graph_) {
      throw std::runtime_error("Optimizer not attached to any graph or graph has been destroyed");
    }
    if (!graph_->engine()) {
      throw std::runtime_error("Graph not compiled");
    }
    update_impl(graph_->handle().get_stream());
  }

  // will dispatch the work on the graph's current stream
  void zero_grads() {
    if (!graph_) {
      throw std::runtime_error("Optimizer not attached to any graph or graph has been destroyed");
    }
    graph_->zero_grads();
  }

  void set_learning_rate(float lr) { learning_rate_ = lr; }
  float get_learning_rate() const { return learning_rate_; }

  virtual std::string name() const = 0;
  virtual OptimizerConfig get_config() const = 0;
  virtual std::unique_ptr<Optimizer> clone() const = 0;

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

  void update_impl(stream s) override {
    auto &params = this->params_;
    for (size_t i = 0; i < params.size(); ++i) {
      DISPATCH_DTYPE(params[i].dtype(), T,
                     update_impl<T>(params[i].data(), params[i].grad(), velocities_[i], s));
    }
  }

  std::string name() const override { return "SGD"; }

  OptimizerConfig get_config() const override {
    OptimizerConfig config;
    config.type = "sgd";
    config.name = "SGD";
    config.set("learning_rate", this->learning_rate_);
    config.set("momentum", momentum_);
    return config;
  }

  std::unique_ptr<Optimizer> clone() const override {
    return std::make_unique<SGD>(this->learning_rate_, momentum_);
  }

protected:
  void on_attach() override {
    if (momentum_ > 0.0f) {
      velocities_.resize(params_.size());
      for (size_t i = 0; i < params_.size(); ++i) {
        velocities_[i] = Tensor(params_[i].shape(), params_[i].dtype(),
                                PoolAllocator::instance(params_[i].device(), nullptr));
        fill(velocities_[i], 0.0f);
      }
    }
  }

private:
  float momentum_;
  Vec<Tensor> velocities_;

  template <typename T>
  void update_impl(Tensor &param, const Tensor &grad, Tensor &velocity, stream s) {
    auto &device = param.device();
    size_t size = param.size();

    if (param.device_type() == DeviceType::CPU) {
      if (momentum_ > 0.0f) {
        create_cpu_task(device, s, cpu::sgd::update_sgd_momentum<T>, param.data_as<T>(),
                        grad.data_as<T>(), velocity.data_as<T>(), size, this->learning_rate_,
                        momentum_);
      } else {
        create_cpu_task(device, s, cpu::sgd::update_sgd<T>, param.data_as<T>(), grad.data_as<T>(),
                        size, this->learning_rate_);
      }
    }
#ifdef TUNX_USE_CUDA
    else if (param.device_type() == DeviceType::CUDA) {
      if (momentum_ > 0.0f) {
        create_cuda_task(device, s, cuda::sgd::update_sgd_momentum<T>, param.data_as<T>(),
                         grad.data_as<T>(), velocity.data_as<T>(), size, this->learning_rate_,
                         momentum_);
      } else {
        create_cuda_task(device, s, cuda::sgd::update_sgd<T>, param.data_as<T>(), grad.data_as<T>(),
                         size, this->learning_rate_);
      }
    }
#endif
    else {
      throw std::runtime_error("Unsupported device type for SGD optimizer");
    }
  }
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

  void update_impl(stream s) override {
    t_++;

    // Precompute bias correction terms outside the loop
    const float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
    const float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));

    for (size_t i = 0; i < params_.size(); ++i) {
      DISPATCH_DTYPE(params_[i].dtype(), T,
                     update_impl<T>(params_[i].data(), params_[i].grad(), m_[i], v_[i],
                                    bias_correction1, bias_correction2, s));
    }
  }

  std::string name() const override { return decouple_weight_decay_ ? "AdamW" : "Adam"; }

  OptimizerConfig get_config() const override {
    OptimizerConfig config;
    config.type = decouple_weight_decay_ ? "adamw" : "adam";
    config.name = decouple_weight_decay_ ? "AdamW" : "Adam";
    config.set("learning_rate", this->learning_rate_);
    config.set("beta1", beta1_);
    config.set("beta2", beta2_);
    config.set("epsilon", epsilon_);
    config.set("weight_decay", weight_decay_);
    config.set("decouple_weight_decay", decouple_weight_decay_);
    return config;
  }

  std::unique_ptr<Optimizer> clone() const override {
    return std::make_unique<Adam>(this->learning_rate_, beta1_, beta2_, epsilon_, weight_decay_,
                                  decouple_weight_decay_);
  }

protected:
  void on_attach() override {
    m_.resize(params_.size());
    v_.resize(params_.size());
    for (size_t i = 0; i < params_.size(); ++i) {
      m_[i] = Tensor(params_[i].shape(), params_[i].dtype(),
                     PoolAllocator::instance(params_[i].device(), nullptr));
      fill(m_[i], 0.0f);
      v_[i] = Tensor(params_[i].shape(), params_[i].dtype(),
                     PoolAllocator::instance(params_[i].device(), nullptr));
      fill(v_[i], 0.0f);
    }
    t_ = 0;
  }

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
  void update_impl(Tensor &param, const Tensor &grad, Tensor &m, Tensor &v, float bias_correction1,
                   float bias_correction2, stream s) {
    auto &device = param.device();
    size_t size = param.size();
    if (param.device_type() == DeviceType::CPU) {
      create_cpu_task(device, s, cpu::adam::update_adam<T>, param.data_as<T>(), grad.data_as<T>(),
                      m.data_as<T>(), v.data_as<T>(), size, this->learning_rate_, beta1_, beta2_,
                      epsilon_, bias_correction1, bias_correction2, weight_decay_,
                      decouple_weight_decay_);
    }
#ifdef TUNX_USE_CUDA
    else if (param.device_type() == DeviceType::CUDA) {
      create_cuda_task(device, s, cuda::adam::update_adam<T>, param.data_as<T>(), grad.data_as<T>(),
                       m.data_as<T>(), v.data_as<T>(), size, this->learning_rate_, beta1_, beta2_,
                       epsilon_, bias_correction1, bias_correction2, weight_decay_,
                       decouple_weight_decay_);
    }
#endif
    else {
      throw std::runtime_error("Unsupported device type for Adam optimizer");
    }
  }
};

class OptimizerFactory {
public:
  static std::unique_ptr<Optimizer> create_from_config(const OptimizerConfig &config) {
    if (config.type == "sgd") {
      float learning_rate = config.get<float>("learning_rate", 0.01f);
      float momentum = config.get<float>("momentum", 0.0f);
      return std::make_unique<SGD>(learning_rate, momentum);
    }
    if (config.type == "adam" || config.type == "adamw") {
      float learning_rate = config.get<float>("learning_rate", 0.001f);
      float beta1 = config.get<float>("beta1", 0.9f);
      float beta2 = config.get<float>("beta2", 0.999f);
      float epsilon = config.get<float>("epsilon", 1e-8f);
      float weight_decay = config.get<float>("weight_decay", 0.0f);
      bool decouple_weight_decay =
          config.get<bool>("decouple_weight_decay", config.type == "adamw");
      return std::make_unique<Adam>(learning_rate, beta1, beta2, epsilon, weight_decay,
                                    decouple_weight_decay);
    }
    throw std::invalid_argument("Unknown optimizer type: " + config.type);
  }

  static std::unique_ptr<Optimizer> create_sgd(float learning_rate = 0.01f, float momentum = 0.0f) {
    return std::make_unique<SGD>(learning_rate, momentum);
  }

  static std::unique_ptr<Optimizer> create_adam(float learning_rate = 0.001f, float beta1 = 0.9f,
                                                float beta2 = 0.999f, float epsilon = 1e-8f,
                                                float weight_decay = 0.0f,
                                                bool decouple_weight_decay = false) {
    return std::make_unique<Adam>(learning_rate, beta1, beta2, epsilon, weight_decay,
                                  decouple_weight_decay);
  }
};

}  // namespace tunx
