#include "nn/optimizers.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "device/pool_allocator.hpp"
#include "device/task.hpp"
#include "nn/optimizers_impl/cpu/adam_kernels.hpp"
#include "nn/optimizers_impl/cpu/sgd_kernels.hpp"
#ifdef TUNX_USE_CUDA
#include "nn/optimizers_impl/cuda/adam_kernels.hpp"
#include "nn/optimizers_impl/cuda/sgd_kernels.hpp"
#endif
#include "tensor/ops.hpp"

namespace tunx {

void Optimizer::attach(Graph &graph) {
  graph_ = &graph;
  auto all_params = graph.params();
  for (const auto &p : all_params) {
    if (p.requires_grad()) {
      params_.push_back(p);
    }
  }
  on_attach();
  std::cout << "Optimizer attached to " << params_.size() << " parameters" << std::endl;
}

void Optimizer::update() {
  if (!graph_) {
    throw std::runtime_error("Optimizer not attached to any graph or graph has been destroyed");
  }
  if (!graph_->engine()) {
    throw std::runtime_error("Graph not compiled");
  }
  update_impl(graph_->handle().get_stream());
}

void Optimizer::zero_grads() {
  if (!graph_) {
    throw std::runtime_error("Optimizer not attached to any graph or graph has been destroyed");
  }
  for (Param &param : params_) {
    param.zero_grad();
  }
}

void SGD::on_attach() {
  if (momentum_ > 0.0f) {
    velocities_.resize(params_.size());
    for (size_t i = 0; i < params_.size(); ++i) {
      velocities_[i] = Tensor(params_[i].shape(), params_[i].dtype(),
                              PoolAllocator::instance(params_[i].device(), nullptr));
      fill(velocities_[i], 0.0f);
    }
  }
}

template <typename T>
void SGD::update_impl_T(Tensor &param, const Tensor &grad, Tensor &velocity, stream s) {
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

void SGD::update_impl(stream s) {
  auto &params = this->params_;
  for (size_t i = 0; i < params.size(); ++i) {
    DISPATCH_DTYPE(params[i].dtype(), T,
                   update_impl_T<T>(params[i].data(), params[i].grad(), velocities_[i], s));
  }
}

OptimizerConfig SGD::get_config() const {
  OptimizerConfig config;
  config.type = "sgd";
  config.name = "SGD";
  config.set("learning_rate", this->learning_rate_);
  config.set("momentum", momentum_);
  return config;
}

void Adam::on_attach() {
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

template <typename T>
void Adam::update_impl_T(Tensor &param, const Tensor &grad, Tensor &m, Tensor &v,
                         float bias_correction1, float bias_correction2, stream s) {
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

void Adam::update_impl(stream s) {
  t_++;

  // Precompute bias correction terms outside the loop
  const float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
  const float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));

  for (size_t i = 0; i < params_.size(); ++i) {
    DISPATCH_DTYPE(params_[i].dtype(), T,
                   update_impl_T<T>(params_[i].data(), params_[i].grad(), m_[i], v_[i],
                                    bias_correction1, bias_correction2, s));
  }
}

OptimizerConfig Adam::get_config() const {
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

std::unique_ptr<Optimizer> OptimizerFactory::create_from_config(const OptimizerConfig &config) {
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
    bool decouple_weight_decay = config.get<bool>("decouple_weight_decay", config.type == "adamw");
    return std::make_unique<Adam>(learning_rate, beta1, beta2, epsilon, weight_decay,
                                  decouple_weight_decay);
  }
  throw std::invalid_argument("Unknown optimizer type: " + config.type);
}

std::unique_ptr<Optimizer> OptimizerFactory::create_sgd(float learning_rate, float momentum) {
  return std::make_unique<SGD>(learning_rate, momentum);
}

std::unique_ptr<Optimizer> OptimizerFactory::create_adam(float learning_rate, float beta1,
                                                         float beta2, float epsilon,
                                                         float weight_decay,
                                                         bool decouple_weight_decay) {
  return std::make_unique<Adam>(learning_rate, beta1, beta2, epsilon, weight_decay,
                                decouple_weight_decay);
}

}  // namespace tunx
