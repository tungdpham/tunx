/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "common/config.hpp"
#include "device/stream.hpp"
#include "device/task.hpp"
#include "loss_impl/cpu/loss_ops.hpp"
#include "tensor/tensor.hpp"
#ifdef TUNX_USE_CUDA
#include "loss_impl/cuda/loss_ops.hpp"
#endif
#include <memory>
#include <stdexcept>
#include <string>

namespace tunx {

using LossConfig = TConfig;

class Loss {
public:
  virtual ~Loss() = default;

  void compute_loss(const Tensor &predictions, const Tensor &targets, float &loss,
                    stream s = nullptr) {
    if (!predictions || !targets) {
      throw std::runtime_error("Predictions and targets cannot be null for compute_loss.");
    }
    if (predictions.device() != targets.device()) {
      throw std::runtime_error(
          "Predictions and targets must be on the same device for compute_loss.");
    }
    return compute_loss_impl(predictions, targets, loss, s);
  }
  void compute_gradient(const Tensor &predictions, const Tensor &targets, Tensor &gradient,
                        stream s = nullptr) {
    if (!predictions || !targets || !gradient) {
      throw std::runtime_error(
          "Predictions, targets, and gradient cannot be null for compute_gradient.");
    }
    if (predictions.device() != targets.device() || predictions.device() != gradient.device()) {
      throw std::runtime_error(
          "Predictions, targets, and gradient must be on the same device for compute_gradient.");
    }
    return compute_gradient_impl(predictions, targets, gradient, s);
  }

  virtual std::string name() const = 0;
  virtual LossConfig get_config() const = 0;
  virtual std::unique_ptr<Loss> clone() const = 0;

  virtual size_t num_parameters() const { return 0; }

  virtual void reset() {}

protected:
  virtual void compute_loss_impl(const Tensor &predictions, const Tensor &targets, float &loss,
                                 stream s) = 0;
  virtual void compute_gradient_impl(const Tensor &predictions, const Tensor &targets,
                                     Tensor &gradient, stream s) = 0;
};

class CrossEntropyLoss : public Loss {
public:
  explicit CrossEntropyLoss(bool use_logits = true, double epsilon = 1e-15)
      : use_logits_(use_logits),
        epsilon_(epsilon) {}

  std::string name() const override { return "CrossEntropyLoss"; }

  LossConfig get_config() const override {
    LossConfig config;
    config.type = "cross_entropy";
    config.name = "CrossEntropyLoss";
    config.set("use_logits", use_logits_);
    config.set("epsilon", epsilon_);
    return config;
  }

  std::unique_ptr<Loss> clone() const override {
    return std::make_unique<CrossEntropyLoss>(use_logits_, epsilon_);
  }

  bool uses_logits() const { return use_logits_; }
  double get_epsilon() const { return epsilon_; }

private:
  bool use_logits_;  // If true, expects logits; if false, expects probabilities
  double epsilon_;

  void compute_loss_impl(const Tensor &predictions, const Tensor &targets, float &loss,
                         stream s) override {
    auto &device = predictions.device();
    size_t num_classes = predictions.shape().back();
    size_t batch_size = 1;
    for (size_t i = 0; i < predictions.dims() - 1; ++i) {
      batch_size *= predictions.shape()[i];
    }

    if (use_logits_) {
      // Use numerically stable logits version
      if (predictions.device_type() == DeviceType::CPU) {
        return create_cpu_task(device, s, cpu::loss::compute_cross_entropy_loss_logits,
                               predictions.dtype(), predictions.data_as<void>(),
                               targets.data_as<int>(), loss, batch_size, num_classes);
      }
#ifdef TUNX_USE_CUDA
      else if (predictions.device_type() == DeviceType::CUDA) {
        return create_cuda_task(device, s, cuda::loss::compute_cross_entropy_loss_logits,
                                predictions.dtype(), predictions.data_as(), targets.data_as<int>(),
                                loss, batch_size, num_classes);
      }
#endif
    } else {
      // Use probabilities version
      if (predictions.device_type() == DeviceType::CPU) {
        return create_cpu_task(device, s, cpu::loss::compute_cross_entropy_loss_probs,
                               predictions.dtype(), predictions.data_as(), targets.data_as<int>(),
                               loss, batch_size, num_classes, epsilon_);
      }
#ifdef TUNX_USE_CUDA
      else if (predictions.device_type() == DeviceType::CUDA) {
        return create_cuda_task(device, s, cuda::loss::compute_cross_entropy_loss_probs,
                                predictions.dtype(), predictions.data_as(), targets.data_as<int>(),
                                loss, batch_size, num_classes, epsilon_);
      }
#endif
    }
    throw std::runtime_error("Unsupported device type for CrossEntropyLoss.");
  }

  void compute_gradient_impl(const Tensor &predictions, const Tensor &targets, Tensor &gradient,
                             stream s) override {
    auto &device = predictions.device();
    size_t num_classes = predictions.shape().back();
    size_t batch_size = 1;
    for (size_t i = 0; i < predictions.dims() - 1; ++i) {
      batch_size *= predictions.shape()[i];
    }

    if (use_logits_) {
      // Use numerically stable logits version
      if (predictions.device_type() == DeviceType::CPU) {
        return create_cpu_task(device, s, cpu::loss::compute_cross_entropy_gradient_logits,
                               predictions.dtype(), predictions.data_as(), targets.data_as<int>(),
                               gradient.data_as(), batch_size, num_classes);
      }
#ifdef TUNX_USE_CUDA
      else if (predictions.device_type() == DeviceType::CUDA) {
        return create_cuda_task(device, s, cuda::loss::compute_cross_entropy_gradient_logits,
                                predictions.dtype(), predictions.data_as(), targets.data_as<int>(),
                                gradient.data_as(), batch_size, num_classes);
      }
#endif
    } else {
      // Use probabilities version
      if (predictions.device_type() == DeviceType::CPU) {
        return create_cpu_task(device, s, cpu::loss::compute_cross_entropy_gradient_probs,
                               predictions.dtype(), predictions.data_as(), targets.data_as<int>(),
                               gradient.data_as(), batch_size, num_classes, epsilon_);
      }
#ifdef TUNX_USE_CUDA
      else if (predictions.device_type() == DeviceType::CUDA) {
        return create_cuda_task(device, s, cuda::loss::compute_cross_entropy_gradient_probs,
                                predictions.dtype(), predictions.data_as(), targets.data_as<int>(),
                                gradient.data_as(), batch_size, num_classes, epsilon_);
      }
#endif
    }
    throw std::runtime_error("Unsupported device type for CrossEntropyLoss.");
  }
};

class MSELoss : public Loss {
public:
  MSELoss() = default;

  std::string name() const override { return "MSELoss"; }

  LossConfig get_config() const override {
    LossConfig config;
    config.type = "mse";
    config.name = "MSELoss";
    return config;
  }

  std::unique_ptr<Loss> clone() const override { return std::make_unique<MSELoss>(); }

private:
  void compute_loss_impl(const Tensor &predictions, const Tensor &targets, float &loss,
                         stream s) override {
    if (predictions.device() != targets.device()) {
      throw std::runtime_error("Predictions and targets must be on the same device for MSELoss.");
    }
    auto &device = predictions.device();
    size_t batch_size = predictions.shape()[0];
    size_t output_size = 1;
    for (size_t i = 1; i < predictions.dims(); ++i) {
      output_size *= predictions.shape()[i];
    }

    if (predictions.device_type() == DeviceType::CPU) {
      return create_cpu_task(device, s, cpu::loss::compute_mse_loss, predictions.dtype(),
                             predictions.data_as(), targets.data_as(), loss, batch_size,
                             output_size);
    }
#ifdef TUNX_USE_CUDA
    else if (predictions.device_type() == DeviceType::CUDA) {
      return create_cuda_task(device, s, cuda::loss::compute_mse_loss, predictions.dtype(),
                              predictions.data_as(), targets.data_as(), loss, batch_size,
                              output_size);
    }
#endif
    throw std::runtime_error("Unsupported device type for MSELoss.");
  }

  void compute_gradient_impl(const Tensor &predictions, const Tensor &targets, Tensor &gradient,
                             stream s) override {
    if (predictions.device() != targets.device() || predictions.device() != gradient.device()) {
      throw std::runtime_error(
          "Predictions, targets, and gradient must be on the same device for MSELoss.");
    }
    auto &device = predictions.device();
    gradient = Tensor(predictions.shape(), predictions.dtype(), predictions.device());
    size_t batch_size = predictions.shape()[0];
    size_t output_size = 1;
    for (size_t i = 1; i < predictions.dims(); ++i) {
      output_size *= predictions.shape()[i];
    }

    if (predictions.device_type() == DeviceType::CPU) {
      return create_cpu_task(device, s, cpu::loss::compute_mse_gradient, predictions.dtype(),
                             predictions.data_as(), targets.data_as(), gradient.data_as(),
                             batch_size, output_size);
    }
#ifdef TUNX_USE_CUDA
    else if (predictions.device_type() == DeviceType::CUDA) {
      return create_cuda_task(device, s, cuda::loss::compute_mse_gradient, predictions.dtype(),
                              predictions.data_as(), targets.data_as(), gradient.data_as(),
                              batch_size, output_size);
    }
#endif
    throw std::runtime_error("Unsupported device type for MSELoss.");
  }
};

class MAELoss : public Loss {
public:
  MAELoss() = default;

  std::string name() const override { return "MAELoss"; }

  LossConfig get_config() const override {
    LossConfig config;
    config.type = "mae";
    config.name = "MAELoss";
    return config;
  }

  std::unique_ptr<Loss> clone() const override { return std::make_unique<MAELoss>(); }

private:
  void compute_loss_impl(const Tensor &predictions, const Tensor &targets, float &loss,
                         stream s) override {
    if (predictions.device() != targets.device()) {
      throw std::runtime_error("Predictions and targets must be on the same device for MAELoss.");
    }
    auto &device = predictions.device();
    size_t batch_size = predictions.shape()[0];
    size_t output_size = 1;
    for (size_t i = 1; i < predictions.dims(); ++i) {
      output_size *= predictions.shape()[i];
    }

    if (predictions.device_type() == DeviceType::CPU) {
      return create_cpu_task(device, s, cpu::loss::compute_mae_loss, predictions.dtype(),
                             predictions.data_as(), targets.data_as(), loss, batch_size,
                             output_size);
    }
#ifdef TUNX_USE_CUDA
    else if (predictions.device_type() == DeviceType::CUDA) {
      return create_cuda_task(device, s, cuda::loss::compute_mae_loss, predictions.dtype(),
                              predictions.data_as(), targets.data_as(), loss, batch_size,
                              output_size);
    }
#endif
    throw std::runtime_error("Unsupported device type for MAELoss.");
  }

  void compute_gradient_impl(const Tensor &predictions, const Tensor &targets, Tensor &gradient,
                             stream s) override {
    if (predictions.device() != targets.device() || predictions.device() != gradient.device()) {
      throw std::runtime_error(
          "Predictions, targets, and gradient must be on the same device for MAELoss.");
    }
    auto &device = predictions.device();
    gradient = Tensor(predictions.shape(), predictions.dtype(), predictions.device());
    size_t batch_size = predictions.shape()[0];
    size_t output_size = 1;
    for (size_t i = 1; i < predictions.dims(); ++i) {
      output_size *= predictions.shape()[i];
    }

    if (predictions.device_type() == DeviceType::CPU) {
      return create_cpu_task(device, s, cpu::loss::compute_mae_gradient, predictions.dtype(),
                             predictions.data_as(), targets.data_as(), gradient.data_as(),
                             batch_size, output_size);
    }
#ifdef TUNX_USE_CUDA
    else if (predictions.device_type() == DeviceType::CUDA) {
      return create_cuda_task(device, s, cuda::loss::compute_mae_gradient, predictions.dtype(),
                              predictions.data_as(), targets.data_as(), gradient.data_as(),
                              batch_size, output_size);
    }
#endif
    throw std::runtime_error("Unsupported device type for MAELoss.");
  }
};

class HuberLoss : public Loss {
public:
  explicit HuberLoss(double delta = 1.0)
      : delta_(delta) {}

  std::string name() const override { return "HuberLoss"; }

  LossConfig get_config() const override {
    LossConfig config;
    config.type = "huber";
    config.name = "HuberLoss";
    config.set("delta", delta_);
    return config;
  }

  std::unique_ptr<Loss> clone() const override { return std::make_unique<HuberLoss>(delta_); }

  void set_delta(double delta) { delta_ = delta; }
  double get_delta() const { return delta_; }

private:
  double delta_;

  void compute_loss_impl(const Tensor &predictions, const Tensor &targets, float &loss,
                         stream s) override {
    if (predictions.device() != targets.device()) {
      throw std::runtime_error("Predictions and targets must be on the same device for HuberLoss.");
    }
    auto &device = predictions.device();
    size_t batch_size = predictions.shape()[0];
    size_t output_size = 1;
    for (size_t i = 1; i < predictions.dims(); ++i) {
      output_size *= predictions.shape()[i];
    }

    if (predictions.device_type() == DeviceType::CPU) {
      return create_cpu_task(device, s, cpu::loss::compute_huber_loss, predictions.dtype(),
                             predictions.data_as(), targets.data_as(), loss, batch_size,
                             output_size, delta_);
    }
#ifdef TUNX_USE_CUDA
    else if (predictions.device_type() == DeviceType::CUDA) {
      return create_cuda_task(device, s, cuda::loss::compute_huber_loss, predictions.dtype(),
                              predictions.data_as(), targets.data_as(), loss, batch_size,
                              output_size, delta_);
    }
#endif
    throw std::runtime_error("Unsupported device type for HuberLoss.");
  }

  void compute_gradient_impl(const Tensor &predictions, const Tensor &targets, Tensor &gradient,
                             stream s) override {
    if (predictions.device() != targets.device() || predictions.device() != gradient.device()) {
      throw std::runtime_error(
          "Predictions, targets, and gradient must be on the same device for HuberLoss.");
    }
    auto &device = predictions.device();
    gradient = Tensor(predictions.shape(), predictions.dtype(), predictions.device());
    size_t batch_size = predictions.shape()[0];
    size_t output_size = 1;
    for (size_t i = 1; i < predictions.dims(); ++i) {
      output_size *= predictions.shape()[i];
    }

    if (predictions.device_type() == DeviceType::CPU) {
      return create_cpu_task(device, s, cpu::loss::compute_huber_gradient, predictions.dtype(),
                             predictions.data_as(), targets.data_as(), gradient.data_as(),
                             batch_size, output_size, delta_);
    }
#ifdef TUNX_USE_CUDA
    else if (predictions.device_type() == DeviceType::CUDA) {
      return create_cuda_task(device, s, cuda::loss::compute_huber_gradient, predictions.dtype(),
                              predictions.data_as(), targets.data_as(), gradient.data_as(),
                              batch_size, output_size, delta_);
    }
#endif
    throw std::runtime_error("Unsupported device type for HuberLoss.");
  }
};

class LossFactory {
public:
  static std::unique_ptr<Loss> create(const std::string &loss_type) {
    if (loss_type == "cross_entropy" || loss_type == "ce") {
      return std::make_unique<CrossEntropyLoss>(true);  // Default to logits
    }
    // Backward compatibility: logsoftmax_cross_entropy -> CrossEntropyLoss with use_logits=true
    if (loss_type == "logsoftmax_cross_entropy" || loss_type == "logsoftmax_ce") {
      return std::make_unique<CrossEntropyLoss>(true);
    }
    if (loss_type == "mse" || loss_type == "mean_squared_error") {
      return std::make_unique<MSELoss>();
    }
    if (loss_type == "mae" || loss_type == "mean_absolute_error") {
      return std::make_unique<MAELoss>();
    }
    if (loss_type == "huber") {
      return std::make_unique<HuberLoss>();
    }
    throw std::invalid_argument("Unknown loss type: " + loss_type);
  }

  static std::unique_ptr<Loss> create_from_config(const LossConfig &config) {
    if (config.type == "cross_entropy" || config.type == "ce") {
      bool use_logits = config.get<bool>("use_logits", true);
      double epsilon = config.get<double>("epsilon", 1e-15);
      return std::make_unique<CrossEntropyLoss>(use_logits, epsilon);
    }
    // Backward compatibility: logsoftmax_cross_entropy -> CrossEntropyLoss with use_logits=true
    if (config.type == "logsoftmax_cross_entropy" || config.type == "logsoftmax_ce") {
      return std::make_unique<CrossEntropyLoss>(true);
    }
    if (config.type == "mse" || config.type == "mean_squared_error") {
      return std::make_unique<MSELoss>();
    }
    if (config.type == "mae" || config.type == "mean_absolute_error") {
      return std::make_unique<MAELoss>();
    }
    if (config.type == "huber") {
      double delta = config.get("delta", 1.0);
      return std::make_unique<HuberLoss>(delta);
    }
    throw std::invalid_argument("Unknown loss type: " + config.type);
  }

  static std::unique_ptr<Loss> create_cross_entropy(bool use_logits = true,
                                                    double epsilon = 1e-15) {
    return std::make_unique<CrossEntropyLoss>(use_logits, epsilon);
  }

  static std::unique_ptr<Loss> create_mse() { return std::make_unique<MSELoss>(); }

  static std::unique_ptr<Loss> create_mae() { return std::make_unique<MAELoss>(); }

  static std::unique_ptr<Loss> create_huber(double delta = 1.0) {
    return std::make_unique<HuberLoss>(delta);
  }
};

}  // namespace tunx
