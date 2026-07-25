#pragma once

#include <memory>
#include <string>

#include "common/config.hpp"
#include "device/stream.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

using LossConfig = TConfig;

class Loss {
public:
  virtual ~Loss() = default;

  void compute_loss(const Tensor &predictions, const Tensor &targets, float &loss,
                    stream s = nullptr);

  void compute_gradient(const Tensor &predictions, const Tensor &targets, Tensor &gradient,
                        stream s = nullptr);

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
  LossConfig get_config() const override;
  std::unique_ptr<Loss> clone() const override {
    return std::make_unique<CrossEntropyLoss>(use_logits_, epsilon_);
  }

  bool uses_logits() const { return use_logits_; }
  double get_epsilon() const { return epsilon_; }

private:
  bool use_logits_;  // If true, expects logits; if false, expects probabilities
  double epsilon_;

  void compute_loss_impl(const Tensor &predictions, const Tensor &targets, float &loss,
                         stream s) override;

  void compute_gradient_impl(const Tensor &predictions, const Tensor &targets, Tensor &gradient,
                             stream s) override;
};

class MSELoss : public Loss {
public:
  MSELoss() = default;

  std::string name() const override { return "MSELoss"; }
  LossConfig get_config() const override;
  std::unique_ptr<Loss> clone() const override { return std::make_unique<MSELoss>(); }

private:
  void compute_loss_impl(const Tensor &predictions, const Tensor &targets, float &loss,
                         stream s) override;

  void compute_gradient_impl(const Tensor &predictions, const Tensor &targets, Tensor &gradient,
                             stream s) override;
};

class MAELoss : public Loss {
public:
  MAELoss() = default;

  std::string name() const override { return "MAELoss"; }
  LossConfig get_config() const override;
  std::unique_ptr<Loss> clone() const override { return std::make_unique<MAELoss>(); }

private:
  void compute_loss_impl(const Tensor &predictions, const Tensor &targets, float &loss,
                         stream s) override;

  void compute_gradient_impl(const Tensor &predictions, const Tensor &targets, Tensor &gradient,
                             stream s) override;
};

class HuberLoss : public Loss {
public:
  explicit HuberLoss(double delta = 1.0)
      : delta_(delta) {}

  std::string name() const override { return "HuberLoss"; }
  LossConfig get_config() const override;
  std::unique_ptr<Loss> clone() const override { return std::make_unique<HuberLoss>(delta_); }

  void set_delta(double delta) { delta_ = delta; }
  double get_delta() const { return delta_; }

private:
  double delta_;

  void compute_loss_impl(const Tensor &predictions, const Tensor &targets, float &loss,
                         stream s) override;

  void compute_gradient_impl(const Tensor &predictions, const Tensor &targets, Tensor &gradient,
                             stream s) override;
};

class LossFactory {
public:
  static std::unique_ptr<Loss> create(const std::string &loss_type);
  static std::unique_ptr<Loss> create_from_config(const LossConfig &config);
  static std::unique_ptr<Loss> create_cross_entropy(bool use_logits = true, double epsilon = 1e-15);
  static std::unique_ptr<Loss> create_mse();
  static std::unique_ptr<Loss> create_mae();
  static std::unique_ptr<Loss> create_huber(double delta = 1.0);
};

}  // namespace tunx
