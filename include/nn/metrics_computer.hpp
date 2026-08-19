#pragma once

#include "nn/train_config.hpp"
#include "tensor/tensor.hpp"
#include "nn/metrics.hpp"
#include <optional>
#include <cmath>

namespace tunx {

struct ComputedMetrics {
    std::optional<double> loss;
    std::optional<double> accuracy_pct;
    std::optional<double> precision;
    std::optional<double> recall;
    std::optional<double> f1_score;
    std::optional<double> perplexity;
    std::optional<double> top_k_accuracy;
    std::optional<double> mae;
    std::optional<double> mse;
    std::optional<double> rmse;
    std::optional<double> time_ms;
    
    int corrects = 0;
    size_t num_samples = 0;
};

class MetricsComputer {
public:
    MetricsComputer(const LogMode& log_mode) : log_mode_(log_mode) {}
    
    ComputedMetrics compute(const Tensor& predictions, const Tensor& labels, std::optional<double> loss = std::nullopt) const {
        ComputedMetrics metrics;
        metrics.loss = loss;
        
        if (log_mode_.log_accuracy || log_mode_.log_precision || log_mode_.log_recall || 
            log_mode_.log_f1_score || log_mode_.log_perplexity || log_mode_.log_top_k_accuracy ||
            log_mode_.log_mae || log_mode_.log_mse || log_mode_.log_rmse) {
            
            size_t batch_size = 1;
            for (size_t i = 0; i < predictions.dims() - 1; ++i) {
                batch_size *= predictions.shape()[i];
            }
            metrics.num_samples = batch_size;
        }

        if (log_mode_.log_accuracy) {
            metrics.corrects = compute_class_corrects(predictions, labels);
            if (metrics.num_samples > 0) {
                // compute accuracy as percentage matching the rest of the code
                metrics.accuracy_pct = (static_cast<double>(metrics.corrects) * 100.0) / metrics.num_samples;
            }
        }
        if (log_mode_.log_precision) {
            metrics.precision = compute_precision(predictions, labels);
        }
        if (log_mode_.log_recall) {
            metrics.recall = compute_recall(predictions, labels);
        }
        if (log_mode_.log_f1_score) {
            metrics.f1_score = compute_f1_score(predictions, labels);
        }
        if (log_mode_.log_perplexity) {
            if (loss.has_value()) {
                metrics.perplexity = std::exp(loss.value());
            } else {
                metrics.perplexity = compute_perplexity(predictions, labels);
            }
        }
        if (log_mode_.log_top_k_accuracy) {
            metrics.top_k_accuracy = compute_top_k_accuracy(predictions, labels, 5);
        }
        if (log_mode_.log_mae) {
            metrics.mae = compute_mae(predictions, labels);
        }
        if (log_mode_.log_mse) {
            metrics.mse = compute_mse(predictions, labels);
        }
        if (log_mode_.log_rmse) {
            metrics.rmse = compute_rmse(predictions, labels);
        }
        return metrics;
    }

private:
    LogMode log_mode_;
};

} // namespace tunx
