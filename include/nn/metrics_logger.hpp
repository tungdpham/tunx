#pragma once

#include "nn/train_config.hpp"
#include "nn/metrics_computer.hpp"
#include "common/csv_logger.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace tunx {

class MetricsLogger {
public:
    MetricsLogger(const std::string& name, const std::string& log_dir, const LogMode& log_mode) 
        : log_mode_(log_mode) {
        
        std::vector<std::string> step_headers = {"epoch", "step"};
        if (log_mode_.log_loss) step_headers.push_back("loss");
        if (log_mode_.log_accuracy) step_headers.push_back("accuracy_pct");
        if (log_mode_.log_precision) step_headers.push_back("precision");
        if (log_mode_.log_recall) step_headers.push_back("recall");
        if (log_mode_.log_f1_score) step_headers.push_back("f1_score");
        if (log_mode_.log_perplexity) step_headers.push_back("perplexity");
        if (log_mode_.log_top_k_accuracy) step_headers.push_back("top_k_accuracy");
        if (log_mode_.log_mae) step_headers.push_back("mae");
        if (log_mode_.log_mse) step_headers.push_back("mse");
        if (log_mode_.log_rmse) step_headers.push_back("rmse");
        step_headers.push_back("time_ms");

        std::string base_path = log_dir.empty() ? "" : log_dir + "/";

        train_step_logger_ = std::make_unique<CsvLogger>(
            name + "_train_step", base_path + name + "_train_step.csv", step_headers);
        val_step_logger_ = std::make_unique<CsvLogger>(
            name + "_val_step", base_path + name + "_val_step.csv", step_headers);

        std::vector<std::string> epoch_headers = {"epoch"};
        if (log_mode_.log_loss) {
            epoch_headers.push_back("train_loss");
            epoch_headers.push_back("val_loss");
        }
        if (log_mode_.log_accuracy) {
            epoch_headers.push_back("train_accuracy_pct");
            epoch_headers.push_back("val_accuracy_pct");
        }
        
        epoch_logger_ = std::make_unique<CsvLogger>(
            name + "_epoch", base_path + name + "_epoch.csv", epoch_headers);
    }
    
    void log_train_step(int epoch, int step, const ComputedMetrics& metrics) {
        train_step_logger_->log(to_string_map(epoch, step, metrics));
        train_step_logger_->flush();
    }
    
    void log_val_step(int epoch, int step, const ComputedMetrics& metrics) {
        val_step_logger_->log(to_string_map(epoch, step, metrics));
        val_step_logger_->flush();
    }
    
    void log_epoch(int epoch, const std::unordered_map<std::string, double>& metrics) {
        std::unordered_map<std::string, std::string> row;
        row["epoch"] = std::to_string(epoch);
        for (const auto& [k, v] : metrics) {
            row[k] = std::to_string(v);
        }
        epoch_logger_->log(row);
        epoch_logger_->flush();
    }

private:
    LogMode log_mode_;
    std::unique_ptr<CsvLogger> train_step_logger_;
    std::unique_ptr<CsvLogger> val_step_logger_;
    std::unique_ptr<CsvLogger> epoch_logger_;
    
    std::unordered_map<std::string, std::string> to_string_map(int epoch, int step, const ComputedMetrics& metrics) {
        std::unordered_map<std::string, std::string> row;
        row["epoch"] = std::to_string(epoch);
        row["step"] = std::to_string(step);
        
        if (metrics.loss) row["loss"] = std::to_string(*metrics.loss);
        if (metrics.accuracy_pct) row["accuracy_pct"] = std::to_string(*metrics.accuracy_pct);
        if (metrics.precision) row["precision"] = std::to_string(*metrics.precision);
        if (metrics.recall) row["recall"] = std::to_string(*metrics.recall);
        if (metrics.f1_score) row["f1_score"] = std::to_string(*metrics.f1_score);
        if (metrics.perplexity) row["perplexity"] = std::to_string(*metrics.perplexity);
        if (metrics.top_k_accuracy) row["top_k_accuracy"] = std::to_string(*metrics.top_k_accuracy);
        if (metrics.mae) row["mae"] = std::to_string(*metrics.mae);
        if (metrics.mse) row["mse"] = std::to_string(*metrics.mse);
        if (metrics.rmse) row["rmse"] = std::to_string(*metrics.rmse);
        if (metrics.time_ms) row["time_ms"] = std::to_string(*metrics.time_ms);
        
        return row;
    }
};

} // namespace tunx
