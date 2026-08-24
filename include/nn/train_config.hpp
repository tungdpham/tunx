/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <string>

#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/schedulers.hpp"
#include "type/type.hpp"

namespace tunx {

enum class ProfilerType { NONE = 0, NORMAL = 1, CUMULATIVE = 2 };
enum class TrainingMode { CLASSIFICATION = 0, REGRESSION = 1, CUSTOM = 2 };

struct LogMode {
  bool log_loss = true;
  bool log_accuracy = true;
  bool log_precision = false;
  bool log_recall = false;
  bool log_f1_score = false;
  bool log_perplexity = false;
  bool log_top_k_accuracy = false;
  bool log_mae = false;
  bool log_mse = false;
  bool log_rmse = false;
};

struct TrainingConfig {
  // Trainer params
  int epochs = 10;
  size_t batch_size = 32;
  int64 max_steps = -1;  // -1 for no limit, otherwise max number of batches per epoch
  std::string train_mode = "auto";
  float lr_initial = 0.001f;
  int gradient_accumulation_steps = 1;
  int progress_print_interval = 100;
  int64 num_threads = 8;
  ProfilerType profiler_type = ProfilerType::NONE;
  bool print_layer_profiling = false;
  bool print_memory_usage = false;
  std::string model_name = "cifar10_resnet9";
  std::string model_path = "";
  std::string dataset_name = "";
  std::string dataset_path = "data";
  DeviceID device_id;
  DType_t io_dtype = DType_t::FP32;
  DType_t param_dtype = DType_t::FP32;
  DType_t compute_dtype = DType_t::FP32;
  std::string log_dir = "logs";

  bool prefetch_data = false;
  size_t prefetch_depth = 2;
  bool async_pipeline = true;
  bool augmentation = true;
  bool benchmark_mode = false;

  LogMode log_mode;

  size_t num_microbatches = 2;

  OptimizerConfig optimizer_config;
  SchedulerConfig scheduler_config;
  LossConfig loss_config;

  void print_config() const;
  void load_from_json(const std::string &config_path);
};

}  // namespace tunx
