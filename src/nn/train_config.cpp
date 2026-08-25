#include "nn/train_config.hpp"

#include <fstream>
#include <string>
#include <vector>

namespace tunx {
static std::string normalize_train_mode(std::string mode) {
  for (char &c : mode) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  if (mode == "epoch" || mode == "batch" || mode == "auto") {
    return mode;
  }
  std::cerr << "Warning: invalid TRAIN_MODE/tunx_TRAIN_MODE=\"" << mode
            << "\". Expected epoch, batch, or auto. Falling back to auto." << std::endl;
  return "auto";
}

static void parse_optimizer_json(const nlohmann::json &j, OptimizerConfig &cfg) {
  cfg.type = j.value("type", cfg.type.empty() ? "adam" : cfg.type);
  for (auto &[k, v] : j.items()) {
    if (k == "type" || k == "name") continue;
    if (v.is_boolean())
      cfg.set(k, v.get<bool>());
    else if (v.is_number())
      cfg.set(k, v.get<float>());
    else if (v.is_string())
      cfg.set<string>(k, v.get<string>());
  }
}

static void parse_scheduler_json(const nlohmann::json &j, SchedulerConfig &cfg) {
  cfg.type = j.value("type", cfg.type.empty() ? "no_op" : cfg.type);
  for (auto &[k, v] : j.items()) {
    if (k == "type" || k == "name") continue;
    if (v.is_boolean())
      cfg.set(k, v.get<bool>());
    else if (v.is_number_integer())
      cfg.set(k, v.get<size_t>());
    else if (v.is_number_float())
      cfg.set(k, v.get<float>());
    else if (v.is_string())
      cfg.set<string>(k, v.get<string>());
    else if (v.is_array()) {
      std::vector<size_t> arr;
      for (const auto &item : v) arr.push_back(item.get<size_t>());
      cfg.set(k, arr);
    }
  }
}

static void parse_loss_json(const nlohmann::json &j, LossConfig &cfg) {
  cfg.type = j.value("type", cfg.type.empty() ? "logsoftmax_cross_entropy" : cfg.type);
  for (auto &[k, v] : j.items()) {
    if (k == "type" || k == "name") continue;
    if (v.is_boolean())
      cfg.set(k, v.get<bool>());
    else if (v.is_number())
      cfg.set(k, v.get<double>());
    else if (v.is_string())
      cfg.set<string>(k, v.get<string>());
  }
}

void TrainingConfig::print_config() const {
  std::cout << "Training Configuration:" << std::endl;
  std::cout << "  Benchmark Mode: " << (benchmark_mode ? "On" : "Off") << std::endl;
  std::cout << "  Bootstrap Offload: " << (bootstrap_offload ? "On" : "Off") << std::endl;
  std::cout << "  Epochs: " << epochs << std::endl;
  std::cout << "  Batch Size: " << batch_size << std::endl;
  std::cout << "  Max Steps: " << max_steps << std::endl;
  std::cout << "  Train Mode: " << train_mode << std::endl;
  std::cout << "  Initial Learning Rate: " << lr_initial << std::endl;
  std::cout << "  Gradient Accumulation Steps: " << gradient_accumulation_steps << std::endl;
  std::cout << "  Progress Print Interval (batches): " << progress_print_interval << std::endl;
  std::cout << "  Number of Threads: " << num_threads << std::endl;
  std::cout << "  Profiler Type: "
            << (profiler_type == ProfilerType::NONE
                    ? "None"
                    : (profiler_type == ProfilerType::NORMAL ? "Normal" : "Cumulative"))
            << std::endl;
  std::cout << "  Print Profiling Info: " << (print_layer_profiling ? "Yes" : "No") << std::endl;
  std::cout << "  Print Memory Usage: " << (print_memory_usage ? "Yes" : "No") << std::endl;
  std::cout << "  Print Ablation Info: " << (print_ablation ? "Yes" : "No") << std::endl;
  std::cout << "  Number of Microbatches: " << num_microbatches << std::endl;
  std::cout << "  Device ID: "
            << device_type_to_string(device_id.type) + std::to_string(device_id.id) << std::endl;
  std::cout << "  Data Prefetch: " << (prefetch_data ? "Yes" : "No") << std::endl;
  std::cout << "  Prefetch Depth: " << prefetch_depth << std::endl;
  std::cout << "  Async Pipeline Flag: " << (async_pipeline ? "Yes" : "No") << std::endl;
  std::cout << "  Augmentation: " << (augmentation ? "Yes" : "No") << std::endl;
  std::cout << "  Optimizer Type: " << optimizer_config.type << std::endl;
  std::cout << "  Scheduler Type: " << scheduler_config.type << std::endl;
  std::cout << "  Loss Type: " << loss_config.type << std::endl;
}

void TrainingConfig::load_from_json(const string &config_path) {
  std::ifstream file(config_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open config file: " + config_path);
  }

  nlohmann::json config;
  file >> config;
  file.close();

  epochs = config.value("epochs", epochs);
  batch_size = config.value("batch_size", batch_size);
  max_steps = config.value("max_steps", max_steps);
  train_mode = normalize_train_mode(config.value("train_mode", train_mode));
  lr_initial = config.value("lr_initial", lr_initial);
  gradient_accumulation_steps =
      config.value("gradient_accumulation_steps", gradient_accumulation_steps);
  progress_print_interval = config.value("progress_print_interval", progress_print_interval);
  num_threads = config.value("num_threads", num_threads);
  string profiler_type_str = config.value("profiler_type", "NONE");
  if (profiler_type_str == "NORMAL") {
    profiler_type = ProfilerType::NORMAL;
  } else if (profiler_type_str == "CUMULATIVE") {
    profiler_type = ProfilerType::CUMULATIVE;
  } else {
    profiler_type = ProfilerType::NONE;
  }
  print_layer_profiling = config.value("print_layer_profiling", print_layer_profiling);
  print_memory_usage = config.value("print_memory_usage", print_memory_usage);
  print_ablation = config.value("print_ablation", print_ablation);
  num_microbatches = config.value("num_microbatches", num_microbatches);
  if (config.contains("device")) {
    string device_str = config["device"];
    device_id = DeviceID::from_string(device_str);
  }
  model_name = config.value("model_name", model_name);
  model_path = config.value("model_path", model_path);
  dataset_name = config.value("dataset_name", dataset_name);
  dataset_path = config.value("dataset_path", dataset_path);
  string io_dtype_str = config.value("io_dtype", dtype_to_string(io_dtype));
  io_dtype = string_to_dtype(io_dtype_str);
  string param_dtype_str = config.value("param_dtype", dtype_to_string(param_dtype));
  param_dtype = string_to_dtype(param_dtype_str);
  string compute_dtype_str = config.value("compute_dtype", dtype_to_string(compute_dtype));
  compute_dtype = string_to_dtype(compute_dtype_str);

  prefetch_data = config.value("prefetch_data", prefetch_data);
  prefetch_depth = config.value("prefetch_depth", prefetch_depth);
  async_pipeline = config.value("async_pipeline", async_pipeline);
  augmentation = config.value("augmentation", augmentation);
  benchmark_mode = config.value("benchmark_mode", benchmark_mode);
  bootstrap_offload = config.value("bootstrap_offload", bootstrap_offload);

  // Parse LogMode settings from JSON
  if (config.contains("log_mode")) {
    auto log_config = config["log_mode"];
    log_mode.log_loss = log_config.value("log_loss", log_mode.log_loss);
    log_mode.log_accuracy = log_config.value("log_accuracy", log_mode.log_accuracy);
    log_mode.log_precision = log_config.value("log_precision", log_mode.log_precision);
    log_mode.log_recall = log_config.value("log_recall", log_mode.log_recall);
    log_mode.log_f1_score = log_config.value("log_f1_score", log_mode.log_f1_score);
    log_mode.log_perplexity = log_config.value("log_perplexity", log_mode.log_perplexity);
    log_mode.log_top_k_accuracy =
        log_config.value("log_top_k_accuracy", log_mode.log_top_k_accuracy);
    log_mode.log_mae = log_config.value("log_mae", log_mode.log_mae);
    log_mode.log_mse = log_config.value("log_mse", log_mode.log_mse);
    log_mode.log_rmse = log_config.value("log_rmse", log_mode.log_rmse);
  }
  if (config.contains("optimizer")) parse_optimizer_json(config["optimizer"], optimizer_config);
  if (config.contains("scheduler")) parse_scheduler_json(config["scheduler"], scheduler_config);
  if (config.contains("loss")) parse_loss_json(config["loss"], loss_config);
}
}  // namespace tunx