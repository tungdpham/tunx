/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/train.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <cctype>
#include <chrono>
#include <cstddef>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include "data_loading/batch_prefetcher.hpp"
#include "device/pool_allocator.hpp"
#include "device/stream.hpp"
#include "nn/edge_profile.hpp"
#include "nn/execution_plan.hpp"
#include "nn/graph_executor.hpp"
#include "nn/metrics_computer.hpp"
#include "nn/metrics_logger.hpp"
#include "nn/tensor_bundle.hpp"
#include "threading/thread_wrapper.hpp"
#include "type/type.hpp"

using namespace std;

namespace tunx {

static std::string training_artifact_name(const TrainingConfig &config) {
  if (!config.model_name.empty()) {
    return config.model_name;
  }
  if (!config.model_path.empty()) {
    return std::filesystem::path(config.model_path).stem().string();
  }
  return "graph";
}

void print_timing_table(const std::vector<std::pair<std::string, double>> &timings) {
  // Table Header
  fmt::print("{:<25} | {:>12}\n", "Layer Name", "Time (ms)");
  fmt::print("{:-<25}-+-{:-<12}\n", "", "");  // Divider line

  // Rows
  for (const auto &[layer, time_ms] : timings) {
    fmt::print("{:<25} | {:>10.3f} ms\n", layer, time_ms);
  }
}

static void log_edge_profiles(const std::map<Edge, EdgeProfile> &profiles,
                              std::unique_ptr<CsvLogger> &csv_logger) {
  fmt::print("\n{:=^80}\n", " Edge Profiles ");
  fmt::print("{:<30} | {:>10} | {:>12} | {:>12}\n", "Layer Name", "Time (ms)", "Peak Mem (B)",
             "Net Mem (B)");
  fmt::print("{:-<30}-+-{:-<10}-+-{:-<12}-+-{:-<12}\n", "", "", "", "");
  for (const auto &[edge, profile] : profiles) {
    fmt::print("{:<30} | {:>10.3f} | {:>12} | {:>12}\n", edge->layer()->name(), profile.exec_time,
               profile.total_mem, profile.net_mem);
  }
  fmt::print("{:=^80}\n\n", "");

  if (csv_logger) {
    for (const auto &[edge, profile] : profiles) {
      std::unordered_map<std::string, std::string> row = {
          {"layer_name", edge->layer()->name()},
          {"time_ms", std::to_string(profile.exec_time)},
          {"peak_mem_b", std::to_string(profile.total_mem)},
          {"net_mem_b", std::to_string(profile.net_mem)}};
      csv_logger->log(row);
    }
  }
}

static void log_execution_plan_stats(GraphExecutor &executor, TensorBundle &inputs,
                                     std::unique_ptr<CsvLogger> &csv_logger) {
  ExecutionPlan naive_plan;
  naive_plan.order = executor.graph().edges();
  auto naive_stats = executor.profile_forward_plan(inputs, naive_plan);

  std::ofstream debug_stream("debug_macro_solver.txt");

  auto &built_plan = executor.build_plans(inputs);
  auto &macro_plan = built_plan.forward_plan;
  auto macro_stats = executor.profile_forward_plan(inputs, macro_plan);

  fmt::print("\n{:=^80}\n", " Execution Plan Stats ");
  fmt::print("Naive Order Peak Memory: {} bytes\n", naive_stats.peak_mem);
  fmt::print("Naive Path: ");
  for (size_t i = 0; i < naive_plan.order.size(); ++i) {
    fmt::print("{}", naive_plan.order[i]->layer()->name());
    if (i != naive_plan.order.size() - 1) fmt::print(" -> ");
  }
  fmt::print("\n\n");

  fmt::print("Macro Order Peak Memory: {} bytes\n", macro_stats.peak_mem);
  fmt::print("Macro Path: ");
  for (size_t i = 0; i < macro_plan.order.size(); ++i) {
    fmt::print("{}", macro_plan.order[i]->layer()->name());
    if (i != macro_plan.order.size() - 1) fmt::print(" -> ");
  }
  fmt::print("\n");
  fmt::print("{:=^80}\n\n", "");

  if (csv_logger) {
    for (size_t i = 0; i < naive_stats.edge_stats.size(); ++i) {
      const auto &s = naive_stats.edge_stats[i];
      std::unordered_map<std::string, std::string> row = {
          {"plan", "naive"},
          {"step", std::to_string(i)},
          {"layer_name", s.layer_name},
          {"allocated_mem", std::to_string(s.allocated_mem)},
          {"peak_mem", std::to_string(s.peak_mem)},
      };
      csv_logger->log(row);
    }
    for (size_t i = 0; i < macro_stats.edge_stats.size(); ++i) {
      const auto &s = macro_stats.edge_stats[i];
      std::unordered_map<std::string, std::string> row = {
          {"plan", "macro"},
          {"step", std::to_string(i)},
          {"layer_name", s.layer_name},
          {"allocated_mem", std::to_string(s.allocated_mem)},
          {"peak_mem", std::to_string(s.peak_mem)},
      };
      csv_logger->log(row);
    }
  }
}

static void log_execution_plan_stats_backward(GraphExecutor &executor, TensorBundle &inputs,
                                              std::unique_ptr<CsvLogger> &csv_logger) {
  ExecutionPlan naive_plan;
  for (auto it = executor.graph().edges().rbegin(); it != executor.graph().edges().rend(); ++it) {
    naive_plan.order.push_back(*it);
  }

  auto naive_stats = executor.profile_backward_plan(inputs, naive_plan);

  auto &built_plan = executor.build_plans(inputs);
  auto &macro_plan = built_plan.backward_plan;
  auto macro_stats = executor.profile_backward_plan(inputs, macro_plan);

  fmt::print("\n{:=^80}\n", " Backward Execution Plan Stats ");
  fmt::print("Naive Order Peak Memory: {} bytes\n", naive_stats.peak_mem);
  fmt::print("Naive Path: ");
  for (size_t i = 0; i < naive_plan.order.size(); ++i) {
    fmt::print("{}", naive_plan.order[i]->layer()->name());
    if (i != naive_plan.order.size() - 1) fmt::print(" -> ");
  }
  fmt::print("\n\n");

  fmt::print("Macro Order Peak Memory: {} bytes\n", macro_stats.peak_mem);
  fmt::print("Macro Path: ");
  for (size_t i = 0; i < macro_plan.order.size(); ++i) {
    fmt::print("{}", macro_plan.order[i]->layer()->name());
    if (i != macro_plan.order.size() - 1) fmt::print(" -> ");
  }
  fmt::print("\n");
  fmt::print("{:=^80}\n\n", "");

  if (csv_logger) {
    for (size_t i = 0; i < naive_stats.edge_stats.size(); ++i) {
      const auto &s = naive_stats.edge_stats[i];
      std::unordered_map<std::string, std::string> row = {
          {"plan", "naive"},
          {"step", std::to_string(i)},
          {"layer_name", s.layer_name},
          {"allocated_mem", std::to_string(s.allocated_mem)},
          {"peak_mem", std::to_string(s.peak_mem)},
      };
      csv_logger->log(row);
    }
    for (size_t i = 0; i < macro_stats.edge_stats.size(); ++i) {
      const auto &s = macro_stats.edge_stats[i];
      std::unordered_map<std::string, std::string> row = {
          {"plan", "macro"},
          {"step", std::to_string(i)},
          {"layer_name", s.layer_name},
          {"allocated_mem", std::to_string(s.allocated_mem)},
          {"peak_mem", std::to_string(s.peak_mem)},
      };
      csv_logger->log(row);
    }
  }
}

static Result train_epoch(Graph &graph, unique_ptr<Dataset> &train_dataset,
                          unique_ptr<Optimizer> &optimizer, const unique_ptr<Loss> &criterion,
                          unique_ptr<Scheduler> &scheduler, const TrainingConfig &config,
                          MetricsLogger &logger, int epoch,
                          std::unique_ptr<CsvLogger> &forward_profiles_logger,
                          std::unique_ptr<CsvLogger> &forward_plan_logger,
                          std::unique_ptr<CsvLogger> &backward_profiles_logger,
                          std::unique_ptr<CsvLogger> &backward_plan_logger) {
  auto train_start = chrono::high_resolution_clock::now();
  Tensor batch_data, batch_labels;
  Device &model_device = graph.device();
  auto &mem_pool = PoolAllocator::instance(model_device, model_device.default_stream());

  cout << "Starting training epoch..." << endl;
  graph.set_mode(ExecutionMode::TRAIN);
  train_dataset->shuffle();
  train_dataset->reset();

  float total_loss = 0.0;
  int total_corrects = 0;
  size_t total_class_num = 0;
  int num_batches = 0;
  int grad_accum_counter = 0;

  std::unique_ptr<BatchPrefetcher> prefetcher;
  if (config.prefetch_data) {
    prefetcher =
        std::make_unique<BatchPrefetcher>(*train_dataset, config.batch_size, config.prefetch_depth);
    prefetcher->start();
    cout << "Data prefetching enabled. Depth: " << config.prefetch_depth << endl;
  }

  auto get_next = [&](Tensor &data, Tensor &labels) -> bool {
    if (prefetcher) {
      return prefetcher->next(data, labels);
    }
    return train_dataset->get_batch(config.batch_size, data, labels);
  };

  graph.save_dot("current_graph.dot");

  GraphExecutor executor(graph);

  // profiling
  if (config.print_layer_profiling) {
    get_next(batch_data, batch_labels);
    Tensor device_input = to_device(batch_data, model_device);
    Tensor device_labels = to_device(batch_labels, model_device);
    TensorBundle inputs{{"input", device_input}};
    auto &built_plan = executor.build_plans(inputs);
    log_edge_profiles(built_plan.forward_edge_profiles, forward_profiles_logger);
    log_execution_plan_stats(executor, inputs, forward_plan_logger);

    log_edge_profiles(built_plan.backward_edge_profiles, backward_profiles_logger);
    log_execution_plan_stats_backward(executor, inputs, backward_plan_logger);

    // avoid affecting training
    train_dataset->reset();
  }

  cout << "Training batches: " << train_dataset->size() << endl;
  while (get_next(batch_data, batch_labels) &&
         (config.max_steps == -1 || num_batches < config.max_steps)) {
    auto batch_start = chrono::high_resolution_clock::now();
    ++num_batches;
    Tensor device_input = to_device(batch_data, model_device);
    Tensor device_labels = to_device(batch_labels, model_device);

    TensorBundle inputs{{"input", device_input}};

    TensorBundle outputs = executor.forward(inputs);

    Tensor &predictions = outputs.get("output");

    size_t batch_size = 1;
    for (size_t i = 0; i < predictions.dims() - 1; ++i) {
      batch_size *= predictions.shape()[i];
    }
    total_class_num += batch_size;

    float loss;
    criterion->compute_loss(predictions, device_labels, loss);
    total_loss += loss;

    MetricsComputer metrics_comp(config.log_mode);
    ComputedMetrics computed_metrics = metrics_comp.compute(predictions, device_labels, loss);
    int batch_corrects = computed_metrics.corrects;

    Tensor loss_gradient = Tensor(predictions.shape(), predictions.dtype(), mem_pool);
    criterion->compute_gradient(predictions, device_labels, loss_gradient);

    predictions = Tensor();  // free prediction buffer early

    if (config.gradient_accumulation_steps > 1) {
      loss_gradient *= (1.0f / static_cast<float>(config.gradient_accumulation_steps));
    }

    TensorBundle output_grads{{"output", loss_gradient}};

    executor.backward(output_grads);

    if (++grad_accum_counter == config.gradient_accumulation_steps) {
      grad_accum_counter = 0;

      optimizer->update();

      optimizer->zero_grads();

      if (scheduler) {
        scheduler->step();
      }
    }
    model_device.default_stream().sync();
    auto batch_end = chrono::high_resolution_clock::now();
    auto batch_duration = chrono::duration_cast<chrono::milliseconds>(batch_end - batch_start);

    // Log batch metrics
    computed_metrics.time_ms = batch_duration.count();
    if (config.log_mode.log_accuracy) {
      total_corrects += batch_corrects;
    }
    logger.log_train_step(epoch, num_batches, computed_metrics);

    if (num_batches % config.progress_print_interval == 0) {
      cout << "Batch ID: " << num_batches << ", Batch's Loss: " << fixed << setprecision(4) << loss
           << ", Cumulative Accuracy: " << setprecision(2)
           << (total_corrects * 100.0 / total_class_num) << "%";
      if (config.log_mode.log_f1_score && computed_metrics.f1_score) {
        cout << ", F1: " << setprecision(4) << *computed_metrics.f1_score;
      }
      if (config.log_mode.log_perplexity && computed_metrics.perplexity) {
        cout << ", PPL: " << setprecision(2) << *computed_metrics.perplexity;
      }
      cout << ", Batch Time: " << batch_duration.count() << "ms" << endl;
    }
  }
  cout << endl;

  const double avg_train_loss = total_loss / num_batches;
  const double avg_train_accuracy = static_cast<double>(total_corrects) / total_class_num;

  auto train_end = chrono::high_resolution_clock::now();
  auto train_epoch_duration = chrono::duration_cast<chrono::milliseconds>(train_end - train_start);
  cout << "Training epoch completed in " << train_epoch_duration.count() << "ms" << endl;
  return {avg_train_loss, avg_train_accuracy};
}

static void train_val(Graph &graph, unique_ptr<Dataset> &train_dataset,
                      unique_ptr<Dataset> &val_dataset, unique_ptr<Optimizer> &optimizer,
                      const unique_ptr<Loss> &criterion, unique_ptr<Scheduler> &scheduler,
                      const TrainingConfig &config, MetricsLogger &logger,
                      std::unique_ptr<CsvLogger> &forward_profiles_logger,
                      std::unique_ptr<CsvLogger> &forward_plan_logger,
                      std::unique_ptr<CsvLogger> &backward_profiles_logger,
                      std::unique_ptr<CsvLogger> &backward_plan_logger) {
  ThreadWrapper thread_wrapper({config.num_threads});

  double best_val_accuracy = 0.0;
  const std::string artifact_name = training_artifact_name(config);

  thread_wrapper.execute([&]() -> void {
    for (int epoch = 0; epoch < config.epochs; ++epoch) {
      cout << "Epoch " << epoch + 1 << "/" << config.epochs << endl;

      // train phrase
      auto [avg_train_loss, avg_train_accuracy] =
          train_epoch(graph, train_dataset, optimizer, criterion, scheduler, config, logger,
                      epoch + 1, forward_profiles_logger, forward_plan_logger,
                      backward_profiles_logger, backward_plan_logger);

      // validation phrase
      auto [avg_val_loss, avg_val_accuracy] =
          validate_model(graph, val_dataset, criterion, config, &logger, epoch + 1);

      if (avg_val_accuracy > best_val_accuracy) {
        best_val_accuracy = avg_val_accuracy;
        cout << "New best validation accuracy: " << fixed << setprecision(2)
             << best_val_accuracy * 100.0 << "%" << endl;
        try {
          filesystem::create_directories("model_snapshots");
          string filepath = "model_snapshots/" + artifact_name;
          ofstream file(filepath, ios::binary);
          if (!file.is_open()) {
            throw runtime_error("Failed to open file: " + filepath);
          }
          graph.save_state(file);
          file.close();
          cout << "Model saved to " << filepath << endl;
        } catch (const exception &e) {
          cerr << "Error saving model: " << e.what() << endl;
        }
      }

      cout << string(60, '-') << endl;
      cout << "Epoch " << epoch + 1 << "/" << config.epochs << endl;
      cout << "Training   - Loss: " << fixed << setprecision(4) << avg_train_loss
           << ", Accuracy: " << setprecision(2) << avg_train_accuracy * 100.0 << "%" << endl;
      cout << "Validation - Loss: " << fixed << setprecision(4) << avg_val_loss
           << ", Accuracy: " << setprecision(2) << avg_val_accuracy * 100.0 << "%" << endl;
      cout << string(60, '=') << endl;

      // Log epoch metrics
      {
        std::unordered_map<std::string, double> metrics;
        if (config.log_mode.log_loss) {
          metrics["train_loss"] = avg_train_loss;
          metrics["val_loss"] = avg_val_loss;
        }
        if (config.log_mode.log_accuracy) {
          metrics["train_accuracy_pct"] = avg_train_accuracy * 100.0;
          metrics["val_accuracy_pct"] = avg_val_accuracy * 100.0;
        }
        logger.log_epoch(epoch + 1, metrics);
      }
    }
  });

  thread_wrapper.clean_buffers();
}

static void train_step(Graph &graph, unique_ptr<Dataset> &train_dataset,
                       const unique_ptr<Optimizer> &optimizer, const unique_ptr<Loss> &criterion,
                       const unique_ptr<Scheduler> &scheduler, const TrainingConfig &config,
                       MetricsLogger &logger, std::unique_ptr<CsvLogger> &forward_profiles_logger,
                       std::unique_ptr<CsvLogger> &forward_plan_logger,
                       std::unique_ptr<CsvLogger> &backward_profiles_logger,
                       std::unique_ptr<CsvLogger> &backward_plan_logger) {
  ThreadWrapper thread_wrapper({config.num_threads});

  Tensor batch_data, batch_labels;
  cout << "Starting training epoch..." << endl;
  graph.set_mode(ExecutionMode::TRAIN);
  train_dataset->shuffle();
  train_dataset->reset();

  Device &model_device = graph.device();
  auto &mem_pool = PoolAllocator::instance(model_device, model_device.default_stream());

  int grad_accum_counter = 0;
  const std::string artifact_name = training_artifact_name(config);

  train_dataset->reset();
  auto start_time = chrono::high_resolution_clock::now();

  std::unique_ptr<BatchPrefetcher> prefetcher;
  auto start_prefetcher = [&]() {
    prefetcher.reset();
    if (config.prefetch_data) {
      prefetcher = std::make_unique<BatchPrefetcher>(*train_dataset, config.batch_size,
                                                     config.prefetch_depth);
      prefetcher->start();
    }
  };
  start_prefetcher();
  if (config.prefetch_data) {
    cout << "Data prefetching enabled. Depth: " << config.prefetch_depth << endl;
  }

  auto get_next = [&](Tensor &data, Tensor &labels) -> bool {
    if (prefetcher) {
      return prefetcher->next(data, labels);
    }
    return train_dataset->get_batch(config.batch_size, data, labels);
  };

  GraphExecutor executor(graph);
  thread_wrapper.execute([&]() -> void {
    for (int steps = 0; steps < config.max_steps; ++steps) {
      if (!get_next(batch_data, batch_labels)) {
        if (prefetcher) {
          prefetcher->stop();
        }
        train_dataset->shuffle();
        train_dataset->reset();
        start_prefetcher();
        if (!get_next(batch_data, batch_labels)) {
          break;
        }
      }
      auto batch_start = chrono::high_resolution_clock::now();
      Tensor device_input = to_device(batch_data, model_device);
      Tensor device_labels = to_device(batch_labels, model_device);
      TensorBundle inputs{{"input", device_input}};

      TensorBundle outputs = executor.forward(inputs);

      Tensor predictions = outputs.get("output");
      float loss;
      criterion->compute_loss(predictions, device_labels, loss);

      MetricsComputer metrics_comp(config.log_mode);
      ComputedMetrics computed_metrics = metrics_comp.compute(predictions, device_labels, loss);
      int corrects = computed_metrics.corrects;

      Tensor loss_gradient = Tensor(predictions.shape(), predictions.dtype(), mem_pool);
      criterion->compute_gradient(predictions, device_labels, loss_gradient);

      if (config.gradient_accumulation_steps > 1) {
        loss_gradient *= (1.0f / static_cast<float>(config.gradient_accumulation_steps));
      }
      TensorBundle output_grads{{"output", loss_gradient}};

      executor.backward(output_grads);

      auto batch_end = chrono::high_resolution_clock::now();
      auto batch_duration = chrono::duration_cast<chrono::milliseconds>(batch_end - batch_start);
      if (++grad_accum_counter == config.gradient_accumulation_steps) {
        grad_accum_counter = 0;
        optimizer->update();
        optimizer->zero_grads();
        if (scheduler) {
          scheduler->step();
        }
      }

      size_t num_samples = 1;
      for (size_t i = 0; i < predictions.dims() - 1; ++i) {
        num_samples *= predictions.shape()[i];
      }

      double batch_acc_pct = corrects * 100.0 / num_samples;

      // Log batch metrics for benchmarking.
      computed_metrics.time_ms = batch_duration.count();
      logger.log_train_step(1, steps, computed_metrics);

      if (steps % config.progress_print_interval == 0) {
        cout << "Batch ID: " << steps << ", Batch's Loss: " << fixed << setprecision(4) << loss
             << ", Batch's Accuracy: " << setprecision(2) << batch_acc_pct << "%";
        if (config.log_mode.log_f1_score && computed_metrics.f1_score) {
          cout << ", F1: " << setprecision(4) << *computed_metrics.f1_score;
        }
        if (config.log_mode.log_perplexity && computed_metrics.perplexity) {
          cout << ", PPL: " << setprecision(2) << *computed_metrics.perplexity;
        }
        cout << ", Batch Time: " << batch_duration.count() << "ms" << endl;
      }
    }

    // training epoch done, print time taken
    auto end_time = chrono::high_resolution_clock::now();
    auto epoch_duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "Training completed in " << epoch_duration.count() << "ms" << endl;

    // save model
    try {
      filesystem::create_directories("model_snapshots");
      string filepath = "model_snapshots/" + artifact_name;
      ofstream file(filepath, ios::binary);
      if (!file.is_open()) {
        throw runtime_error("Failed to open file: " + filepath);
      }
      executor.graph().save_state(file);
      file.close();
      cout << "Model saved to " << filepath << endl;
    } catch (const exception &e) {
      cerr << "Error saving model: " << e.what() << endl;
    }
  });
}

void train_model(Graph &graph, unique_ptr<Dataset> &train_dataset, unique_ptr<Dataset> &val_dataset,
                 unique_ptr<Optimizer> &optimizer, const unique_ptr<Loss> &criterion,
                 unique_ptr<Scheduler> &scheduler, const TrainingConfig &config) {
  optimizer->attach(graph);

  cout << "Training batches: " << train_dataset->size() / config.batch_size << endl;
  cout << "Validation batches: " << val_dataset->size() / config.batch_size << endl;

  vector<size_t> data_shape = train_dataset->get_data_shape();
  data_shape.insert(data_shape.begin(), config.batch_size);  // add batch dim

  bool is_val = config.max_steps == -1;

  std::string timestamp = csv_timestamp();
  std::string artifact_name = training_artifact_name(config);

  std::unique_ptr<CsvLogger> forward_profiles_logger;
  std::unique_ptr<CsvLogger> forward_plan_logger;
  std::unique_ptr<CsvLogger> backward_profiles_logger;
  std::unique_ptr<CsvLogger> backward_plan_logger;

  if (config.print_layer_profiling) {
    std::vector<std::string> profile_headers = {"layer_name", "time_ms", "peak_mem_b", "net_mem_b"};

    std::string fw_profile_csv_path =
        "tunx_" + artifact_name + "_" + timestamp + "_forward_edge_profiles.csv";
    if (!config.log_dir.empty()) fw_profile_csv_path = config.log_dir + "/" + fw_profile_csv_path;
    forward_profiles_logger =
        std::make_unique<CsvLogger>("forward_edge_profiles", fw_profile_csv_path, profile_headers);

    std::string bw_profile_csv_path =
        "tunx_" + artifact_name + "_" + timestamp + "_backward_edge_profiles.csv";
    if (!config.log_dir.empty()) bw_profile_csv_path = config.log_dir + "/" + bw_profile_csv_path;
    backward_profiles_logger =
        std::make_unique<CsvLogger>("backward_edge_profiles", bw_profile_csv_path, profile_headers);

    std::vector<std::string> plan_headers = {"plan", "step", "layer_name", "allocated_mem",
                                             "peak_mem"};

    std::string fw_plan_csv_path =
        "tunx_" + artifact_name + "_" + timestamp + "_forward_execution_plan_stats.csv";
    if (!config.log_dir.empty()) fw_plan_csv_path = config.log_dir + "/" + fw_plan_csv_path;
    forward_plan_logger =
        std::make_unique<CsvLogger>("forward_execution_plan_stats", fw_plan_csv_path, plan_headers);

    std::string bw_plan_csv_path =
        "tunx_" + artifact_name + "_" + timestamp + "_backward_execution_plan_stats.csv";
    if (!config.log_dir.empty()) bw_plan_csv_path = config.log_dir + "/" + bw_plan_csv_path;
    backward_plan_logger = std::make_unique<CsvLogger>("backward_execution_plan_stats",
                                                       bw_plan_csv_path, plan_headers);
  }

  MetricsLogger metrics_logger("tunx_" + artifact_name + "_" + timestamp, config.log_dir,
                               config.log_mode);

  if (is_val) {
    train_val(graph, train_dataset, val_dataset, optimizer, criterion, scheduler, config,
              metrics_logger, forward_profiles_logger, forward_plan_logger,
              backward_profiles_logger, backward_plan_logger);
  } else {
    train_step(graph, train_dataset, optimizer, criterion, scheduler, config, metrics_logger,
               forward_profiles_logger, forward_plan_logger, backward_profiles_logger,
               backward_plan_logger);
  }
}

Result validate_model(Graph &graph, unique_ptr<Dataset> &val_dataset,
                      const unique_ptr<Loss> &criterion, const TrainingConfig &config,
                      MetricsLogger *logger, int epoch) {
  Tensor batch_data, batch_labels;

  graph.set_mode(ExecutionMode::EVAL);
  val_dataset->reset();

  cout << "Starting validation..." << endl;
  double val_loss = 0.0;
  double val_corrects = 0.0;
  int val_batches = 0;
  sref<Device> model_device = graph.device();

  GraphExecutor executor(graph);
  while (val_dataset->get_batch(config.batch_size, batch_data, batch_labels)) {
    Tensor device_input = to_device(batch_data, model_device);
    TensorBundle inputs{{"input", device_input}};
    TensorBundle outputs = executor.forward(inputs);
    Tensor predictions = outputs.get("output");

    Tensor device_labels = to_device(batch_labels, model_device);
    float loss;
    criterion->compute_loss(predictions, device_labels, loss);
    val_loss += loss;

    MetricsComputer metrics_comp(config.log_mode);
    ComputedMetrics computed_metrics = metrics_comp.compute(predictions, device_labels, loss);
    int batch_corrects = computed_metrics.corrects;
    val_corrects += batch_corrects;
    ++val_batches;

    if (logger) {
      logger->log_val_step(epoch, val_batches, computed_metrics);
    }
  }

  double avg_val_loss = val_loss / val_batches;
  double avg_val_accuracy = val_corrects / val_dataset->size();

  return {avg_val_loss, avg_val_accuracy};
}

}  // namespace tunx
