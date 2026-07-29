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
#include "nn/metrics_logger.hpp"
#include "nn/metrics_computer.hpp"
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

static Result train_epoch(Graph &graph, unique_ptr<Dataset> &train_dataset,
                          unique_ptr<Optimizer> &optimizer, const unique_ptr<Loss> &criterion,
                          unique_ptr<Scheduler> &scheduler, const TrainingConfig &config,
                          MetricsLogger &logger, int epoch, std::unique_ptr<CsvLogger> &mem_logger) {
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

  cout << "Training batches: " << train_dataset->size() << endl;
  while (get_next(batch_data, batch_labels) &&
         (config.max_steps == -1 || num_batches < config.max_steps)) {
    auto batch_start = chrono::high_resolution_clock::now();
    ++num_batches;
    Tensor device_input = to_device(batch_data, model_device);
    Tensor device_labels = to_device(batch_labels, model_device);

    TensorBundle inputs{{"input", device_input}};

    if (config.print_layer_memory_usage && epoch == 1 && num_batches == 1) {
      if (mem_logger) {
        graph.enable_memory_profiling(true, mem_logger.get());
      }
    }

    TensorBundle outputs = graph.forward(inputs);

    if (config.print_layer_memory_usage && epoch == 1 && num_batches == 1) {
      graph.enable_memory_profiling(false);
      if (mem_logger) mem_logger->flush();
    }

    Tensor predictions = outputs.get("output");

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

    Tensor loss_gradient = Tensor(predictions.shape(), batch_data.dtype(), mem_pool);
    criterion->compute_gradient(predictions, device_labels, loss_gradient);

    predictions = Tensor();  // free prediction buffer early

    if (config.gradient_accumulation_steps > 1) {
      loss_gradient *= (1.0f / static_cast<float>(config.gradient_accumulation_steps));
    }

    TensorBundle output_grads{{"output", loss_gradient}};
    graph.backward(output_grads);

    auto batch_end = chrono::high_resolution_clock::now();
    auto batch_duration = chrono::duration_cast<chrono::milliseconds>(batch_end - batch_start);

    if (++grad_accum_counter == config.gradient_accumulation_steps) {
      grad_accum_counter = 0;

      optimizer->update();
      model_device.default_stream().sync();

      optimizer->zero_grads();
      model_device.default_stream().sync();

      if (scheduler) {
        scheduler->step();
      }
    }

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
      /*
      print_timing_table(graph.profiling_details());
      cout << "Forward time: " << forward_duration << ", Backward time: " << grad_backward_duration
           << ", Loss time: " << loss_duration << ", Gradient time: " << gradient_duration
           << ", Update time: " << update_time << "ms, " << "Zero grads time: " << zero_grads_time
           << "ms" << endl;
      */
    }
    graph.clear_profiling_details();
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
                      const TrainingConfig &config, std::unique_ptr<CsvLogger> &mem_logger) {
  ThreadWrapper thread_wrapper({config.num_threads});

  double best_val_accuracy = 0.0;
  const std::string artifact_name = training_artifact_name(config);
  MetricsLogger logger("tunx_" + artifact_name, config.log_dir, config.log_mode);

  thread_wrapper.execute([&]() -> void {
    for (int epoch = 0; epoch < config.epochs; ++epoch) {
      cout << "Epoch " << epoch + 1 << "/" << config.epochs << endl;

      // train phrase
      auto [avg_train_loss, avg_train_accuracy] = train_epoch(
          graph, train_dataset, optimizer, criterion, scheduler, config, logger, epoch + 1, mem_logger);

      // validation phrase
      auto [avg_val_loss, avg_val_accuracy] =
          validate_model(graph, val_dataset, criterion, config, logger, epoch + 1);

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
                       std::unique_ptr<CsvLogger> &mem_logger) {
  ThreadWrapper thread_wrapper({config.num_threads});

  Tensor batch_data, batch_labels;
  cout << "Starting training epoch..." << endl;
  graph.set_mode(ExecutionMode::TRAIN);
  train_dataset->shuffle();
  train_dataset->reset();

  Device &model_device = graph.device();
  auto &mem_pool = PoolAllocator::instance(model_device, nullptr);

  int grad_accum_counter = 0;
  const std::string artifact_name = training_artifact_name(config);
  MetricsLogger logger("tunx_" + artifact_name, config.log_dir, config.log_mode);

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

      if (config.print_layer_memory_usage && steps == 0) {
        if (mem_logger) {
          graph.enable_memory_profiling(true, mem_logger.get());
        }
      }

      TensorBundle outputs = graph.forward(inputs);

      if (config.print_layer_memory_usage && steps == 0) {
        graph.enable_memory_profiling(false);
        if (mem_logger) mem_logger->flush();
      }

      Tensor predictions = outputs.get("output");
      float loss;
      criterion->compute_loss(predictions, device_labels, loss);

      MetricsComputer metrics_comp(config.log_mode);
      ComputedMetrics computed_metrics = metrics_comp.compute(predictions, device_labels, loss);
      int corrects = computed_metrics.corrects;

      Tensor loss_gradient = Tensor(predictions.shape(), batch_data.dtype(), mem_pool);
      criterion->compute_gradient(predictions, device_labels, loss_gradient);

      if (config.gradient_accumulation_steps > 1) {
        loss_gradient *= (1.0f / static_cast<float>(config.gradient_accumulation_steps));
      }
      TensorBundle output_grads{{"output", loss_gradient}};
      graph.backward(output_grads);

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
      graph.save_state(file);
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

  std::unique_ptr<CsvLogger> mem_logger;
  if (config.print_layer_memory_usage) {
    std::string artifact_name = training_artifact_name(config);
    std::vector<std::string> headers = {"layer", "peak_usage_bytes", "retained_bytes", "unused_bytes", "reserved_bytes"};
    std::string mem_csv_path = "tunx_" + artifact_name + "_" + csv_timestamp() + "_memory.csv";
    if (!config.log_dir.empty()) {
      mem_csv_path = config.log_dir + "/" + mem_csv_path;
    }
    mem_logger = std::make_unique<CsvLogger>("memory", mem_csv_path, headers);
  }

  if (is_val) {
    train_val(graph, train_dataset, val_dataset, optimizer, criterion, scheduler, config, mem_logger);
  } else {
    train_step(graph, train_dataset, optimizer, criterion, scheduler, config, mem_logger);
  }
}

static Result validate_model_impl(Graph &graph, unique_ptr<Dataset> &val_dataset,
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

  while (val_dataset->get_batch(config.batch_size, batch_data, batch_labels)) {
    Tensor device_input = to_device(batch_data, model_device);
    TensorBundle inputs{{"input", device_input}};
    TensorBundle outputs = graph.forward(inputs);
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

Result validate_model(Graph &graph, unique_ptr<Dataset> &val_dataset,
                      const unique_ptr<Loss> &criterion, const TrainingConfig &config,
                      MetricsLogger &logger, int epoch) {
  return validate_model_impl(graph, val_dataset, criterion, config, &logger, epoch);
}

Result validate_model(Graph &graph, unique_ptr<Dataset> &val_dataset,
                      const unique_ptr<Loss> &criterion, const TrainingConfig &config) {
  return validate_model_impl(graph, val_dataset, criterion, config, nullptr, 0);
}

}  // namespace tunx
