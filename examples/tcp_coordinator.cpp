/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "distributed/tcp_coordinator.hpp"

#include <getopt.h>

#include <cstdlib>
#include <iostream>
#include <memory>

#include "data_loading/dataset_factory.hpp"
#include "device/pool_allocator.hpp"
#include "distributed/coordinator.hpp"
#include "distributed/endpoint.hpp"
#include "distributed/tcp_worker.hpp"
#include "distributed/train.hpp"
#include "nn/example_graphs.hpp"
#include "nn/graph.hpp"
#include "nn/graph_executor.hpp"
#include "partitioner/graph_partitioner.hpp"

using namespace tunx;
using namespace std;

int main(int argc, char *argv[]) {
  std::string config_path;
  static struct option long_options[] = {
      {"config", required_argument, 0, 'c'}, {"help", no_argument, 0, 'h'}, {0, 0, 0, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "c:h", long_options, nullptr)) != -1) {
    switch (opt) {
      case 'c':
        config_path = optarg;
        break;
      case 'h':
        cout << "Usage: " << argv[0] << " [options]" << endl;
        cout << "Options:" << endl;
        cout << "  --config <path>    Path to the JSON configuration file" << endl;
        cout << "  -h, --help         Show this help message" << endl;
        return 0;
      default:
        return 1;
    }
  }

  if (config_path.empty()) {
    cerr << "Error: Configuration file path is required. Use --config <path> to specify it."
         << endl;
    return 1;
  }

  ExampleGraphs::register_defaults();

  TrainingConfig train_config;
  train_config.load_from_json(config_path);
  train_config.print_config();

  TCPConfig tcp_config;
  tcp_config.load_from_json(config_path);
  tcp_config.print_config();

  Device &device = DeviceManager::instance().get(train_config.device_id);
  auto &allocator = PoolAllocator::instance(device, device.default_stream());

  GraphOpts opts{
      .s = device.default_stream(),
      .io_dtype = train_config.io_dtype,
      .param_dtype = train_config.param_dtype,
      .compute_dtype = train_config.compute_dtype,
  };

  Graph graph =
      load_or_create_graph(train_config.model_name, train_config.model_path, allocator, opts);

  if (train_config.dataset_name.empty()) {
    throw std::runtime_error("dataset_name variable is not set!");
  }
  auto [train_dataset, val_dataset] = DatasetFactory::create(
      train_config.dataset_name, train_config.dataset_path, train_config.io_dtype);
  if (!train_dataset || !val_dataset) {
    cerr << "Failed to create data loaders for model: " << train_config.model_name << endl;
    return 1;
  }

  auto criterion = LossFactory::create_from_config(train_config.loss_config);

  auto optimizer = OptimizerFactory::create_from_config(train_config.optimizer_config);

  auto scheduler =
      SchedulerFactory::create_from_config(train_config.scheduler_config, optimizer.get());

  cout << "Configured " << tcp_config.workers.size() << " workers:" << endl;
  Vec<Endpoint> worker_endpoints;
  for (const auto &w : tcp_config.workers) {
    worker_endpoints.push_back(w.endpoint);
    cout << w.endpoint.to_json().dump(4) << endl;
  }

  Endpoint coordinator_endpoint = Endpoint::tcp(tcp_config.host, tcp_config.port);

  if (tcp_config.local_worker_position < 0 ||
      tcp_config.local_worker_position >= static_cast<int>(tcp_config.workers.size())) {
    throw std::runtime_error("Local worker position out of bounds");
  }
  Endpoint local_worker_endpoint = tcp_config.workers[tcp_config.local_worker_position].endpoint;

  cout << "Local worker endpoint: " << local_worker_endpoint.to_json().dump(4) << endl;

  // hard-coded for now
  auto worker = std::make_unique<TCPWorker>(local_worker_endpoint, train_config.device_id,
                                            train_config.bootstrap_offload);

  // Sample compute times for ComputeBandwidthPartitioner
  std::unique_ptr<PartitionerBase> partitioner;

  Tensor batch_data, batch_labels;
  std::unordered_map<std::string, size_t> global_node_profiles;
  std::unordered_map<std::shared_ptr<LayerImpl>, double> global_edge_costs;
  double optimizer_step_time = 0.0;
  double zero_grads_time = 0.0;
  DeviceMesh mesh;

  for (size_t i = 0; i < tcp_config.workers.size(); ++i) {
    mesh.compute_powers.push_back(tcp_config.workers[i].compute_power);
    if (i < tcp_config.workers.size() - 1) {
      if (!tcp_config.workers[i].interconnect_speeds.empty()) {
        mesh.link_speeds.push_back(tcp_config.workers[i].interconnect_speeds[0]);
      } else {
        mesh.link_speeds.push_back(3000000.0);  // fallback 3GB/s
      }
    }
  }

  if (train_dataset->get_batch(train_config.batch_size, batch_data, batch_labels)) {
    cout << "Profiling graph edges for partitioner..." << endl;
    GraphExecutor executor(graph);
    TensorBundle input_map({{"input", batch_data}});

    // Warmup
    for (int i = 0; i < 5; ++i) {
      executor.forward(input_map);
      executor.clear_residuals();
    }
    // Measured step
    auto [output_map, edge_profiles, node_profiles] =
        executor.profile_edges_forward(input_map, false);

    auto &mem_pool = PoolAllocator::instance(graph.device(), graph.device().default_stream());
    TensorBundle output_grad_map;
    for (const auto &[name, tensor] : output_map) {
      output_grad_map.set(name, Tensor(tensor.shape(), tensor.dtype(), mem_pool));
    }
    auto [backward_output_map, backward_edge_profiles] =
        executor.profile_edges_backward(output_grad_map, false);

    for (const auto &pair : node_profiles) {
      global_node_profiles[pair.first->uid()] = pair.second;
    }

    optimizer->attach(graph);
    for (int i = 0; i < 5; ++i) {
      optimizer->zero_grads();
      optimizer->update();
    }
    graph.device().default_stream().sync();

    auto zero_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
      optimizer->zero_grads();
    }
    graph.device().default_stream().sync();
    auto zero_end = std::chrono::high_resolution_clock::now();
    zero_grads_time =
        std::chrono::duration_cast<std::chrono::microseconds>(zero_end - zero_start).count() /
        1000.0 / 10.0;

    auto opt_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
      optimizer->update();
    }
    graph.device().default_stream().sync();
    auto opt_end = std::chrono::high_resolution_clock::now();
    optimizer_step_time =
        std::chrono::duration_cast<std::chrono::microseconds>(opt_end - opt_start).count() /
        1000.0 / 10.0;

    // Use the first compute power as the baseline multiplier
    double baseline_power = mesh.compute_powers.empty() ? 1.0 : mesh.compute_powers[0];

    for (const auto &pair : edge_profiles) {
      global_edge_costs[pair.first->layer()] += pair.second.exec_time;
    }
    for (const auto &pair : backward_edge_profiles) {
      global_edge_costs[pair.first->layer()] += pair.second.exec_time;
    }

    auto compute_cost_fn = [&global_edge_costs, baseline_power](const Edge &edge) -> double {
      double cost = 0.0;
      auto it = global_edge_costs.find(edge->layer());
      if (it != global_edge_costs.end()) {
        cost = it->second;
      }
      return cost > 0.0 ? cost * baseline_power : 1.0;
    };

    if (tcp_config.partition_policy == "equal") {
      cout << "Using equal edge count GraphPartitioner..." << endl;
      std::vector<size_t> equal_ratios(tcp_config.workers.size(), 1);
      partitioner = std::make_unique<GraphPartitioner>(equal_ratios);
    } else if (tcp_config.partition_policy == "compute") {
      cout << "Using Compute-only ComputeBandwidthPartitioner..." << endl;
      auto compute_only_activation_fn = [](const Node &) -> double { return 0.0; };
      partitioner = std::make_unique<ComputeBandwidthPartitioner>(
          mesh, compute_cost_fn, compute_only_activation_fn, train_config.num_microbatches,
          optimizer_step_time, zero_grads_time);
    } else {
      cout << "Using Compute+Bandwidth ComputeBandwidthPartitioner..." << endl;
      auto activation_size_fn = [node_profiles](const Node &node) -> double {
        auto it = node_profiles.find(node);
        if (it != node_profiles.end()) {
          return static_cast<double>(it->second) * 2.0;  // 2x for forward + backward communication
        }
        return 1048576.0;  // fallback to 1MB if unknown
      };
      partitioner = std::make_unique<ComputeBandwidthPartitioner>(
          mesh, compute_cost_fn, activation_size_fn, train_config.num_microbatches,
          optimizer_step_time, zero_grads_time);
    }
    train_dataset->reset();
  } else {
    cout << "Warning: Could not get a batch to profile. Falling back to uniform GraphPartitioner."
         << endl;
    std::vector<size_t> fallback_ratios;
    for (const auto &w : tcp_config.workers)
      fallback_ratios.push_back(static_cast<size_t>(w.compute_power));
    partitioner = std::make_unique<GraphPartitioner>(fallback_ratios);
  }

  CoordinatorConfig coordinator_config{
      std::move(graph),
      std::move(optimizer),
      std::move(scheduler),
      std::move(partitioner),
      coordinator_endpoint,
      std::move(worker),
      std::move(worker_endpoints),
  };

  NetworkCoordinator coordinator(std::move(tcp_config), std::move(coordinator_config));

  coordinator.initialize();

  cout << "\n=== Unified Partition Metrics ===" << endl;
  auto &partitions = coordinator.get_partitions();
  for (size_t k = 0; k < partitions.size(); ++k) {
    size_t start = partitions[k].start_layer;
    size_t end = start + partitions[k].layer_count;
    cout << "Worker " << (k + 1) << " edges: " << partitions[k].layer_count << " (edges " << start
         << " to " << (end - 1) << ")" << endl;
    if (k < partitions.size() - 1) {
      double boundary_bytes = 0.0;
      for (const auto &uid : partitions[k].output_uids) {
        auto it = global_node_profiles.find(uid);
        if (it != global_node_profiles.end()) {
          boundary_bytes += static_cast<double>(it->second);
        } else {
          boundary_bytes += 1048576.0;  // 1MB fallback
        }
      }
      cout << "Boundary " << (k + 1) << " -> " << (k + 2)
           << " size (MiB): " << std::round(boundary_bytes / (1024.0 * 1024.0)) << endl;
    }
  }
  cout << "=================================\n" << endl;

  if (partitions.size() > 0 && !mesh.compute_powers.empty()) {
    double sum_latencies = 0.0;
    double bottleneck = 0.0;
    double baseline_power = mesh.compute_powers.empty() ? 1.0 : mesh.compute_powers[0];
    for (size_t k = 0; k < partitions.size(); ++k) {
      double compute_time = 0.0;
      for (const auto &edge : partitions[k].graph.edges()) {
        auto it = global_edge_costs.find(edge->layer());
        if (it != global_edge_costs.end()) {
          compute_time += (it->second > 0.0 ? it->second * baseline_power : 1.0);
        } else {
          compute_time += 1.0;
        }
      }
      compute_time /= mesh.compute_powers[k];

      double comm_time = 0.0;
      if (k < partitions.size() - 1) {
        double boundary_bytes = 0.0;
        for (const auto &uid : partitions[k].output_uids) {
          auto it = global_node_profiles.find(uid);
          if (it != global_node_profiles.end()) {
            boundary_bytes += static_cast<double>(it->second) * 2.0;
          } else {
            boundary_bytes += 1048576.0;  // fallback to 1MB if unknown
          }
        }
        comm_time = (boundary_bytes / mesh.link_speeds[k]) * 1000.0;
      }

      sum_latencies += std::max(compute_time, comm_time);
      bottleneck = std::max({bottleneck, compute_time, comm_time});
    }

    double pipeline_bubble = (train_config.num_microbatches > 0)
                                 ? (sum_latencies - bottleneck) / train_config.num_microbatches
                                 : 0.0;
    double predicted_step_time =
        bottleneck + pipeline_bubble + optimizer_step_time + zero_grads_time;

    cout << "\n=== Predicted Step Metrics ===" << endl;
    cout << "Predicted Bottleneck J (ms): " << bottleneck << endl;
    cout << "Pipeline Bubble (ms): " << pipeline_bubble << endl;
    cout << "Optimizer Step (ms): " << optimizer_step_time << endl;
    cout << "Zero Gradients (ms): " << zero_grads_time << endl;
    cout << "Predicted Total Step Time (ms): " << predicted_step_time << endl;
    cout << "================================\n" << endl;
  }

  if (!coordinator.deploy_stages()) {
    cerr << "Failed to deploy stages. Make sure workers are running." << endl;
    return 1;
  }

  coordinator.start();
  train_model(coordinator, train_dataset, val_dataset, criterion, train_config);
  coordinator.stop();
  return 0;
}