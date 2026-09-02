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

  Graph graph = load_or_create_graph(train_config.model_name, train_config.model_path, allocator);

  if (train_config.dataset_name.empty()) {
    throw std::runtime_error("dataset_name variable is not set!");
  }
  auto [train_dataset, val_dataset] =
      DatasetFactory::create(train_config.dataset_name, train_config.dataset_path);
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
  if (tcp_config.partition_policy == "equal") {
    cout << "Using equal edge count GraphPartitioner..." << endl;
    std::vector<size_t> equal_ratios(tcp_config.workers.size(), 1);
    partitioner = std::make_unique<GraphPartitioner>(equal_ratios);
  } else if (train_dataset->get_batch(train_config.batch_size, batch_data, batch_labels)) {
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
        executor.profile_edges_forward(input_map, true);

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

    // Use the first compute power as the baseline multiplier
    double baseline_power = mesh.compute_powers.empty() ? 1.0 : mesh.compute_powers[0];

    auto compute_cost_fn = [edge_profiles, baseline_power](const Edge &edge) -> double {
      auto it = edge_profiles.find(edge);
      if (it != edge_profiles.end()) {
        return it->second.exec_time * baseline_power;
      }
      return 1.0;
    };

    if (tcp_config.partition_policy == "compute") {
      cout << "Using Compute-only ComputeBandwidthPartitioner..." << endl;
      auto compute_only_activation_fn = [](const Node &) -> double { return 0.0; };
      partitioner = std::make_unique<ComputeBandwidthPartitioner>(mesh, compute_cost_fn,
                                                                  compute_only_activation_fn);
    } else {
      cout << "Using Compute+Bandwidth ComputeBandwidthPartitioner..." << endl;
      auto activation_size_fn = [node_profiles](const Node &node) -> double {
        auto it = node_profiles.find(node);
        if (it != node_profiles.end()) {
          return static_cast<double>(it->second);
        }
        return 1048576.0;  // fallback to 1MB if unknown
      };
      partitioner =
          std::make_unique<ComputeBandwidthPartitioner>(mesh, compute_cost_fn, activation_size_fn);
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

  if (!coordinator.deploy_stages()) {
    cerr << "Failed to deploy stages. Make sure workers are running." << endl;
    return 1;
  }

  coordinator.start();
  train_model(coordinator, train_dataset, val_dataset, criterion, train_config);
  coordinator.stop();
  return 0;
}