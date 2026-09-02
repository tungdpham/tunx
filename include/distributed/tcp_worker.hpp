/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <csignal>
#include <memory>

#include "device/device_manager.hpp"
#include "device/device_type.hpp"
#include "device/pool_allocator.hpp"
#include "device/stream.hpp"
#include "distributed/endpoint.hpp"
#include "tcp_communicator.hpp"
#include "worker.hpp"

namespace tunx {

/**
 * @brief TCP-based pipeline stage worker
 *
 * Standalone worker process that listens for stage configurations
 * from a coordinator and processes distributed pipeline jobs.
 */
class TCPWorker : public Worker {
public:
  /**
   * @brief Constructor with optional thread affinity configuration
   * @param listen_port Port to listen on for connections
   * @param use_gpu Whether to use CUDA for processing
   * @param use_ecore_affinity Whether to bind worker threads to E-cores for efficiency
   * @param max_ecore_threads Maximum number of E-cores to use (-1 for all available)
   * @param io_threads Number of IO threads for the TCP communicator (default: 1)
   */
  explicit TCPWorker(Endpoint endpoint, DeviceID device_id, bool bootstrap_offload,
                     TCPCommunicator::Config config = TCPCommunicator::Config())
      : Worker(device_id, bootstrap_offload) {
    auto &device = DeviceManager::instance().get(device_id);
    std::cout << "Device type: " << device_type_to_string(device.device_type()) << std::endl;
    auto &allocator = PoolAllocator::instance(device, device.default_stream());
    auto communicator = std::make_unique<TCPCommunicator>(endpoint, allocator, config);

    communicator->start_server();

    this->communicator_ = std::move(communicator);
  }

  ~TCPWorker() {}
};
}  // namespace tunx
