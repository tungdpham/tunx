/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>

#include "device/device_manager.hpp"
#include "roce_communicator.hpp"
#include "worker.hpp"

namespace tunx {

/**
 * @brief RoCE-based pipeline stage worker
 *
 * Standalone worker process that listens for stage configurations
 * from a coordinator and processes distributed pipeline jobs using RDMA.
 */
class RoCEWorker : public Worker {
public:
  /**
   * @brief Constructor for RoCE worker
   * @param host Hostname or IP to bind to (for TCP handshake)
   * @param port TCP port for initial connection setup
   * @param device_name IB device name (e.g., "mlx5_0")
   * @param gid_index GID index for RoCE
   * @param use_gpu Whether to use CUDA for processing
   */
  explicit RoCEWorker(Endpoint worker_endpoint, DeviceID device_id, bool bootstrap_offload)
      : Worker(device_id, bootstrap_offload) {
    auto communicator = RoCECommunicator::create(worker_endpoint, RoCECommunicator::Config{});

    communicator->start_server();

    this->communicator_ = std::move(communicator);
  }

  ~RoCEWorker() override { stop(); }
};

}  // namespace tunx
