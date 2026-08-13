/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <nlohmann/json.hpp>

#include "data_loading/dataset.hpp"
#include "data_loading/regression_dataset.hpp"
#include "nn/graph.hpp"
#include "nn/loss.hpp"
#include "nn/metrics_logger.hpp"
#include "nn/optimizers.hpp"
#include "nn/schedulers.hpp"
#include "nn/train_config.hpp"

#ifdef USE_TBB
#include <tbb/info.h>
#include <tbb/scalable_allocator.h>
#include <tbb/task_arena.h>
#endif

#ifdef USE_MK
#include <mkl.h>
#endif

namespace tunx {

#ifdef USE_TBB
inline void tbb_cleanup();
#endif

struct Result {
  double avg_loss = 0.0f;
  double avg_accuracy = -1.0f;
};

Result validate_model(Graph &graph, std::unique_ptr<Dataset> &val_dataset,
                      const std::unique_ptr<Loss> &criterion, const TrainingConfig &config,
                      MetricsLogger *logger = nullptr, int epoch = 1);

void train_model(Graph &graph, std::unique_ptr<Dataset> &train_dataset,
                 std::unique_ptr<Dataset> &val_dataset, std::unique_ptr<Optimizer> &optimizer,
                 const std::unique_ptr<Loss> &criterion, std::unique_ptr<Scheduler> &scheduler,
                 const TrainingConfig &config = TrainingConfig());

}  // namespace tunx