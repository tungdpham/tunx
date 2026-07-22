/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>

#include "type/type.hpp"

namespace tunx {
namespace cpu {
namespace loss {

// CrossEntropy Loss (from probabilities)
void compute_cross_entropy_loss_probs(DType_t dtype, const void *predictions, const int *labels,
                                      float &loss, size_t batch_size, size_t num_classes,
                                      double epsilon);

void compute_cross_entropy_gradient_probs(DType_t dtype, const void *predictions, const int *labels,
                                          void *grad_output, size_t batch_size, size_t num_classes,
                                          double epsilon);

// CrossEntropy Loss (from logits)
void compute_cross_entropy_gradient_logits(DType_t dtype, const void *logits, const int *labels,
                                           void *grad_output, size_t batch_size,
                                           size_t num_classes);

void compute_cross_entropy_loss_logits(DType_t dtype, const void *logits, const int *labels,
                                       float &loss, size_t batch_size, size_t num_classes);

// MSE Loss
void compute_mse_loss(DType_t dtype, const void *predictions, const void *targets, float &loss,
                      size_t batch_size, size_t output_size);

void compute_mse_gradient(DType_t dtype, const void *predictions, const void *targets,
                          void *grad_output, size_t batch_size, size_t output_size);

// MAE Loss
void compute_mae_loss(DType_t dtype, const void *predictions, const void *targets, float &loss,
                      size_t batch_size, size_t output_size);

void compute_mae_gradient(DType_t dtype, const void *predictions, const void *targets,
                          void *grad_output, size_t batch_size, size_t output_size);

// Huber Loss
void compute_huber_loss(DType_t dtype, const void *predictions, const void *targets, float &loss,
                        size_t batch_size, size_t output_size, double delta);

void compute_huber_gradient(DType_t dtype, const void *predictions, const void *targets,
                            void *grad_output, size_t batch_size, size_t output_size, double delta);

}  // namespace loss
}  // namespace cpu
}  // namespace tunx
