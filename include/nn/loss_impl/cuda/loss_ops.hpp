/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "type/type.hpp"
#ifdef TUNX_USE_CUDA
#include <cuda_runtime.h>

#include <cstddef>

namespace tunx {
namespace cuda {
namespace loss {

// CrossEntropy Loss (from probabilities)
void compute_cross_entropy_loss_probs(DType_t type, const void *predictions, const int *labels,
                                      float &loss, size_t batch_size, size_t num_classes,
                                      double epsilon, cudaStream_t stream);

void compute_cross_entropy_gradient_probs(DType_t dtype, const void *predictions, const int *labels,
                                          void *grad_output, size_t batch_size, size_t num_classes,
                                          double epsilon, cudaStream_t stream);

// CrossEntropy Loss (from logits)
void compute_cross_entropy_loss_logits(DType_t dtype, const void *logits, const int *labels,
                                       float &loss, size_t batch_size, size_t num_classes,
                                       cudaStream_t stream = 0);

void compute_cross_entropy_gradient_logits(DType_t dtype, const void *logits, const int *labels,
                                           void *grad_output, size_t batch_size, size_t num_classes,
                                           cudaStream_t stream = 0);

// MSE Loss
void compute_mse_loss(DType_t dtype, const void *predictions, const void *targets, float &loss,
                      size_t batch_size, size_t output_size, cudaStream_t stream);

void compute_mse_gradient(DType_t dtype, const void *predictions, const void *targets,
                          void *grad_output, size_t batch_size, size_t output_size,
                          cudaStream_t stream);

// MAE Loss
void compute_mae_loss(DType_t dtype, const void *predictions, const void *targets, float &loss,
                      size_t batch_size, size_t output_size, cudaStream_t stream);

void compute_mae_gradient(DType_t dtype, const void *predictions, const void *targets,
                          void *grad_output, size_t batch_size, size_t output_size,
                          cudaStream_t stream);

// Huber Loss
void compute_huber_loss(DType_t dtype, const void *predictions, const void *targets, float &loss,
                        size_t batch_size, size_t output_size, double delta, cudaStream_t stream);

void compute_huber_gradient(DType_t dtype, const void *predictions, const void *targets,
                            void *grad_output, size_t batch_size, size_t output_size, double delta,
                            cudaStream_t stream);

}  // namespace loss
}  // namespace cuda
}  // namespace tunx

#endif