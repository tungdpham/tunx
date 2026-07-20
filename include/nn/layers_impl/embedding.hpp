/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "nn/siso_layer.hpp"
#include "nn/param.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

namespace internal {
class EmbeddingImpl : public SISOLayerImpl {
private:
  size_t vocab_size_;
  size_t embed_dim_;
  size_t padding_idx_;
  Param weight_;

  void init_impl() override;
  Tensor forward_impl(const Tensor &input, Residuals &residualsuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residualsuals) override;

public:
  EmbeddingImpl(size_t vocab_size, size_t embed_dim, const std::string &name = "embedding",
                     size_t padding_idx = static_cast<size_t>(-1));

  static constexpr const char *TYPE_NAME = "embedding";

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;
  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;


  static std::shared_ptr<EmbeddingImpl> create_from_config(const LayerConfig &config);
};

}  // namespace internal

class Embedding : public LayerRef<internal::EmbeddingImpl> {
public:
  Embedding(size_t vocab_size, size_t embed_dim, const std::string &name = "embedding",
                 size_t padding_idx = static_cast<size_t>(-1))
      : LayerRef(std::make_shared<internal::EmbeddingImpl>(vocab_size, embed_dim, name, padding_idx)) {}

  using LayerRef<internal::EmbeddingImpl>::LayerRef;
};

}  // namespace tunx
