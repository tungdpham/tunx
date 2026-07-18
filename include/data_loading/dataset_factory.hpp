#pragma once

#include "data_loading/dataset.hpp"

namespace tunx {

/**
 * A pair of data datasets for training and validation/testing
 */
struct DatasetPair {
  std::unique_ptr<Dataset> train;
  std::unique_ptr<Dataset> val;
};

/**
 * Factory class for creating data datasets by string name
 */
class DatasetFactory {
public:
  /**
   * Create a pair of data datasets (train and val) for a given dataset type
   * @param dataset_type Type of dataset (e.g., "mnist", "cifar10", "cifar100", "tiny_imagenet",
   * "imagenet100")
   * @param dataset_path Path to the dataset directory or file
   * @return DatasetPair containing the created datasets
   */
  static DatasetPair create(const std::string &dataset_type, const std::string &dataset_path,
                               DType_t io_dtype_ = DType_t::FP32);

  /**
   * Get list of available dataset types
   */
  static Vec<std::string> available_loaders() {
    return {"mnist", "cifar10", "cifar100", "tiny_imagenet", "imagenet100"};
  }
};

}  // namespace tunx