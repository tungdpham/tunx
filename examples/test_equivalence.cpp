#include <sys/stat.h>

#include <iostream>
#include <string>
#include <vector>

#include "device/device_allocator.hpp"
#include "device/device_manager.hpp"
#include "equivalence_utils.hpp"
#include "nn/example_graphs.hpp"
#include "nn/graph.hpp"
#include "nn/graph_executor.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"

using namespace tunx;

int main(int argc, char** argv) {
  std::string model_name = "tunx_v1";
  std::string pt_dir = "";
  std::string tunx_dir = "";
  size_t batch_size = 1;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      model_name = argv[++i];
    } else if (arg == "--pt_dir" && i + 1 < argc) {
      pt_dir = argv[++i];
    } else if (arg == "--tunx_dir" && i + 1 < argc) {
      tunx_dir = argv[++i];
    } else if (arg == "--batch_size" && i + 1 < argc) {
      batch_size = std::stoi(argv[++i]);
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  }

  if (pt_dir.empty() || tunx_dir.empty()) {
    std::cerr << "Usage: " << argv[0] << " --model <name> --pt_dir <dir> --tunx_dir <dir> [--batch_size <N>]"
              << std::endl;
    return 1;
  }

  // Ensure output directory exists
  mkdir(tunx_dir.c_str(), 0777);

  initializeDefaultDevices();
  DeviceManager& manager = DeviceManager::instance();
  auto device_ids = manager.get_all();
  if (device_ids.empty()) {
    std::cerr << "No devices found." << std::endl;
    return 1;
  }

  // Find CUDA device if available
  tunx::DeviceID target_device_id = device_ids[0];
  for (auto id : device_ids) {
    if (manager.get(id).device_type() == DeviceType::CUDA) {
      target_device_id = id;
      break;
    }
  }

  Device& device = manager.get(target_device_id);
  IAllocator& allocator = DeviceAllocator::instance(device);

  std::cout << "Creating model " << model_name << " on " << device.get_name() << std::endl;
  ExampleGraphs::register_defaults();

  GraphOpts opts;

  Graph graph = ExampleGraphs::create(model_name, allocator, opts);

  std::cout << "Loading initial parameters from " << pt_dir << std::endl;
  for (auto& edge : graph.edges()) {
    auto layer = edge->layer();
    auto params = layer->params();
    if (params.empty()) continue;

    std::string name = layer->name();

    if (params.size() >= 1) {
      std::string path = pt_dir + "/" + name + ".weight.bin";
      if (file_exists(path)) {
        load_tensor_bin(params[0].data(), path);
      } else {
        std::cerr << "Warning: Missing " << path << std::endl;
      }
    }
    if (params.size() >= 2) {
      std::string path = pt_dir + "/" + name + ".bias.bin";
      if (file_exists(path)) {
        load_tensor_bin(params[1].data(), path);
      } else {
        std::cerr << "Warning: Missing " << path << std::endl;
      }
    }
    // BN also has running mean and var, but in TunX Layer::params() only returns trainable
    // parameters (weight, bias). For full correctness, running stats should be loaded too, but
    // TunX's BatchNorm doesn't expose running_mean in params(). If it diverges, we might need to
    // expose them. For 1 step equivalence, initial running stats are 0 and 1, which both frameworks
    // default to.
  }

  std::cout << "Loading inputs and labels..." << std::endl;
  bool is_lm = (model_name.find("gpt2") != std::string::npos);
  Vec<size_t> input_shape = {batch_size, 224, 224, 3};
  Vec<size_t> label_shape = {batch_size};
  if (is_lm) {
    input_shape = {batch_size, 1024};
    label_shape = {batch_size, 1024};
  }

  Tensor inputs = Tensor(input_shape, DType_t::FP32, allocator);
  load_tensor_bin(inputs, pt_dir + "/inputs.bin");

  Tensor labels;
  if (is_lm) {
    labels = Tensor(label_shape, DType_t::INT32, allocator);
  } else {
    labels = Tensor(label_shape, DType_t::FP32, allocator);
  }
  load_tensor_bin(labels, pt_dir + "/labels.bin");

  std::cout << "Running forward pass..." << std::endl;
  GraphExecutor executor(graph);

  TensorBundle input_tensors{{"input", inputs}};

  TensorBundle outputs = executor.forward(input_tensors);
  Tensor predictions = outputs.get("output");

  std::cout << "Dumping outputs..." << std::endl;
  save_tensor_bin(predictions, tunx_dir + "/outputs.bin");

  std::cout << "Running backward pass..." << std::endl;

  std::shared_ptr<Loss> criterion = std::make_shared<CrossEntropyLoss>();
  float loss;
  criterion->compute_loss(predictions, labels, loss);
  Tensor loss_gradient = Tensor(predictions.shape(), predictions.dtype(), allocator);
  criterion->compute_gradient(predictions, labels, loss_gradient);

  TensorBundle output_grads{{"output", loss_gradient}};

  executor.backward(output_grads);

  std::cout << "Dumping gradients..." << std::endl;

  for (auto& edge : graph.edges()) {
    auto layer = edge->layer();
    auto params = layer->params();
    if (params.empty()) continue;

    std::string name = layer->name();

    if (params.size() >= 1) {
      save_tensor_bin(params[0].grad(), tunx_dir + "/" + name + ".weight.grad.bin");
    }
    if (params.size() >= 2) {
      save_tensor_bin(params[1].grad(), tunx_dir + "/" + name + ".bias.grad.bin");
    }
  }

  std::cout << "Running optimizer step..." << std::endl;
  std::shared_ptr<Optimizer> optimizer = std::make_shared<Adam>(1e-3, 0.9, 0.999, 1e-8, 3e-4);
  optimizer->attach(graph);
  optimizer->update();

  std::cout << "Dumping updated parameters..." << std::endl;
  for (auto& edge : graph.edges()) {
    auto layer = edge->layer();
    auto params = layer->params();
    if (params.empty()) continue;

    std::string name = layer->name();

    if (params.size() >= 1) {
      save_tensor_bin(params[0].data(), tunx_dir + "/" + name + ".weight.updated.bin");
    }
    if (params.size() >= 2) {
      save_tensor_bin(params[1].data(), tunx_dir + "/" + name + ".bias.updated.bin");
    }
  }

  std::cout << "Dump complete for " << model_name << " in " << tunx_dir << std::endl;

  return 0;
}
