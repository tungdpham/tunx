#include <cuda_runtime.h>
#include <sys/stat.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <cstdlib>

#include "data_loading/imagenet100_dataset.hpp"
#include "data_loading/open_webtext_dataset.hpp"

#include "device/device_allocator.hpp"
#include "device/device_manager.hpp"
#include "equivalence_utils.hpp"
#include "nn/engines/cuda_engine.hpp"
#include "nn/engines/cudnn_engine.hpp"
#include "nn/example_graphs.hpp"
#include "nn/graph.hpp"
#include "nn/graph_executor.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"

using namespace tunx;

int main(int argc, char** argv) {
  std::string model_name = "resnet50";
  std::string pt_dir = "";
  std::string tunx_dir = "";
  size_t batch_size = 8;
  size_t steps = 1000;
  int seed = 42;
  std::string executor_mode = "optimized";
  std::string engine_mode = "default";
  bool no_aug = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      model_name = argv[++i];
    } else if (arg == "--pt-dir" && i + 1 < argc) {
      pt_dir = argv[++i];
    } else if (arg == "--tunx-dir" && i + 1 < argc) {
      tunx_dir = argv[++i];
    } else if (arg == "--batch-size" && i + 1 < argc) {
      batch_size = std::stoi(argv[++i]);
    } else if (arg == "--steps" && i + 1 < argc) {
      steps = std::stoi(argv[++i]);
    } else if (arg == "--seed" && i + 1 < argc) {
      seed = std::stoi(argv[++i]);
    } else if (arg == "--no-aug") {
      no_aug = true;
    } else if (arg == "--executor-mode" && i + 1 < argc) {
      executor_mode = argv[++i];
    } else if (arg == "--engine" && i + 1 < argc) {
      engine_mode = argv[++i];
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  }

  if (pt_dir.empty() || tunx_dir.empty()) {
    std::cerr << "Usage: " << argv[0]
              << " --model <name> --pt-dir <dir> --tunx-dir <dir> [--batch-size <N>] [--steps <N>] "
                 "[--seed <N>] [--no-aug]"
              << " [--executor-mode optimized|naive|linear|branching|joining]"
              << " [--engine default|cuda|cudnn]" << std::endl;
    return 1;
  }

  mkdir(tunx_dir.c_str(), 0777);

  DeviceManager& manager = DeviceManager::instance();
  auto device_ids = manager.get_all();
  if (device_ids.empty()) {
    std::cerr << "No devices found." << std::endl;
    return 1;
  }

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
  opts.seed = seed;
  if (engine_mode == "cuda") {
    opts.engine = make_engine<CUDAEngine>();
  } else if (engine_mode == "cudnn") {
    opts.engine = make_engine<CuDNNEngine>();
  } else if (engine_mode != "default") {
    std::cerr << "Unknown engine: " << engine_mode << std::endl;
    return 1;
  }

  std::string actual_model_name = model_name;
  if (model_name == "resnet50") {
    actual_model_name = "imagenet100_resnet50";
  } else if (model_name == "gpt2") {
    actual_model_name = "gpt2_small";
  }

  Graph graph = ExampleGraphs::create(actual_model_name, allocator, opts);

  std::cout << "Loading initial parameters from " << pt_dir << std::endl;
  for (auto& edge : graph.edges()) {
    auto layer = edge->layer();
    auto params = layer->params();
    if (params.empty()) continue;

    std::string name = layer->name();

    bool is_attn = (name.find("attn") != std::string::npos);
    if (is_attn && (params.size() == 8 || params.size() == 4)) {
      bool has_bias = (params.size() == 8);
      const char* suffixes_w[] = {".q.weight.bin", ".k.weight.bin", ".v.weight.bin",
                                  ".out.weight.bin"};
      const char* suffixes_b[] = {".q.bias.bin", ".k.bias.bin", ".v.bias.bin", ".out.bias.bin"};
      for (size_t i = 0; i < 4; ++i) {
        size_t w_idx = has_bias ? (i * 2) : i;
        std::string w_path = pt_dir + "/" + name + suffixes_w[i];
        if (file_exists(w_path)) load_tensor_bin(params[w_idx].data(), w_path);

        if (has_bias) {
          std::string b_path = pt_dir + "/" + name + suffixes_b[i];
          if (file_exists(b_path)) load_tensor_bin(params[w_idx + 1].data(), b_path);
        }
      }
    } else {
      if (params.size() >= 1) {
        std::string path = pt_dir + "/" + name + ".weight.bin";
        if (file_exists(path)) load_tensor_bin(params[0].data(), path);
      }
      if (params.size() >= 2) {
        std::string path = pt_dir + "/" + name + ".bias.bin";
        if (file_exists(path)) load_tensor_bin(params[1].data(), path);
      }
    }
  }

  bool is_lm = (model_name.find("gpt2") != std::string::npos);
  Vec<size_t> single_input_shape = {batch_size, 224, 224, 3};
  Vec<size_t> single_label_shape = {batch_size};
  if (is_lm) {
    single_input_shape = {batch_size, 1024};
    single_label_shape = {batch_size, 1024};
  }

  std::unique_ptr<Dataset> dataset;
  if (is_lm) {
    auto ds = std::make_unique<OpenWebText>(1024, DType_t::INT32);
    const char* path = std::getenv("OPENWEBTEXT_PATH");
    std::string root = path ? path : "data/open-web-text/train.bin";
    if (root.length() > 10 && root.substr(root.length() - 10) == "/train.bin") {
        root = root.substr(0, root.length() - 10);
    }
    if (!ds->load_data(root)) {
        std::cerr << "Failed to load OpenWebText from " << root << std::endl;
        return 1;
    }
    dataset = std::move(ds);
  } else {
    auto ds = std::make_unique<ImageNet100>(DType_t::FP32);
    const char* path = std::getenv("IMAGENET100_ROOT");
    std::string root = path ? path : "data/imagenet-100";
    if (!ds->load_data(root, true)) {
        std::cerr << "Failed to load ImageNet100 from " << root << std::endl;
        return 1;
    }
    if (no_aug) ds->set_disable_augmentation(true);
    dataset = std::move(ds);
  }

  // Load indices
  std::string indices_path = pt_dir + "/indices_trajectory.bin";
  Vec<size_t> all_indices;
  {
    std::ifstream is(indices_path, std::ios::binary);
    if (!is) {
       std::cerr << "Cannot open " << indices_path << std::endl;
       return 1;
    }
    is.seekg(0, std::ios::end);
    size_t size = is.tellg();
    is.seekg(0, std::ios::beg);
    size_t num_indices = size / sizeof(int32_t);
    std::vector<int32_t> temp_indices(num_indices);
    is.read((char*)temp_indices.data(), size);
    for (int32_t idx : temp_indices) all_indices.push_back(idx);
  }
  
  if (all_indices.size() < steps * batch_size) {
      std::cerr << "Not enough indices for requested steps and batch size!" << std::endl;
      return 1;
  }

  Tensor inputs;
  if (is_lm) {
    inputs = Tensor(single_input_shape, DType_t::INT32, allocator);
  } else {
    inputs = Tensor(single_input_shape, DType_t::FP32, allocator);
  }

  Tensor labels = Tensor(single_label_shape, DType_t::INT32, allocator);

  GraphExecutor executor(graph);
  SolverOptions solver_options;
  if (executor_mode == "naive") {
    solver_options = SolverOptions{true, false, false, false};
  } else if (executor_mode == "linear") {
    solver_options = SolverOptions{false, true, false, false};
  } else if (executor_mode == "branching") {
    solver_options = SolverOptions{false, true, true, false};
  } else if (executor_mode == "joining") {
    solver_options = SolverOptions{false, true, false, true};
  } else {
    solver_options = SolverOptions{false, true, false, false};
  }

  TensorBundle input_tensors{{"input", inputs}};
  executor.build_plans(input_tensors, solver_options);

  std::shared_ptr<Loss> criterion = std::make_shared<CrossEntropyLoss>();
  std::shared_ptr<Optimizer> optimizer = std::make_shared<Adam>(1e-3, 0.9, 0.999, 1e-8, 3e-4);
  optimizer->attach(graph);

  std::string csv_path = tunx_dir + "/tunx_trajectory_seed_" + std::to_string(seed) + ".csv";
  std::ofstream csv_file(csv_path);
  if (is_lm) {
    csv_file << "step,loss,perplexity\n";
  } else {
    csv_file << "step,loss,accuracy\n";
  }

  size_t input_bytes = inputs.num_bytes();
  size_t label_bytes = labels.num_bytes();
  auto stream = executor.graph().handle().get_stream();

  std::cout << "Running multi-step convergence training (" << steps << " steps)..." << std::endl;

  for (size_t step = 0; step < steps; ++step) {
    optimizer->zero_grads();

    // Copy step inputs and labels
    Vec<size_t> step_indices;
    for (size_t i = 0; i < batch_size; ++i) {
       step_indices.push_back(all_indices[step * batch_size + i]);
    }

    Tensor inputs_host, labels_host;
    if (!dataset->get_batch_by_indices(step_indices, inputs_host, labels_host)) {
        std::cerr << "Failed to get batch for step " << step << std::endl;
        return 1;
    }

    if (device.device_type() == DeviceType::CUDA) {
      cudaMemcpy(inputs.data_as<void>(), inputs_host.data_as<void>(),
                 input_bytes, cudaMemcpyHostToDevice);
      cudaMemcpy(labels.data_as<void>(), labels_host.data_as<void>(),
                 label_bytes, cudaMemcpyHostToDevice);
    } else {
      memcpy(inputs.data_as<void>(), inputs_host.data_as<void>(),
             input_bytes);
      memcpy(labels.data_as<void>(), labels_host.data_as<void>(),
             label_bytes);
    }

    TensorBundle outputs = executor.forward(input_tensors);
    Tensor predictions = outputs.get("output");

    float loss = 0.0f;
    criterion->compute_loss(predictions, labels, loss);

    // Compute accuracy/perplexity
    if (is_lm) {
      float ppl = std::exp(std::min(loss, 20.0f));
      csv_file << (step + 1) << "," << loss << "," << ppl << "\n";
    } else {
      // Calculate accuracy
      Tensor preds_host = to_host(predictions);
      Tensor labels_host = to_host(labels);
      const float* p_data = preds_host.data_as<float>();
      const int* l_data = labels_host.data_as<int>();

      int correct = 0;
      int num_classes = predictions.shape()[1];
      for (size_t b = 0; b < batch_size; ++b) {
        int best_class = 0;
        float best_val = -1e9f;
        for (int c = 0; c < num_classes; ++c) {
          float val = p_data[b * num_classes + c];
          if (val > best_val) {
            best_val = val;
            best_class = c;
          }
        }
        if (best_class == l_data[b]) {
          correct++;
        }
      }
      float acc = 100.0f * correct / batch_size;
      csv_file << (step + 1) << "," << loss << "," << acc << "\n";
    }

    if ((step + 1) % 100 == 0) {
      std::cout << "Step " << (step + 1) << "/" << steps << " | Loss: " << loss << std::endl;
    }

    Tensor loss_gradient = Tensor(predictions.shape(), predictions.dtype(), allocator);
    criterion->compute_gradient(predictions, labels, loss_gradient);

    TensorBundle output_grads{{"output", loss_gradient}};
    executor.backward(output_grads);

    if (device.device_type() == DeviceType::CUDA) cudaDeviceSynchronize();

    optimizer->update();
  }

  csv_file.close();
  std::cout << "Convergence test complete. Results saved to " << csv_path << std::endl;

  return 0;
}
