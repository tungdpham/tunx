// Isolated conv1-only forward pass: loads the exact conv1 weight and input dumped by
// dump_utils.py, runs a single Conv2D op (no bias, stride 2, pad 3, 7x7 kernel matching
// ResNet50's stem conv) and dumps the raw output for comparison against PyTorch's isolated
// conv1 output. This removes every other op (BN/ReLU/maxpool/residual blocks) from the
// comparison so any divergence can only come from the conv itself (cuDNN plan/algorithm,
// reduction order) rather than accumulated depth.
#include <iostream>
#include <string>

#include "device/device_allocator.hpp"
#include "device/device_manager.hpp"
#include "equivalence_utils.hpp"
#include "nn/graph.hpp"
#include "nn/graph_executor.hpp"
#include "nn/layers_impl/conv2d.hpp"

using namespace tunx;

int main(int argc, char** argv) {
  std::string pt_dir = "";
  std::string tunx_dir = "";
  size_t batch_size = 1;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--pt-dir" && i + 1 < argc) {
      pt_dir = argv[++i];
    } else if (arg == "--tunx-dir" && i + 1 < argc) {
      tunx_dir = argv[++i];
    } else if (arg == "--batch-size" && i + 1 < argc) {
      batch_size = std::stoi(argv[++i]);
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  }

  if (pt_dir.empty() || tunx_dir.empty()) {
    std::cerr << "Usage: " << argv[0] << " --pt-dir <dir> --tunx-dir <dir> [--batch-size <N>]"
              << std::endl;
    return 1;
  }

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

  // Matches ResNet50's stem conv exactly: in=3, out=64, 7x7, stride 2, pad 3, no bias
  // fused (PyTorch's conv1 has no bias either since it's followed by BN).
  Conv2D conv1(3, 64, 7, 7, 2, 2, 3, 3, /*use_bias=*/false, "conv1");

  Graph graph;
  Node input_node = graph.make_node("input");
  graph.set_input(input_node);
  Node output_node = conv1(input_node);
  output_node->set_uid("output");
  graph.set_output(output_node);
  graph.compile(allocator);

  std::string weight_path = pt_dir + "/conv1.weight.bin";
  if (!file_exists(weight_path)) {
    std::cerr << "Missing " << weight_path << std::endl;
    return 1;
  }
  load_tensor_bin(conv1.params()[0].data(), weight_path);

  Tensor inputs({batch_size, 224, 224, 3}, DType_t::FP32, allocator);
  load_tensor_bin(inputs, pt_dir + "/inputs.bin");

  GraphExecutor executor(graph);
  TensorBundle input_tensors{{"input", inputs}};
  TensorBundle outputs = executor.forward(input_tensors);
  cudaDeviceSynchronize();

  Tensor output = outputs.get("output");

  double sum_sq = 0.0;
  Tensor host_output = to_host(output);
  const float* data = host_output.data_as<float>();
  for (size_t i = 0; i < host_output.size(); ++i) {
    sum_sq += static_cast<double>(data[i]) * static_cast<double>(data[i]);
  }
  std::cout << "conv1 (isolated) output L2 norm: " << std::sqrt(sum_sq) << std::endl;

  mkdir(tunx_dir.c_str(), 0777);
  save_tensor_bin(output, tunx_dir + "/conv1_isolated.act.bin");

  std::cout << "Dump complete: " << tunx_dir << "/conv1_isolated.act.bin" << std::endl;

  return 0;
}
