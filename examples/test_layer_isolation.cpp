// Compare a single layer using shared PyTorch inputs, parameters and upstream gradients.
#include <filesystem>

#include "device/device_allocator.hpp"
#include "device/device_manager.hpp"
#include "equivalence_utils.hpp"
#include "nn/engines/cuda_engine.hpp"
#include "nn/engines/cudnn_engine.hpp"
#include "nn/graph.hpp"
#include "nn/graph_executor.hpp"
#include "nn/layers_impl/batchnorm.hpp"
#include "nn/layers_impl/conv2d.hpp"
#include "nn/layers_impl/dense.hpp"
#include "nn/layers_impl/maxpool2d.hpp"
#include "nn/layers_impl/relu.hpp"
using namespace tunx;
int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: test_layer_isolation CASE ENGINE INPUT_DIR OUTPUT_DIR\n";
    return 1;
  }
  std::string kind = argv[1], engine = argv[2], source = argv[3], dest = argv[4];
  std::filesystem::create_directories(dest);
  auto& manager = DeviceManager::instance();
  auto ids = manager.get_all();
  auto id = ids.at(0);
  for (auto candidate : ids)
    if (manager.get(candidate).device_type() == DeviceType::CUDA) {
      id = candidate;
      break;
    }
  auto& allocator = DeviceAllocator::instance(manager.get(id));
  Graph graph;
  auto input = graph.input("input");
  Node output;
  Vec<size_t> shape = {8, 56, 56, 64};
  if (kind == "conv_stem") {
    shape = {8, 224, 224, 3};
    output = Conv2D(3, 64, 7, 7, 2, 2, 3, 3, false, "layer")(input);
  } else if (kind == "conv3")
    output = Conv2D(64, 64, 3, 3, 1, 1, 1, 1, false, "layer")(input);
  else if (kind == "conv1")
    output = Conv2D(64, 256, 1, 1, 1, 1, 0, 0, false, "layer")(input);
  else if (kind == "batchnorm" || kind == "batchnorm_relu")
    output = BatchNorm(64, 1e-5f, 0.1f, true, kind == "batchnorm_relu", "layer")(input);
  else if (kind == "batchnorm_relu_odd") {
    shape = {8, 56, 56, 63};
    output = BatchNorm(63, 1e-5f, 0.1f, true, true, "layer")(input);
  } else if (kind == "maxpool" || kind == "maxpool_relu") {
    shape = {8, 112, 112, 64};
    output = MaxPool2D(3, 3, 2, 2, 1, 1, "layer")(input);
  } else if (kind == "relu")
    output = ReLU("layer")(input);
  else if (kind == "dense") {
    shape = {8, 2048};
    output = Dense(2048, 100, true, "layer")(input);
  } else {
    std::cerr << "Unknown case\n";
    return 1;
  }
  output->set_uid("output");
  graph.set_output(output);
  GraphOpts opts;
  if (engine == "cuda")
    opts.engine = make_engine<CUDAEngine>();
  else if (engine == "cudnn")
    opts.engine = make_engine<CuDNNEngine>();
  else if (engine != "default")
    return 1;
  graph.compile(allocator, opts);
  for (auto& edge : graph.edges()) {
    auto params = edge->layer()->params();
    size_t i = 0;
    for (auto& param : params)
      if (param.requires_grad()) {
        load_tensor_bin(param.data(), source + (i++ == 0 ? "/weight.bin" : "/bias.bin"));
        fill(param.grad(), 0);
      }
  }
  Tensor x(shape, DType_t::FP32, allocator);
  load_tensor_bin(x, source + "/inputs.bin");
  GraphExecutor executor(graph);
  executor.set_backward_grad_hook([&](const Edge&, const std::map<std::string, Tensor>&,
                                      const std::map<std::string, Tensor>& grads) {
    for (const auto& [uid, t] : grads) save_tensor_bin(t, dest + "/input_grad.bin");
  });
  TensorBundle inputs{{"input", x}};
  auto outputs = executor.forward(inputs);
  auto y = outputs.get("output");
  save_tensor_bin(y, dest + "/outputs.bin");
  Tensor dy(y.shape(), DType_t::FP32, allocator);
  load_tensor_bin(dy, source + "/grad_output.bin");
  TensorBundle grads{{"output", dy}};
  executor.backward(grads);
  cudaDeviceSynchronize();
  for (auto& edge : graph.edges()) {
    size_t i = 0;
    for (auto& param : edge->layer()->params())
      if (param.requires_grad())
        save_tensor_bin(param.grad(), dest + (i++ == 0 ? "/weight_grad.bin" : "/bias_grad.bin"));
  }
}
