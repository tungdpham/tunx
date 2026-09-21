// Replay the final ResNet block or SGD updates with shared PyTorch tensors.
#include <filesystem>

#include "device/device_allocator.hpp"
#include "device/device_manager.hpp"
#include "equivalence_utils.hpp"
#include "nn/engines/cuda_engine.hpp"
#include "nn/engines/cudnn_engine.hpp"
#include "nn/graph_builder.hpp"
#include "nn/graph_executor.hpp"
#include "nn/optimizers.hpp"
using namespace tunx;
int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: test_trajectory_isolation replay|sgd ENGINE SOURCE DEST\n";
    return 1;
  }
  std::string mode = argv[1], engine = argv[2], source = argv[3], dest = argv[4];
  std::filesystem::create_directories(dest);
  auto& manager = DeviceManager::instance();
  auto id = manager.get_all().at(0);
  for (auto candidate : manager.get_all())
    if (manager.get(candidate).device_type() == DeviceType::CUDA) {
      id = candidate;
      break;
    }
  auto& allocator = DeviceAllocator::instance(manager.get(id));
  Graph graph;
  auto input = graph.input("input");
  Shape shape = {8, 7, 7, 2048};
  auto output = bottleneck_residual_block(input, shape, 512, 2048, 1, "layer4_block3");
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
  auto each_param = [&](auto action) {
    for (auto& edge : graph.edges()) {
      if (!edge->layer()) continue;
      size_t i = 0;
      for (auto& p : edge->layer()->params())
        if (p.requires_grad()) {
          auto name = edge->layer()->name() + (i++ == 0 ? ".weight" : ".bias");
          action(p, name);
        }
    }
  };
  each_param([&](auto& p, const auto& name) {
    load_tensor_bin(p.data(), source + "/" + name + ".bin");
    fill(p.grad(), 0);
  });
  if (mode == "sgd") {
    SGD optimizer(1e-3, 0.9, 1e-4);
    optimizer.attach(graph);
    for (int step = 1; step <= 3; ++step) {
      auto step_name = "params_step_" + std::to_string(step);
      each_param([&](auto& p, const auto& name) {
        load_tensor_bin(p.grad(), source + "/" + step_name + "/" + name + ".grad.bin");
      });
      optimizer.update();
      cudaDeviceSynchronize();
      std::filesystem::create_directories(dest + "/" + step_name);
      each_param([&](auto& p, const auto& name) {
        save_tensor_bin(p.data(), dest + "/" + step_name + "/" + name + ".updated.bin");
      });
    }
    return 0;
  }
  if (mode != "replay") return 1;
  GraphExecutor executor(graph);
  executor.set_forward_hook([&](const Edge& edge, const std::map<std::string, Tensor>& tensors) {
    if (!edge->layer()) return;
    for (const auto& [uid, t] : tensors)
      save_tensor_bin(t, dest + "/" + edge->layer()->name() + ".output.bin");
  });
  executor.set_backward_grad_hook([&](const Edge& edge, const std::map<std::string, Tensor>&,
                                      const std::map<std::string, Tensor>& tensors) {
    if (!edge->layer()) return;
    for (const auto& [uid, t] : tensors)
      save_tensor_bin(t, dest + "/" + edge->layer()->name() + ".input_grad.bin");
  });
  Tensor x({8, 7, 7, 2048}, DType_t::FP32, allocator);
  load_tensor_bin(x, source + "/inputs.bin");
  TensorBundle inputs{{"input", x}};
  auto outputs = executor.forward(inputs);
  save_tensor_bin(outputs.get("output"), dest + "/outputs.bin");
  Tensor dy({8, 7, 7, 2048}, DType_t::FP32, allocator);
  load_tensor_bin(dy, source + "/grad_output.bin");
  TensorBundle grads{{"output", dy}};
  executor.backward(grads);
  cudaDeviceSynchronize();
  each_param([&](auto& p, const auto& name) {
    save_tensor_bin(p.grad(), dest + "/" + name + ".grad.bin");
  });
}
