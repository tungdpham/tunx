// Isolated per-block PyTorch-vs-TunX equivalence harness.
#include <sys/stat.h>

#include <iostream>
#include <string>

#include "device/device_allocator.hpp"
#include "device/device_manager.hpp"
#include "equivalence_utils.hpp"
#include "nn/graph.hpp"
#include "nn/graph_builder.hpp"
#include "nn/graph_executor.hpp"
#include "nn/optimizers.hpp"

using namespace tunx;

namespace {

Vec<std::string> param_suffixes(size_t count) {
  if (count == 1) return {"weight"};
  if (count == 2) return {"weight", "bias"};
  if (count == 8) {
    return {"q_proj.weight", "q_proj.bias", "k_proj.weight",   "k_proj.bias",
            "v_proj.weight", "v_proj.bias", "out_proj.weight", "out_proj.bias"};
  }
  throw std::runtime_error("Unexpected param count: " + std::to_string(count));
}

// Dump files on disk are always plain FP32 (matching PyTorch's dumps); blocks that require
// BF16 storage (attention/gpt2, for cuDNN's SDPA) transparently cast through FP32 here.
void load_tensor_bin_cast(Tensor& tensor, const std::string& path) {
  if (tensor.dtype() == DType_t::FP32) {
    load_tensor_bin(tensor, path);
    return;
  }
  Tensor fp32_tensor(tensor.shape(), DType_t::FP32, tensor.device());
  load_tensor_bin(fp32_tensor, path);
  cast(fp32_tensor, tensor);
}

void save_tensor_bin_cast(const Tensor& tensor, const std::string& path) {
  if (tensor.dtype() == DType_t::FP32) {
    save_tensor_bin(tensor, path);
    return;
  }
  Tensor fp32_tensor(tensor.shape(), DType_t::FP32, tensor.device());
  cast(tensor, fp32_tensor);
  save_tensor_bin(fp32_tensor, path);
}

void load_params(Graph& graph, const std::string& pt_dir) {
  for (auto& edge : graph.edges()) {
    auto layer = edge->layer();
    auto params = layer->params();
    // Only trainable params (e.g. BatchNorm's non-trainable running-stat buffers are excluded).
    Vec<size_t> trainable;
    for (size_t i = 0; i < params.size(); ++i) {
      if (params[i].requires_grad()) trainable.push_back(i);
    }
    if (trainable.empty()) continue;
    auto suffixes = param_suffixes(trainable.size());
    for (size_t j = 0; j < trainable.size(); ++j) {
      std::string path = pt_dir + "/" + layer->name() + "." + suffixes[j] + ".bin";
      if (file_exists(path)) {
        load_tensor_bin_cast(params[trainable[j]].data(), path);
      } else {
        std::cerr << "Warning: Missing " << path << std::endl;
      }
    }
  }
}

void dump_params(Graph& graph, const std::string& tunx_dir, const std::string& suffix) {
  for (auto& edge : graph.edges()) {
    auto layer = edge->layer();
    auto params = layer->params();
    Vec<size_t> trainable;
    for (size_t i = 0; i < params.size(); ++i) {
      if (params[i].requires_grad()) trainable.push_back(i);
    }
    if (trainable.empty()) continue;
    auto suffixes = param_suffixes(trainable.size());
    for (size_t j = 0; j < trainable.size(); ++j) {
      const Param& p = params[trainable[j]];
      const Tensor& t = suffix == "grad" ? p.grad() : p.data();
      save_tensor_bin_cast(
          t, tunx_dir + "/" + layer->name() + "." + suffixes[j] + "." + suffix + ".bin");
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string block = "residual";
  std::string pt_dir = "";
  std::string tunx_dir = "";
  size_t batch_size = 2;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--block" && i + 1 < argc) {
      block = argv[++i];
    } else if (arg == "--pt-dir" && i + 1 < argc) {
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
    std::cerr << "Usage: " << argv[0]
              << " --block <residual|inception|attention|gpt2> --pt-dir <dir> --tunx-dir <dir> "
                 "[--batch-size <N>]"
              << std::endl;
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

  using Shape = Vec<size_t>;

  // cuDNN's SDPA kernel only supports half/bf16 inputs, so attention/gpt2 blocks must
  // compile with BF16 io/param dtypes (compute stays FP32 for accumulation).
  bool needs_bf16 = (block == "attention" || block == "gpt2");
  GraphOpts opts;
  if (needs_bf16) {
    opts.io_dtype = DType_t::BF16;
    opts.param_dtype = DType_t::BF16;
    opts.compute_dtype = DType_t::FP32;
  }
  DType_t io_dtype = needs_bf16 ? DType_t::BF16 : DType_t::FP32;

  Graph graph;
  Node input = graph.input("input");
  Shape shape;
  Vec<size_t> input_shape;

  std::cout << "Building block: " << block << std::endl;
  if (block == "residual") {
    input_shape = {batch_size, 56, 56, 64};
    shape = input_shape;
    Node output =
        bottleneck_residual_block(input, shape, /*mid=*/64, /*out=*/256, /*stride=*/1, "block");
    output->set_uid("output");
    graph.set_output(output);
  } else if (block == "inception") {
    input_shape = {batch_size, 28, 28, 64};
    shape = input_shape;
    Node output = inception_block(input, shape, /*out_channels=*/32, "block");
    output->set_uid("output");
    graph.set_output(output);
  } else if (block == "attention") {
    input_shape = {batch_size, 16, 64};
    shape = input_shape;
    Node output =
        attention(input, shape, /*embed_dim=*/64, /*num_heads=*/4, /*is_causal=*/true, "block");
    output->set_uid("output");
    graph.set_output(output);
  } else if (block == "gpt2") {
    input_shape = {batch_size, 16, 64};
    shape = input_shape;
    Node output = gpt_block(input, shape, /*embed_dim=*/64, /*num_heads=*/4, /*ffn_dim=*/256,
                            /*dropout_rate=*/0.0f, /*is_causal=*/true, "block");
    output->set_uid("output");
    graph.set_output(output);
  } else {
    std::cerr << "Unknown block: " << block << std::endl;
    return 1;
  }

  graph.compile(allocator, opts);

  std::cout << "Loading initial parameters from " << pt_dir << std::endl;
  load_params(graph, pt_dir);

  Tensor inputs = Tensor(input_shape, io_dtype, allocator);
  load_tensor_bin_cast(inputs, pt_dir + "/inputs.bin");

  GraphExecutor executor(graph);
  TensorBundle input_tensors{{"input", inputs}};

  std::cout << "Running forward pass..." << std::endl;
  TensorBundle outputs = executor.forward(input_tensors);
  Tensor block_output = outputs.get("output");

  std::cout << "Dumping outputs..." << std::endl;
  save_tensor_bin_cast(block_output, tunx_dir + "/outputs.bin");

  std::cout << "Running backward pass..." << std::endl;
  Tensor grad_output = Tensor(block_output.shape(), block_output.dtype(), allocator);
  load_tensor_bin_cast(grad_output, pt_dir + "/grad_output.bin");

  std::shared_ptr<Optimizer> optimizer = std::make_shared<Adam>(1e-3, 0.9, 0.999, 1e-8, 3e-4);
  optimizer->attach(graph);
  optimizer->zero_grads();

  TensorBundle output_grads{{"output", grad_output}};
  executor.backward(output_grads);
  cudaDeviceSynchronize();

  std::cout << "Dumping gradients..." << std::endl;
  dump_params(graph, tunx_dir, "grad");

  std::cout << "Running optimizer step..." << std::endl;
  optimizer->update();

  std::cout << "Dumping updated parameters..." << std::endl;
  dump_params(graph, tunx_dir, "updated");

  std::cout << "Dump complete for block " << block << " in " << tunx_dir << std::endl;

  return 0;
}
