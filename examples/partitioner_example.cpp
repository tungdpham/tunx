#include <iostream>
#include <memory>
#include <vector>

#include "device/device_manager.hpp"
#include "device/pool_allocator.hpp"
#include "nn/example_graphs.hpp"
#include "nn/graph.hpp"
#define private public
#include "nn/graph_executor.hpp"
#undef private
#include "partitioner/graph_partitioner.hpp"
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"

using namespace std;
using namespace tunx;

int main() {
  cout << "Testing ComputeBandwidthPartitioner with CIFAR10 ResNet-9" << endl;

  ExampleGraphs::register_defaults();
  Device& device = getGPU();
  auto& allocator = PoolAllocator::instance(device, device.default_stream());
  
  GraphOpts opts{
      .io_dtype = DType_t::BF16,
      .param_dtype = DType_t::BF16,
  };

  Graph graph = load_or_create_graph("cifar10_resnet9", "", allocator, opts);
  GraphExecutor executor(graph);

  // Setup input (batch size 256, 3 channels, 32x32)
  Tensor input_tensor({256, 32, 32, 3}, DType_t::BF16, device);
  fill(input_tensor, 0.5f, device.default_stream());
  TensorBundle input_map({{"input", input_tensor}});

  cout << "Warming up (5 steps)..." << endl;
  for (int i = 0; i < 5; ++i) {
    executor.forward(input_map);
  }

  cout << "Running measured step..." << endl;
  auto [output_map, edge_profiles, node_profiles] = executor.profile_edges_forward(input_map);
  
  // Total work will be the sum of exec_time for all edges on the baseline machine (Machine 1)
  double total_baseline_time = 0.0;
  for (const auto& [edge, profile] : edge_profiles) {
    total_baseline_time += profile.exec_time;
  }
  cout << "Baseline execution time (sum of edges): " << total_baseline_time << " ms" << endl;

  // Setup: 
  // Machine 1 has double the compute power of Machine 2.
  // Machine 1 compute power = 2.0 (baseline machine where we measured)
  // Machine 2 compute power = 1.0 (half as fast)
  // Interconnect speed = 3 GB/s = 3,000,000 bytes / ms.
  DeviceMesh mesh;
  mesh.compute_powers = {2.0, 1.0}; 
  mesh.link_speeds = {3000000.0};   

  auto compute_cost_fn = [&edge_profiles](const Edge& edge) -> double {
    auto it = edge_profiles.find(edge);
    if (it != edge_profiles.end()) {
      // The measured time is on Machine 1, which has power 2.0.
      // We return the "work units" such that work / power = time.
      // So work_units = time * 2.0
      return it->second.exec_time * 2.0;
    }
    return 1.0; 
  };

  // The activation size for each boundary tensor (batch size 256, dim 1024, FP32)
  auto activation_size_fn = [&node_profiles](const Node& node) -> double {
    auto it = node_profiles.find(node);
    if (it != node_profiles.end()) {
      return static_cast<double>(it->second);
    }
    return 1048576.0; 
  };

  ComputeBandwidthPartitioner partitioner(mesh, compute_cost_fn, activation_size_fn);

  auto partitions = partitioner.partition(graph);

  cout << "Partitioning completed!" << endl;
  cout << "Total partitions: " << partitions.size() << endl;

  for (size_t i = 0; i < partitions.size(); ++i) {
    cout << "\nMachine " << i + 1 << ":" << endl;
    cout << "- Start layer index: " << partitions[i].start_layer << endl;
    cout << "- Number of layers: " << partitions[i].layer_count << endl;
    
    cout << "- Inputs: ";
    for (const auto& uid : partitions[i].input_uids) {
      cout << uid << " ";
    }
    cout << "\n- Outputs: ";
    for (const auto& uid : partitions[i].output_uids) {
      cout << uid << " ";
    }
    cout << endl;
  }

  return 0;
}
