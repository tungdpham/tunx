#include <gtest/gtest.h>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "device/device_manager.hpp"
#include "device/pool_allocator.hpp"
#include "nn/example_graphs.hpp"
#include "nn/graph.hpp"
#include "nn/graph_executor.hpp"
#include "nn/layer_factory.hpp"
#include "nn/tensor_bundle.hpp"
#include "partitioner/graph_partitioner.hpp"
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"

using namespace tunx;

namespace {

Graph build_linear_graph() {
  Graph graph;

  Node input = graph.make_node("input");
  graph.set_input(input);
  Node hidden_a = Dense(8, 8, false, "dense_a")(input);
  hidden_a->set_uid("hidden_a");
  Node hidden_b = Dense(8, 8, false, "dense_b")(hidden_a);
  hidden_b->set_uid("hidden_b");
  Node hidden_c = Dense(8, 8, false, "dense_c")(hidden_b);
  hidden_c->set_uid("hidden_c");
  Node output = Dense(8, 8, false, "dense_d")(hidden_c);
  output->set_uid("output");
  graph.set_output(output);

  return graph;
}

Graph build_branched_graph() {
  Graph graph;

  Node input = graph.make_node("input");
  graph.set_input(input);
  Node left = Dense(8, 8, false, "left_dense")(input);
  left->set_uid("left");
  Node right = Dense(8, 8, false, "right_dense")(input);
  right->set_uid("right");
  Node merged = Add("merge")(left, right);
  merged->set_uid("merged");
  Node output = Dense(8, 8, false, "tail_dense")(merged);
  output->set_uid("output");
  graph.set_output(output);

  return graph;
}

Graph reload_graph(const Graph &graph, IAllocator &allocator) {
  std::stringstream state(std::ios::in | std::ios::out | std::ios::binary);
  graph.save_state(state);
  state.seekg(0);
  return Graph::load_state(state, allocator);
}

TensorBundle make_partition_input_map(const TensorBundle &outputs, const Vec<std::string> &uids) {
  TensorBundle inputs;
  for (const auto &uid : uids) {
    inputs.set(uid, outputs.get(uid));
  }
  return inputs;
}

void expect_tensors_close(const Tensor &lhs, const Tensor &rhs, float tolerance) {
  ASSERT_TRUE(lhs);
  ASSERT_TRUE(rhs);
  ASSERT_EQ(lhs.shape(), rhs.shape());

  Tensor lhs_cpu = to_device(lhs, getHost());
  Tensor rhs_cpu = to_device(rhs, getHost());
  const float *lhs_data = lhs_cpu.data_as<float>();
  const float *rhs_data = rhs_cpu.data_as<float>();
  for (size_t i = 0; i < lhs_cpu.size(); ++i) {
    EXPECT_NEAR(lhs_data[i], rhs_data[i], tolerance) << "Mismatch at index " << i;
  }
}

}  // namespace

class GraphPlannerStateTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    initializeDefaultDevices();
    DeviceManager &manager = DeviceManager::instance();
    Vec<DeviceID> device_ids = manager.get_all();

    has_cpu_ = false;
    for (const DeviceID &id : device_ids) {
      Device &device = manager.get(id);
      if (device.device_type() == DeviceType::CUDA) {
        has_cpu_ = true;
        device_ = device;
        break;
      }
    }

    if (!has_cpu_) {
      GTEST_SKIP() << "No CPU device available, skipping CuDNN engine tests";
    }

    stream_ = device_->default_stream();
  }

  static bool has_cpu_;
  static sref<Device> device_;
  static stream stream_;
};

bool GraphPlannerStateTest::has_cpu_;
sref<Device> GraphPlannerStateTest::device_;
stream GraphPlannerStateTest::stream_;

TEST(GraphPartitionerTest, SplitsLinearGraphIntoRequestedLayerCounts) {
  Graph graph = build_linear_graph();
  GraphPartitioner partitioner({0.25, 0.75});

  std::vector<GraphPartition> partitions = partitioner.partition(graph);

  ASSERT_EQ(partitions.size(), 2);
  EXPECT_EQ(partitions[0].start_layer, 0);
  EXPECT_EQ(partitions[0].layer_count, 1);
  EXPECT_EQ(partitions[0].graph.edges().size(), 1);
  EXPECT_EQ(partitions[0].input_uids, (std::vector<std::string>{"input"}));
  EXPECT_EQ(partitions[0].output_uids, (std::vector<std::string>{"hidden_a"}));

  EXPECT_EQ(partitions[1].start_layer, 1);
  EXPECT_EQ(partitions[1].layer_count, 3);
  EXPECT_EQ(partitions[1].graph.edges().size(), 3);
  EXPECT_EQ(partitions[1].input_uids, (std::vector<std::string>{"hidden_a"}));
  EXPECT_EQ(partitions[1].output_uids, (std::vector<std::string>{"output"}));
}

TEST(GraphPartitionerTest, PreservesBoundaryNodesForBranchedGraph) {
  Graph graph = build_branched_graph();
  GraphPartitioner partitioner({0.5, 0.5});

  std::vector<GraphPartition> partitions = partitioner.partition(graph);

  ASSERT_EQ(partitions.size(), 2);
  EXPECT_EQ(partitions[0].graph.edges().size(), 2);
  EXPECT_EQ(partitions[0].input_uids, (std::vector<std::string>{"input"}));
  EXPECT_EQ(partitions[0].output_uids, (std::vector<std::string>{"left", "right"}));

  EXPECT_EQ(partitions[1].graph.edges().size(), 2);
  EXPECT_EQ(partitions[1].input_uids, (std::vector<std::string>{"left", "right"}));
  EXPECT_EQ(partitions[1].output_uids, (std::vector<std::string>{"output"}));
  EXPECT_EQ(partitions[0].graph.input_uids(), (std::vector<std::string>{"input"}));
  EXPECT_EQ(partitions[0].graph.output_uids(), (std::vector<std::string>{"left", "right"}));
  EXPECT_EQ(partitions[1].graph.input_uids(), (std::vector<std::string>{"left", "right"}));
  EXPECT_EQ(partitions[1].graph.output_uids(), (std::vector<std::string>{"output"}));
}

TEST_F(GraphPlannerStateTest, SaveAndLoadPreservesExplicitBoundaryNodes) {
  auto &allocator = PoolAllocator::instance(device_, stream_);
  Graph graph = build_branched_graph();
  graph.compile(allocator);

  std::stringstream state(std::ios::in | std::ios::out | std::ios::binary);
  graph.save_state(state);
  state.seekg(0);

  Graph loaded = Graph::load_state(state, allocator);

  EXPECT_EQ(loaded.input_uids(), (std::vector<std::string>{"input"}));
  EXPECT_EQ(loaded.output_uids(), (std::vector<std::string>{"output"}));
}

TEST(GraphPartitionerTest, RejectsPartitionRatiosThatResolveToEmptyPartitions) {
  Graph graph = build_linear_graph();
  GraphPartitioner partitioner({0.95, 0.05, 0.01, 0.01, 0.01});

  EXPECT_THROW(partitioner.partition(graph), std::runtime_error);
}

TEST_F(GraphPlannerStateTest, PartitionedResNet9MatchesFullGraphForwardAndBackward) {
  auto &allocator = PoolAllocator::instance(getGPU(), getGPU().default_stream());
  ExampleGraphs::register_defaults();

  Graph full_graph = ExampleGraphs::create("cifar10_resnet9", allocator);
  GraphPartitioner partitioner({2, 1});
  std::vector<GraphPartition> partitions = partitioner.partition(full_graph);

  ASSERT_EQ(partitions.size(), 2u);
  EXPECT_EQ(partitions[0].output_uids.size(), 2u);
  EXPECT_EQ(partitions[1].input_uids.size(), 2u);
  EXPECT_TRUE(std::find(partitions[0].output_uids.begin(), partitions[0].output_uids.end(),
                        "node_25") != partitions[0].output_uids.end());
  EXPECT_TRUE(std::find(partitions[0].output_uids.begin(), partitions[0].output_uids.end(),
                        "node_28") != partitions[0].output_uids.end());
  EXPECT_TRUE(std::find(partitions[1].input_uids.begin(), partitions[1].input_uids.end(),
                        "node_25") != partitions[1].input_uids.end());
  EXPECT_TRUE(std::find(partitions[1].input_uids.begin(), partitions[1].input_uids.end(),
                        "node_28") != partitions[1].input_uids.end());

  Graph stage0 = reload_graph(partitions[0].graph, allocator);
  Graph stage1 = reload_graph(partitions[1].graph, allocator);

  Tensor input = Tensor({1, 32, 32, 3}, DType_t::FP32, getHost());
  fill_uniform(input, -1.0f, 1.0f, 12345ULL);

  TensorBundle full_inputs{{"input", to_device(input, full_graph.device())}};
  GraphExecutor full_executor(full_graph);
  TensorBundle full_outputs = full_executor.forward(full_inputs);

  TensorBundle stage0_inputs{{partitions[0].input_uids.front(), to_device(input, stage0.device())}};
  GraphExecutor stage0_executor(stage0);
  TensorBundle stage0_outputs = stage0_executor.forward(stage0_inputs);
  TensorBundle stage1_inputs = make_partition_input_map(stage0_outputs, partitions[1].input_uids);
  GraphExecutor stage1_executor(stage1);
  TensorBundle stage1_outputs = stage1_executor.forward(stage1_inputs);

  expect_tensors_close(full_outputs.get("output"),
                       stage1_outputs.get(partitions[1].output_uids.front()), 1e-4f);

  Tensor grad_output = Tensor({1, 10}, DType_t::FP32, getHost());
  fill_uniform(grad_output, -1.0f, 1.0f, 12345ULL);

  TensorBundle full_output_grads{{"output", to_device(grad_output, full_graph.device())}};
  TensorBundle full_grad_inputs = full_executor.backward(full_output_grads);

  TensorBundle stage1_output_grads{
      {partitions[1].output_uids.front(), to_device(grad_output, stage1.device())}};
  TensorBundle stage1_grad_inputs = stage1_executor.backward(stage1_output_grads);
  TensorBundle stage0_output_grads =
      make_partition_input_map(stage1_grad_inputs, partitions[0].output_uids);
  TensorBundle stage0_grad_inputs = stage0_executor.backward(stage0_output_grads);

  expect_tensors_close(full_grad_inputs.get("input"),
                       stage0_grad_inputs.get(partitions[0].input_uids.front()), 1e-4f);
}

TEST_F(GraphPlannerStateTest, BackwardAccumulatesGradientsAcrossFanOut) {
  auto &allocator = PoolAllocator::instance(device_, stream_);

  Graph graph;
  {
    Node input = graph.make_node("input");
    graph.set_input(input);

    auto left_dense = Dense(1, 1, false, "left_dense");
    auto right_dense = Dense(1, 1, false, "right_dense");

    Node left = left_dense(input);
    left->set_uid("left");
    Node right = right_dense(input);
    right->set_uid("right");
    Node output = Add("merge")(left, right);
    output->set_uid("output");
    graph.set_output(output);
    graph.compile(allocator);
    fill(left_dense.params()[0].data(), 2.0f);
    fill(right_dense.params()[0].data(), 3.0f);
  }

  Tensor input_tensor = Tensor({1, 1}, DType_t::FP32, device_);
  fill(input_tensor, 1.0f);
  TensorBundle inputs{{"input", input_tensor}};

  GraphExecutor executor(graph);
  TensorBundle outputs = executor.forward(inputs);
  Tensor output = outputs.get("output");
  output = to_host(output);
  EXPECT_NEAR(output.data_as<float>()[0], 5.0f, 1e-5f);

  Tensor grad_output = Tensor({1, 1}, DType_t::FP32, device_);
  fill(grad_output, 1.0f);
  TensorBundle output_grads{{"output", grad_output}};

  TensorBundle grad_inputs = executor.backward(output_grads);
  Tensor grad_input = grad_inputs.get("input");
  grad_input = to_host(grad_input);
  EXPECT_NEAR(grad_input.data_as<float>()[0], 5.0f, 1e-5f);
}

TEST_F(GraphPlannerStateTest, BackwardClearsAccumulatedGradientsBetweenPasses) {
  auto &allocator = PoolAllocator::instance(device_, stream_);

  Graph graph;
  Node input = graph.make_node("input");
  graph.set_input(input);

  auto left_dense = Dense(1, 1, false, "left_dense");
  auto right_dense = Dense(1, 1, false, "right_dense");

  Node left = left_dense(input);
  left->set_uid("left");
  Node right = right_dense(input);
  right->set_uid("right");
  Node output = Add("merge")(left, right);
  output->set_uid("output");
  graph.set_output(output);
  graph.compile(allocator);

  fill(left_dense.params()[0].data(), 2.0f);
  fill(right_dense.params()[0].data(), 3.0f);

  GraphExecutor executor(graph);

  Tensor first_input = Tensor({1, 1}, DType_t::FP32, device_);
  fill(first_input, 1.0f);
  TensorBundle first_inputs{{"input", first_input}};
  executor.forward(first_inputs);

  Tensor first_grad_output = Tensor({1, 1}, DType_t::FP32, device_);
  fill(first_grad_output, 1.0f);
  TensorBundle first_output_grads{{"output", first_grad_output}};
  TensorBundle first_grad_inputs = executor.backward(first_output_grads);
  Tensor grad_input = to_host(first_grad_inputs.get("input"));
  EXPECT_NEAR(grad_input.data_as<float>()[0], 5.0f, 1e-5f);

  Tensor second_input = Tensor({2, 1}, DType_t::FP32, device_);
  fill(second_input, 1.0f);
  TensorBundle second_inputs{{"input", second_input}};
  executor.forward(second_inputs);

  Tensor second_grad_output = Tensor({2, 1}, DType_t::FP32, device_);
  fill(second_grad_output, 1.0f);
  TensorBundle second_output_grads{{"output", second_grad_output}};
  TensorBundle second_grad_inputs = executor.backward(second_output_grads);
  Tensor second_grad_input = to_host(second_grad_inputs.get("input"));
  for (size_t i = 0; i < second_grad_input.size(); ++i) {
    EXPECT_NEAR(second_grad_input.data_as<float>()[i], 5.0f, 1e-5f);
  }
}
