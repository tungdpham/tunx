// A simulated bruteforce to find best execution order.
#include <bits/stdc++.h>

#include <cstddef>

size_t global_mem = 0;
size_t max_global_mem = 0;

void set_global_mem(size_t new_global_mem) {
  global_mem = new_global_mem;
  max_global_mem = std::max(max_global_mem, global_mem);
}

class Tensor {
private:
  struct Impl {
    size_t size;

    Impl(size_t size)
        : size(size) {
      set_global_mem(global_mem + size);
    }

    ~Impl() { set_global_mem(global_mem - size); }
  };

  std::shared_ptr<Impl> impl_;

public:
  Tensor()
      : impl_(nullptr) {}

  Tensor(size_t size)
      : impl_(std::make_shared<Impl>(size)) {}

  operator bool() { return impl_ != nullptr; }

  size_t size() const { return impl_->size; }
};

// activation nodes
class ActNode {
private:
  std::string uuid_;
  size_t size_;

  // populated as OpNode are added
  size_t out_deg_;

  // populated on run
  Tensor data_;
  size_t out_ref_count_;

public:
  ActNode(std::string uuid, size_t size)
      : uuid_(uuid),
        size_(size),
        out_deg_(0),
        out_ref_count_(0) {}

  void increase_out_deg() { out_deg_++; }

  // for run
  void allocate_data() {
    data_ = Tensor(size_);
    out_ref_count_ = out_deg_;
  }

  // for undo_run
  void deallocate_data() { data_ = Tensor(); }

  Tensor data() { return data_; }

  std::string uuid() const { return uuid_; }

  size_t size() const { return size_; }

  size_t out_ref_count() const { return out_ref_count_; }

  size_t out_deg() const { return out_deg_; }

  // for undo_run
  void increase_ref_count() {
    if (out_ref_count_ == 0) {
      data_ = Tensor(size_);
    }
    out_ref_count_++;
  }

  // for run
  void decrement_ref_count() {
    out_ref_count_--;
    if (out_ref_count_ == 0) {
      data_ = Tensor();
    }
  }
};

// operation nodes
class OpNode {
private:
  std::string uuid_;
  size_t workspace_req_;
  Tensor workspace_;
  std::vector<ActNode*> inputs_;
  std::vector<ActNode*> outputs_;
  std::vector<ActNode*> cache_;
  std::vector<Tensor> cached_data_;

public:
  OpNode(std::string uuid, size_t workspace_req, const std::vector<ActNode*>& inputs,
         const std::vector<ActNode*>& outputs, const std::vector<ActNode*>& cache)
      : uuid_(uuid),
        workspace_req_(workspace_req),
        inputs_(inputs),
        outputs_(outputs),
        cache_(cache) {
    for (auto* t : inputs) {
      t->increase_out_deg();
    }
  }

  std::string uuid() const { return uuid_; }

  size_t workspace_req() const { return workspace_req_; }

  const std::vector<ActNode*>& inputs() const { return inputs_; }

  const std::vector<ActNode*>& outputs() const { return outputs_; }

  // simulate run
  void run() {
    // check input tensors availability
    for (auto* t : inputs_) {
      if (!t->data()) {
        throw std::runtime_error("Input tensor " + t->uuid() + " is not available.");
      }
    }

    // SECTION: memory allocation
    // allocate workspace
    workspace_ = Tensor(workspace_req_);

    for (auto* t : outputs_) {
      t->allocate_data();
    }

    for (auto* t : cache_) {
      cached_data_.push_back(t->data());
    }

    // some execution here in real code

    // SECTION: clean up
    // decrement ref count for inputs
    for (auto* t : inputs_) {
      t->decrement_ref_count();
    }
    // free workspace
    workspace_ = Tensor();
  }

  void undo_run() {
    // SECTION: check output ref count = out deg
    // to ensure all OpNode childrens of the activation node have been reverted prior to this
    for (auto* t : outputs_) {
      if (t->out_ref_count() != t->out_deg()) {
        throw std::runtime_error("Output tensor " + t->uuid() + " has unexpected ref count " +
                                 std::to_string(t->out_ref_count()) + " (expected " +
                                 std::to_string(t->out_deg()) + ").");
      }
    }

    // SECTION: undo stuffs
    cached_data_.clear();

    // deallocate output tensors
    for (auto* t : outputs_) {
      t->deallocate_data();
    }

    // increment ref count for inputs
    for (auto* t : inputs_) {
      t->increase_ref_count();
    }
  }
};

class Graph {
private:
  std::map<std::string, OpNode> op_nodes_;
  std::map<std::string, ActNode> act_nodes_;
  std::vector<ActNode*> inputs_;
  std::vector<ActNode*> outputs_;

public:
  ActNode* add_act(std::string uuid, size_t size) {
    act_nodes_.insert({uuid, ActNode(uuid, size)});
    return &act_nodes_.at(uuid);
  }

  OpNode* add_node(std::string uuid, size_t workspace_req, const std::vector<ActNode*>& inputs,
                   const std::vector<ActNode*>& outputs, const std::vector<ActNode*>& cache = {}) {
    op_nodes_.insert({uuid, OpNode(uuid, workspace_req, inputs, outputs, cache)});
    return &op_nodes_.at(uuid);
  }

  void set_input_tensors(ActNode* node) { inputs_.push_back(node); }

  void set_output_tensors(ActNode* node) {
    // increase out deg by 1 to avoid premature free since users need them.
    node->increase_out_deg();
    outputs_.push_back(node);
  }

  size_t simulate_and_print(const std::string& name, const std::vector<std::string>& order) {
    // Allocate graph input tensors
    for (auto* act : inputs_) {
      act->allocate_data();
    }
    size_t prev_max = max_global_mem;
    max_global_mem = global_mem;  // Reset peak memory tracking for this specific simulation

    std::cout << "--- " << name << " Order Step-by-Step Memory Profile ---\n";
    std::cout << std::left << std::setw(15) << "Step (Op)" << std::right << std::setw(20)
              << "Current Mem (B)" << std::setw(20) << "Peak Mem (B)" << "\n";
    std::cout << std::string(55, '-') << "\n";

    std::cout << std::left << std::setw(15) << "(Initial state)" << std::right << std::setw(20)
              << global_mem << std::setw(20) << max_global_mem << "\n";

    for (const auto& op_id : order) {
      op_nodes_.at(op_id).run();
      std::cout << std::left << std::setw(15) << op_id << std::right << std::setw(20) << global_mem
                << std::setw(20) << max_global_mem << "\n";
    }

    size_t peak = max_global_mem;

    // Undo to restore state
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      op_nodes_.at(*it).undo_run();
    }

    // Deallocate inputs
    for (auto* act : inputs_) {
      act->deallocate_data();
    }

    max_global_mem = prev_max;  // Restore old peak memory
    std::cout << "\n";
    return peak;
  }

  void rank_execution_orders(
      const std::vector<std::pair<std::string, std::vector<std::string>>>& named_orders) {
    std::cout << "=== Execution Order Memory Efficiency Ranking ===\n";

    struct OrderResult {
      std::string name;
      size_t peak_mem;
      std::vector<std::string> order;
    };

    std::vector<OrderResult> results;
    for (const auto& [name, order] : named_orders) {
      if (order.empty()) continue;
      // Allocate graph input tensors
      for (auto* act : inputs_) {
        act->allocate_data();
      }
      size_t prev_max = max_global_mem;
      max_global_mem = global_mem;

      for (const auto& op_id : order) {
        op_nodes_.at(op_id).run();
      }
      size_t peak = max_global_mem;

      for (auto it = order.rbegin(); it != order.rend(); ++it) {
        op_nodes_.at(*it).undo_run();
      }
      for (auto* act : inputs_) {
        act->deallocate_data();
      }
      max_global_mem = prev_max;

      results.push_back({name, peak, order});
    }

    std::sort(results.begin(), results.end(),
              [](const OrderResult& a, const OrderResult& b) { return a.peak_mem < b.peak_mem; });

    if (results.empty()) return;

    size_t best_peak = results.front().peak_mem;

    for (size_t i = 0; i < results.size(); ++i) {
      const auto& res = results[i];
      std::cout << i + 1 << ". " << res.name << " Order\n";
      std::cout << "   Peak Memory: " << res.peak_mem << " bytes\n";
      if (res.peak_mem > best_peak) {
        double efficiency = 100.0 * (static_cast<double>(best_peak) / res.peak_mem);
        double savings = 100.0 * (1.0 - static_cast<double>(best_peak) / res.peak_mem);
        std::cout << "   Compared to best: +" << (res.peak_mem - best_peak) << " bytes overhead ("
                  << std::fixed << std::setprecision(1) << savings << "% potential savings)\n";
        std::cout << "   Memory efficiency: " << std::fixed << std::setprecision(1) << efficiency
                  << "%\n";
      } else {
        std::cout << "   Compared to best: Optimal (100.0% efficiency)\n";
      }
      std::cout << "   Order: ";
      for (size_t j = 0; j < res.order.size(); j++) {
        std::cout << res.order[j];
        if (j + 1 < res.order.size()) std::cout << " -> ";
      }
      std::cout << "\n\n";
    }
  }

  std::vector<std::string> find_simple_execution_order() {
    std::vector<std::string> op_ids;
    for (auto& [id, _] : op_nodes_) {
      op_ids.push_back(id);
    }

    std::map<std::string, std::string> tensor_producer;
    for (auto& [op_id, node] : op_nodes_) {
      for (auto* t : node.outputs()) {
        tensor_producer[t->uuid()] = op_id;
      }
    }

    std::map<std::string, std::set<std::string>> deps;
    std::map<std::string, std::set<std::string>> dependents;
    for (auto& [op_id, node] : op_nodes_) {
      deps[op_id] = {};
      for (auto* t : node.inputs()) {
        auto it = tensor_producer.find(t->uuid());
        if (it != tensor_producer.end()) {
          deps[op_id].insert(it->second);
        }
      }
    }
    for (auto& [op_id, dep_set] : deps) {
      for (auto& dep : dep_set) {
        dependents[dep].insert(op_id);
      }
    }

    std::map<std::string, int> in_degree;
    for (auto& id : op_ids) {
      in_degree[id] = static_cast<int>(deps[id].size());
    }

    std::vector<std::string> order;
    std::set<std::string> executed;

    size_t prev_max = max_global_mem;
    for (auto* act : inputs_) {
      act->allocate_data();
    }
    max_global_mem = global_mem;

    auto get_score = [&](const std::string& id) {
      auto& node = op_nodes_.at(id);
      long long workspace = node.workspace_req();
      long long memory_consumes = 0;
      for (auto* t : node.inputs()) {
        if (t->out_ref_count() == 1) {
          memory_consumes += t->size();
        }
      }
      return workspace + memory_consumes;
    };

    struct CompareScore {
      bool operator()(const std::pair<long long, std::string>& a,
                      const std::pair<long long, std::string>& b) const {
        if (a.first != b.first) return a.first < b.first;
        return a.second > b.second;
      }
    };

    std::priority_queue<std::pair<long long, std::string>,
                        std::vector<std::pair<long long, std::string>>, CompareScore>
        pq;
    std::set<std::string> ready_set;

    for (auto& id : op_ids) {
      if (in_degree[id] == 0) {
        ready_set.insert(id);
      }
    }

    for (auto& id : ready_set) {
      pq.push({get_score(id), id});
    }

    while (!pq.empty()) {
      auto [score, best_op] = pq.top();
      pq.pop();

      if (!ready_set.count(best_op)) continue;

      ready_set.erase(best_op);
      executed.insert(best_op);
      order.push_back(best_op);
      op_nodes_.at(best_op).run();

      for (auto& dep : dependents[best_op]) {
        in_degree[dep]--;
        if (in_degree[dep] == 0) {
          ready_set.insert(dep);
        }
      }

      // Re-evaluate scores for all ready nodes as they might have increased
      for (auto& id : ready_set) {
        pq.push({get_score(id), id});
      }
    }

    if (order.size() < op_ids.size()) {
      throw std::runtime_error("Graph has a cycle or unresolved dependencies.");
    }

    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      op_nodes_.at(*it).undo_run();
    }
    for (auto* act : inputs_) {
      act->deallocate_data();
    }
    max_global_mem = prev_max;

    return order;
  }

  // bruteforce on permutation of valid execution orders to find one with best memory efficiency
  // a valid execution order is a topological sort of the graph.
  std::pair<std::vector<std::string>, std::vector<std::string>>
  find_minimum_memory_execution_order() {
    // Collect all op UUIDs.
    std::vector<std::string> op_ids;
    for (auto& [id, _] : op_nodes_) {
      op_ids.push_back(id);
    }

    // Build a map: act node uuid -> the op that produces it (as an output).
    std::map<std::string, std::string> tensor_producer;
    for (auto& [op_id, node] : op_nodes_) {
      for (auto* t : node.outputs()) {
        tensor_producer[t->uuid()] = op_id;
      }
    }

    // Build dependency graph between ops.
    // op A depends on op B if any of A's inputs are produced by B.
    std::map<std::string, std::set<std::string>> deps;        // op -> set of ops it depends on
    std::map<std::string, std::set<std::string>> dependents;  // op -> set of ops that depend on it
    for (auto& [op_id, node] : op_nodes_) {
      deps[op_id] = {};
      for (auto* t : node.inputs()) {
        auto it = tensor_producer.find(t->uuid());
        if (it != tensor_producer.end()) {
          deps[op_id].insert(it->second);
        }
      }
    }
    for (auto& [op_id, dep_set] : deps) {
      for (auto& dep : dep_set) {
        dependents[dep].insert(op_id);
      }
    }

    // Compute in-degree for each op (number of unresolved dependencies).
    std::map<std::string, int> in_degree;
    for (auto& id : op_ids) {
      in_degree[id] = static_cast<int>(deps[id].size());
    }

    // Track the best (minimum peak memory) and worst ordering found so far.
    size_t best_peak = std::numeric_limits<size_t>::max();
    size_t worst_peak = 0;
    std::vector<std::string> best_order;
    std::vector<std::string> worst_order;
    int num_orderings = 0;

    // Backtracking enumeration of all valid topological orderings.
    // Uses OpNode::run()/undo_run() to simulate memory allocation inline,
    // avoiding the need for a separate simulate pass.
    std::vector<std::string> current_order;
    std::set<std::string> executed;

    size_t prev_max = max_global_mem;
    // Allocate graph input tensors before starting the search.
    for (auto* act : inputs_) {
      act->allocate_data();
    }
    max_global_mem = global_mem;

    std::function<void()> backtrack = [&]() {
      if (current_order.size() == op_ids.size()) {
        num_orderings++;
        if (max_global_mem < best_peak) {
          best_peak = max_global_mem;
          best_order = current_order;
        }
        if (max_global_mem > worst_peak) {
          worst_peak = max_global_mem;
          worst_order = current_order;
        }
        return;
      }

      for (auto& id : op_ids) {
        if (executed.count(id)) continue;
        if (in_degree[id] != 0) continue;

        // Choose this op: run it (allocates outputs, frees consumed inputs).
        executed.insert(id);
        current_order.push_back(id);
        for (auto& dep : dependents[id]) {
          in_degree[dep]--;
        }

        size_t saved_max = max_global_mem;
        op_nodes_.at(id).run();

        backtrack();

        // Unchoose: undo the op (restores inputs, deallocates outputs).
        op_nodes_.at(id).undo_run();
        max_global_mem = saved_max;

        current_order.pop_back();
        executed.erase(id);
        for (auto& dep : dependents[id]) {
          in_degree[dep]++;
        }
      }
    };

    backtrack();

    // Print results.
    std::cout << "=== Memory-Optimal Execution Order Search ===\n";
    std::cout << "Total valid topological orderings explored: " << num_orderings << "\n\n";

    // Deallocate graph input tensors after the search.
    for (auto* act : inputs_) {
      act->deallocate_data();
    }
    max_global_mem = prev_max;

    return {best_order, worst_order};
  }
};

size_t random_act_size() {
  int rand_val = rand() % 100;
  if (rand_val < 30) return 1280;
  if (rand_val < 60) return 2560;
  if (rand_val < 80) return 5120;
  if (rand_val < 90) return 10240;
  return 4096;
}

ActNode* build_random_diamond_dag(Graph& g, ActNode* input, int depth, int& node_counter) {
  if (depth == 0) {
    return input;
  }

  std::string prefix = "d" + std::to_string(depth) + "_" + std::to_string(node_counter++);

  int structure_type = rand() % 3;

  if (structure_type == 0) {
    // Sequential
    auto act = g.add_act(prefix + "_seq_act", 1000 + rand() % 1000);
    g.add_node(prefix + "_seq_conv", 500 + rand() % 500, {input}, {act});
    return build_random_diamond_dag(g, act, depth - 1, node_counter);
  } else if (structure_type == 1) {
    // Diamond
    auto left_act = g.add_act(prefix + "_left_act", 1000 + rand() % 1000);
    g.add_node(prefix + "_left_conv", 500 + rand() % 500, {input}, {left_act});
    auto left_out = build_random_diamond_dag(g, left_act, depth - 1, node_counter);

    auto right_act = g.add_act(prefix + "_right_act", 1000 + rand() % 1000);
    g.add_node(prefix + "_right_conv", 500 + rand() % 500, {input}, {right_act});
    auto right_out = build_random_diamond_dag(g, right_act, depth - 1, node_counter);

    auto merge_act = g.add_act(prefix + "_merge_act", 1000 + rand() % 1000);
    g.add_node(prefix + "_add", 100 + rand() % 100, {left_out, right_out}, {merge_act});

    return merge_act;
  } else {
    // Residual (skip connection)
    auto main_act = g.add_act(prefix + "_main_act", 1000 + rand() % 1000);
    g.add_node(prefix + "_main_conv", 500 + rand() % 500, {input}, {main_act});
    auto main_out = build_random_diamond_dag(g, main_act, depth - 1, node_counter);

    auto merge_act = g.add_act(prefix + "_res_add_act", 1000 + rand() % 1000);
    g.add_node(prefix + "_res_add", 100 + rand() % 100, {main_out, input}, {merge_act});

    return merge_act;
  }
}

Graph sample_graph() {
  Graph g;

  // Register all activation nodes with sizes (simulating feature map sizes in bytes).
  auto input = g.add_act("input", 10000);
  auto a1 = g.add_act("a1", 2000);
  auto a2 = g.add_act("a2", 2000);
  auto a3 = g.add_act("a3", 1500);
  auto a4 = g.add_act("a4", 1500);

  auto b1 = g.add_act("b1", 2000);
  auto b2 = g.add_act("b2", 2000);
  auto b3 = g.add_act("b3", 1500);
  auto b4 = g.add_act("b4", 1500);

  auto c1 = g.add_act("c1", 1000);
  auto c2 = g.add_act("c2", 500);
  auto c3 = g.add_act("c3", 500);
  auto c4 = g.add_act("c4", 1000);

  auto merged = g.add_act("merged", 3000);
  auto output = g.add_act("output", 500);

  g.set_input_tensors(input);
  g.set_output_tensors(output);

  // A path
  g.add_node("a_conv1", 500, {input}, {a1});
  g.add_node("a_add1", 100, {a1, input}, {a2});

  g.add_node("a_conv2", 400, {a2}, {a3}, {a2});
  g.add_node("a_add2", 100, {a3, a2}, {a4});

  // B path
  g.add_node("b_conv1", 500, {input}, {b1});
  g.add_node("b_add1", 100, {b1, input}, {b2});

  g.add_node("b_conv2", 400, {b2}, {b3}, {b2});
  g.add_node("b_add2", 100, {b3, b2}, {b4});

  // C path
  g.add_node("c_conv1", 500, {input}, {c1});
  g.add_node("c_conv2", 500, {c1}, {c2});
  g.add_node("c_conv3", 500, {c1}, {c3});
  g.add_node("c_aggregate", 400, {c2, c3}, {c4});

  // Merge: concatenate outputs from towers
  g.add_node("merge", 200, {a4, b4, c4}, {merged});

  // Fully-connected classification head
  g.add_node("fc", 300, {merged}, {output});

  return g;
}

Graph random_graph(int depth, int& node_counter) {
  Graph g;
  auto input = g.add_act("input", 10000);
  g.set_input_tensors(input);
  auto output = build_random_diamond_dag(g, input, depth, node_counter);
  g.set_output_tensors(output);
  return g;
}

signed main() {
  int trials;
  std::cin >> trials;

  while (trials--) {
    global_mem = 0;
    max_global_mem = 0;

    int node_counter = 0;
    Graph g = random_graph(3, node_counter);

    auto simple_order = g.find_simple_execution_order();

    auto [best_order, worst_order] = g.find_minimum_memory_execution_order();

    g.rank_execution_orders(
        {{"BEST", best_order}, {"SIMPLE", simple_order}, {"WORST", worst_order}});

    if (!best_order.empty()) {
      g.simulate_and_print("BEST", best_order);
    }
    if (!worst_order.empty()) {
      g.simulate_and_print("WORST", worst_order);
    }
    g.simulate_and_print("SIMPLE", simple_order);
  }

  return 0;
}
