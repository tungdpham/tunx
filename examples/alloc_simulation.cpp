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

  std::pair<std::map<std::string, std::set<std::string>>,
            std::map<std::string, std::set<std::string>>>
  get_dependencies() const {
    std::map<std::string, std::string> tensor_producer;
    for (const auto& [op_id, node] : op_nodes_) {
      for (auto* t : node.outputs()) {
        tensor_producer[t->uuid()] = op_id;
      }
    }

    std::map<std::string, std::set<std::string>> deps;
    std::map<std::string, std::set<std::string>> dependents;
    for (const auto& [op_id, node] : op_nodes_) {
      deps[op_id] = {};
      for (auto* t : node.inputs()) {
        auto it = tensor_producer.find(t->uuid());
        if (it != tensor_producer.end()) {
          deps[op_id].insert(it->second);
        }
      }
    }
    for (const auto& [op_id, dep_set] : deps) {
      for (const auto& dep : dep_set) {
        dependents[dep].insert(op_id);
      }
    }
    return {deps, dependents};
  }

  void print_graph() const {
    std::cout << "--- Graph Structure ---\n";
    std::cout << "Inputs: ";
    for (auto* act : inputs_) std::cout << act->uuid() << " ";
    std::cout << "\nOutputs: ";
    for (auto* act : outputs_) std::cout << act->uuid() << " ";
    std::cout << "\n\nOperations (Topological Order):\n";

    auto [deps, dependents] = get_dependencies();

    std::map<std::string, int> in_degree;
    for (const auto& [id, _] : op_nodes_) {
      in_degree[id] = static_cast<int>(deps.at(id).size());
    }

    std::vector<std::string> order;
    std::queue<std::string> q;
    for (const auto& [id, deg] : in_degree) {
      if (deg == 0) q.push(id);
    }

    while (!q.empty()) {
      std::string curr = q.front();
      q.pop();
      order.push_back(curr);
      for (const auto& next : dependents[curr]) {
        in_degree[next]--;
        if (in_degree[next] == 0) {
          q.push(next);
        }
      }
    }

    // Fallback for any disconnected/cyclic nodes (though shouldn't happen here)
    if (order.size() != op_nodes_.size()) {
      for (const auto& [id, _] : op_nodes_) {
        if (std::find(order.begin(), order.end(), id) == order.end()) {
          order.push_back(id);
        }
      }
    }

    for (const auto& id : order) {
      const auto& op = op_nodes_.at(id);
      std::cout << "  " << id << " (workspace: " << op.workspace_req() << ")";
      std::cout << ", Inputs: ";
      for (auto* t : op.inputs()) std::cout << t->uuid() << " (size: " << t->size() << ")";
      std::cout << ", Outputs: ";
      for (auto* t : op.outputs()) std::cout << t->uuid() << " (size: " << t->size() << ")";
      std::cout << "\n";
    }
    std::cout << "-----------------------\n";
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

  std::map<std::string, double> rank_execution_orders(
      const std::vector<std::pair<std::string, std::vector<std::string>>>& named_orders) {
    std::map<std::string, double> efficiencies;
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

    if (results.empty()) return efficiencies;

    size_t best_peak = results.front().peak_mem;

    for (size_t i = 0; i < results.size(); ++i) {
      const auto& res = results[i];
      double efficiency = 100.0;
      std::cout << i + 1 << ". " << res.name << " Order\n";
      std::cout << "   Peak Memory: " << res.peak_mem << " bytes\n";
      if (res.peak_mem > best_peak) {
        efficiency = 100.0 * (static_cast<double>(best_peak) / res.peak_mem);
        double savings = 100.0 * (1.0 - static_cast<double>(best_peak) / res.peak_mem);
        std::cout << "   Compared to best: +" << (res.peak_mem - best_peak) << " bytes overhead ("
                  << std::fixed << std::setprecision(1) << savings << "% potential savings)\n";
        std::cout << "   Memory efficiency: " << std::fixed << std::setprecision(1) << efficiency
                  << "%\n";
      } else {
        std::cout << "   Compared to best: Optimal (100.0% efficiency)\n";
      }
      efficiencies[res.name] = efficiency;
      std::cout << "   Order: ";
      for (size_t j = 0; j < res.order.size(); j++) {
        std::cout << res.order[j];
        if (j + 1 < res.order.size()) std::cout << " -> ";
      }
      std::cout << "\n\n";
    }
    return efficiencies;
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

    struct MacroNode {
      std::string id;
      std::vector<std::string> ops;
      long long a;
      long long b;
    };

    std::map<std::string, MacroNode> macros;
    std::map<std::string, std::set<std::string>> macro_deps;
    std::map<std::string, std::set<std::string>> macro_dependents;

    for (auto& id : op_ids) {
      auto& node = op_nodes_.at(id);
      long long all_outputs = 0;
      for (auto* t : node.outputs()) {
        all_outputs += t->size();
      }

      long long workspace = node.workspace_req();
      long long total_memory_for_execution = all_outputs + workspace;

      long long memory_generate = all_outputs;
      long long memory_consumes = 0;
      for (auto* t : node.inputs()) {
        if (t->out_ref_count() == 1) {
          memory_consumes += t->size();
        }
      }

      MacroNode mn;
      mn.id = id;
      mn.ops = {id};
      mn.a = total_memory_for_execution;
      mn.b = memory_generate - memory_consumes;
      macros[id] = mn;

      macro_deps[id] = deps[id];
      macro_dependents[id] = dependents[id];
    }

    auto compare = [](const MacroNode& i, const MacroNode& j) {
      long long cost_ij = std::max(i.a, i.b + j.a);
      long long cost_ji = std::max(j.a, j.b + i.a);
      if (cost_ij != cost_ji) return cost_ij < cost_ji;
      if (i.b != j.b) return i.b < j.b;
      return i.a < j.a;
    };

    int next_macro_id = 0;
    while (true) {
      std::string best_Y = "";
      std::string best_X = "";

      for (auto& [Y_id, Y_node] : macros) {
        if (macro_deps[Y_id].size() == 1) {
          std::string X_id = *macro_deps[Y_id].begin();
          auto& X_node = macros[X_id];

          if (compare(Y_node, X_node)) {
            if (best_Y == "" || compare(Y_node, macros[best_Y])) {
              best_Y = Y_id;
              best_X = X_id;
            }
          }
        }
      }
      for (auto& [X_id, dependents_set] : macro_dependents) {
        if (dependents_set.size() == 1) {
          std::string Y_id = *dependents_set.begin();
          if (compare(macros[Y_id], macros[X_id])) {
            if (best_Y == "" || compare(macros[Y_id], macros[best_Y])) {
              best_Y = Y_id;
              best_X = X_id;
            }
          }
        }
      }

      if (best_Y == "") break;

      std::string XY_id = "macro_" + std::to_string(next_macro_id++);
      MacroNode XY;
      XY.id = XY_id;
      XY.ops = macros[best_X].ops;
      XY.ops.insert(XY.ops.end(), macros[best_Y].ops.begin(), macros[best_Y].ops.end());
      XY.a = std::max(macros[best_X].a, macros[best_X].b + macros[best_Y].a);
      XY.b = macros[best_X].b + macros[best_Y].b;

      macros[XY_id] = XY;

      macro_deps[XY_id] = macro_deps[best_X];
      for (auto& dep : macro_deps[best_Y]) {
        if (dep != best_X) macro_deps[XY_id].insert(dep);
      }

      macro_dependents[XY_id] = macro_dependents[best_X];
      macro_dependents[XY_id].erase(best_Y);
      for (auto& child : macro_dependents[best_Y]) {
        macro_dependents[XY_id].insert(child);
      }

      for (auto& parent : macro_deps[XY_id]) {
        macro_dependents[parent].erase(best_X);
        macro_dependents[parent].erase(best_Y);
        macro_dependents[parent].insert(XY_id);
      }
      for (auto& child : macro_dependents[XY_id]) {
        macro_deps[child].erase(best_X);
        macro_deps[child].erase(best_Y);
        macro_deps[child].insert(XY_id);
      }

      macros.erase(best_X);
      macros.erase(best_Y);
      macro_deps.erase(best_X);
      macro_deps.erase(best_Y);
      macro_dependents.erase(best_X);
      macro_dependents.erase(best_Y);
    }

    std::vector<std::string> final_order;
    std::set<std::string> executed_macros;
    std::vector<std::string> ready_macros;

    for (auto& [id, deps_set] : macro_deps) {
      if (deps_set.empty()) {
        ready_macros.push_back(id);
      }
    }

    while (final_order.size() < op_ids.size()) {
      if (ready_macros.empty()) {
        throw std::runtime_error("Graph has a cycle or unresolved dependencies.");
      }

      std::string best_macro = ready_macros[0];
      int best_idx = 0;
      for (size_t i = 1; i < ready_macros.size(); ++i) {
        if (compare(macros[ready_macros[i]], macros[best_macro])) {
          best_macro = ready_macros[i];
          best_idx = static_cast<int>(i);
        }
      }

      executed_macros.insert(best_macro);
      ready_macros.erase(ready_macros.begin() + best_idx);

      for (auto& op : macros[best_macro].ops) {
        final_order.push_back(op);
      }

      for (auto& child : macro_dependents[best_macro]) {
        bool ready = true;
        for (auto& parent : macro_deps[child]) {
          if (!executed_macros.count(parent)) {
            ready = false;
            break;
          }
        }
        if (ready) {
          ready_macros.push_back(child);
        }
      }
    }

    for (auto* act : inputs_) {
      act->allocate_data();
    }
    for (auto& op : final_order) {
      op_nodes_.at(op).run();
    }
    for (auto it = final_order.rbegin(); it != final_order.rend(); ++it) {
      op_nodes_.at(*it).undo_run();
    }
    for (auto* act : inputs_) {
      act->deallocate_data();
    }

    return final_order;
  }

  // Optimized search for the execution order with minimum peak memory.
  // Uses branch-and-bound pruning, heuristic-guided candidate ordering,
  // bitmask memoization, and iterative DFS to avoid stack overflow.
  std::vector<std::string> find_minimum_memory_execution_order() {
    // Collect all op UUIDs and assign bit indices.
    std::vector<std::string> op_ids;
    std::map<std::string, int> op_index;  // op uuid -> bit index
    for (auto& [id, _] : op_nodes_) {
      op_index[id] = static_cast<int>(op_ids.size());
      op_ids.push_back(id);
    }
    const int n = static_cast<int>(op_ids.size());
    if (n > 256) {
      throw std::runtime_error("Too many ops for bitmask memoization (max 256).");
    }

    auto [deps, dependents] = get_dependencies();

    // Compute in-degree for each op (number of unresolved dependencies).
    std::map<std::string, int> in_degree;
    for (auto& id : op_ids) {
      in_degree[id] = static_cast<int>(deps[id].size());
    }

    // Heuristic scoring: prefer ops that free the most memory relative to workspace cost.
    // Lower score = more memory freed = more promising to explore first.
    auto get_score = [&](const std::string& id) -> long long {
      auto& node = op_nodes_.at(id);
      long long score = static_cast<long long>(node.workspace_req());
      // Subtract memory freed by inputs at last reference (will be deallocated).
      for (auto* t : node.inputs()) {
        if (t->out_ref_count() == 1) {
          score -= static_cast<long long>(t->size());
        }
      }
      // Add memory consumed by outputs.
      for (auto* t : node.outputs()) {
        score += static_cast<long long>(t->size());
      }
      return score;
    };

    // Memoization: executed_bitmask -> best peak memory reaching this state.
    // If we reach the same set of executed ops with a worse peak, we can prune.
    std::unordered_map<std::bitset<256>, size_t> memo;

    // Track the best (minimum peak memory) ordering found so far.
    // Seed with the heuristic ordering to enable aggressive pruning from the start.
    size_t best_peak = std::numeric_limits<size_t>::max();
    std::vector<std::string> best_order;
    int num_orderings = 0;
    int num_pruned = 0;

    // --- Seed best_peak with the greedy heuristic solution ---
    {
      auto heuristic_order = find_simple_execution_order();
      if (!heuristic_order.empty()) {
        // Simulate to get the peak memory of the heuristic solution.
        for (auto* act : inputs_) {
          act->allocate_data();
        }
        size_t saved = max_global_mem;
        max_global_mem = global_mem;
        for (const auto& op_id : heuristic_order) {
          op_nodes_.at(op_id).run();
        }
        best_peak = max_global_mem;
        best_order = heuristic_order;
        num_orderings = 1;  // Count heuristic as first explored ordering.
        // Undo
        for (auto it = heuristic_order.rbegin(); it != heuristic_order.rend(); ++it) {
          op_nodes_.at(*it).undo_run();
        }
        for (auto* act : inputs_) {
          act->deallocate_data();
        }
        max_global_mem = saved;
      }
    }

    // --- Iterative DFS with explicit stack ---
    // Each stack frame stores the state needed for backtracking.
    struct Frame {
      std::vector<std::string> candidates;  // sorted ready ops for this level
      int candidate_idx;                    // which candidate we're currently trying
      std::string chosen_op;                // the op we chose (empty if not yet chosen)
      size_t saved_max;                     // max_global_mem before running this op
      std::bitset<256> executed_mask;       // bitmask of executed ops before this frame
    };

    std::vector<std::string> current_order;
    std::set<std::string> executed;
    std::bitset<256> executed_mask;

    size_t prev_max = max_global_mem;
    // Allocate graph input tensors before starting the search.
    for (auto* act : inputs_) {
      act->allocate_data();
    }
    max_global_mem = global_mem;

    // Build initial candidates.
    auto get_sorted_candidates = [&]() -> std::vector<std::string> {
      std::vector<std::pair<long long, std::string>> scored;
      for (auto& id : op_ids) {
        if (executed.count(id)) continue;
        if (in_degree[id] != 0) continue;
        scored.push_back({get_score(id), id});
      }
      // Sort ascending: lowest score (most memory-freeing) first.
      std::sort(scored.begin(), scored.end());
      std::vector<std::string> result;
      result.reserve(scored.size());
      for (auto& [s, id] : scored) {
        result.push_back(std::move(id));
      }
      return result;
    };

    std::vector<Frame> stack;
    // Push initial frame.
    stack.push_back({get_sorted_candidates(), 0, "", 0, std::bitset<256>()});

    while (!stack.empty()) {
      auto& frame = stack.back();

      // If we previously chose an op, we need to undo it before trying the next.
      if (!frame.chosen_op.empty()) {
        // Undo the previously chosen op.
        op_nodes_.at(frame.chosen_op).undo_run();
        max_global_mem = frame.saved_max;
        current_order.pop_back();
        executed.erase(frame.chosen_op);
        executed_mask.reset(op_index[frame.chosen_op]);
        for (auto& dep : dependents[frame.chosen_op]) {
          in_degree[dep]++;
        }
        frame.chosen_op.clear();
        frame.candidate_idx++;  // move to next candidate
      }

      // Try the next candidate.
      bool pushed_child = false;
      while (frame.candidate_idx < static_cast<int>(frame.candidates.size())) {
        const auto& id = frame.candidates[frame.candidate_idx];

        // Choose this op.
        executed.insert(id);
        executed_mask.set(op_index[id]);
        current_order.push_back(id);
        for (auto& dep : dependents[id]) {
          in_degree[dep]--;
        }

        frame.saved_max = max_global_mem;
        frame.chosen_op = id;
        op_nodes_.at(id).run();

        // Branch-and-bound: prune if current peak already exceeds best found.
        // Use > (not >=) because max_global_mem is monotonically non-decreasing,
        // so a path at exactly best_peak can still complete but can't improve.
        if (max_global_mem > best_peak) {
          num_pruned++;
          // Undo and try next candidate.
          op_nodes_.at(id).undo_run();
          max_global_mem = frame.saved_max;
          current_order.pop_back();
          executed.erase(id);
          executed_mask.reset(op_index[id]);
          for (auto& dep : dependents[id]) {
            in_degree[dep]++;
          }
          frame.chosen_op.clear();
          frame.candidate_idx++;
          continue;
        }

        // Memoization: check if we've seen this state with better or equal peak.
        auto memo_it = memo.find(executed_mask);
        if (memo_it != memo.end() && memo_it->second <= max_global_mem) {
          num_pruned++;
          // Undo and try next candidate.
          op_nodes_.at(id).undo_run();
          max_global_mem = frame.saved_max;
          current_order.pop_back();
          executed.erase(id);
          executed_mask.reset(op_index[id]);
          for (auto& dep : dependents[id]) {
            in_degree[dep]++;
          }
          frame.chosen_op.clear();
          frame.candidate_idx++;
          continue;
        }
        memo[executed_mask] = max_global_mem;

        // Check if we've completed a full ordering.
        if (static_cast<int>(current_order.size()) == n) {
          num_orderings++;
          if (max_global_mem < best_peak) {
            best_peak = max_global_mem;
            best_order = current_order;
          }
          // Don't push a child frame; the while loop will undo and try next.
          // Advance candidate_idx so the undo at top of loop triggers correctly
          // by leaving chosen_op set — the top-of-loop will undo it.
          break;
        }

        // Push child frame with new candidates.
        stack.push_back({get_sorted_candidates(), 0, "", 0, executed_mask});
        pushed_child = true;
        break;
      }

      // If no more candidates and we didn't push a child, pop this frame.
      if (!pushed_child && (frame.chosen_op.empty() ||
                            frame.candidate_idx >= static_cast<int>(frame.candidates.size()) - 1)) {
        // If there's still a chosen op that needs undoing (complete ordering case).
        if (!frame.chosen_op.empty()) {
          op_nodes_.at(frame.chosen_op).undo_run();
          max_global_mem = frame.saved_max;
          current_order.pop_back();
          executed.erase(frame.chosen_op);
          executed_mask.reset(op_index[frame.chosen_op]);
          for (auto& dep : dependents[frame.chosen_op]) {
            in_degree[dep]++;
          }
        }
        stack.pop_back();
      }
    }

    // Print results.
    std::cout << "=== Memory-Optimal Execution Order Search ===\n";
    std::cout << "Total valid orderings explored: " << num_orderings << "\n";
    std::cout << "Total subtrees pruned: " << num_pruned << "\n";
    std::cout << "Best peak memory: " << best_peak << " bytes\n\n";

    // Deallocate graph input tensors after the search.
    for (auto* act : inputs_) {
      act->deallocate_data();
    }
    max_global_mem = prev_max;

    return best_order;
  }

  std::vector<std::string> get_simple_order_for_subgraph(
      const std::set<std::string>& subgraph_ops) {
    auto [deps, dependents] = get_dependencies();

    struct MacroNode {
      std::string id;
      std::vector<std::string> ops;
      long long a;
      long long b;
    };

    std::map<std::string, MacroNode> macros;
    std::map<std::string, std::set<std::string>> macro_deps;
    std::map<std::string, std::set<std::string>> macro_dependents;

    for (const auto& id : subgraph_ops) {
      auto& node = op_nodes_.at(id);
      long long all_outputs = 0;
      for (auto* t : node.outputs()) all_outputs += t->size();
      long long workspace = node.workspace_req();
      long long total_memory_for_execution = all_outputs + workspace;
      long long memory_generate = all_outputs;
      long long memory_consumes = 0;
      for (auto* t : node.inputs()) {
        if (t->out_ref_count() == 1) memory_consumes += t->size();
      }

      MacroNode mn;
      mn.id = id;
      mn.ops = {id};
      mn.a = total_memory_for_execution;
      mn.b = memory_generate - memory_consumes;
      macros[id] = mn;

      macro_deps[id] = {};
      for (const auto& d : deps.at(id)) {
        if (subgraph_ops.count(d)) macro_deps[id].insert(d);
      }
      macro_dependents[id] = {};
      for (const auto& d : dependents.at(id)) {
        if (subgraph_ops.count(d)) macro_dependents[id].insert(d);
      }
    }

    auto compare = [](const MacroNode& i, const MacroNode& j) {
      long long cost_ij = std::max(i.a, i.b + j.a);
      long long cost_ji = std::max(j.a, j.b + i.a);
      if (cost_ij != cost_ji) return cost_ij < cost_ji;
      if (i.b != j.b) return i.b < j.b;
      return i.a < j.a;
    };

    int next_macro_id = 0;
    while (true) {
      std::string best_Y = "";
      std::string best_X = "";
      for (auto& [Y_id, Y_node] : macros) {
        if (macro_deps[Y_id].size() == 1) {
          std::string X_id = *macro_deps[Y_id].begin();
          if (compare(Y_node, macros[X_id])) {
            if (best_Y == "" || compare(Y_node, macros[best_Y])) {
              best_Y = Y_id;
              best_X = X_id;
            }
          }
        }
      }
      for (auto& [X_id, dependents_set] : macro_dependents) {
        if (dependents_set.size() == 1) {
          std::string Y_id = *dependents_set.begin();
          if (compare(macros[Y_id], macros[X_id])) {
            if (best_Y == "" || compare(macros[Y_id], macros[best_Y])) {
              best_Y = Y_id;
              best_X = X_id;
            }
          }
        }
      }
      if (best_Y == "") break;
      std::string XY_id = "macro_" + std::to_string(next_macro_id++);
      MacroNode XY;
      XY.id = XY_id;
      XY.ops = macros[best_X].ops;
      XY.ops.insert(XY.ops.end(), macros[best_Y].ops.begin(), macros[best_Y].ops.end());
      XY.a = std::max(macros[best_X].a, macros[best_X].b + macros[best_Y].a);
      XY.b = macros[best_X].b + macros[best_Y].b;
      macros[XY_id] = XY;
      macro_deps[XY_id] = macro_deps[best_X];
      for (auto& dep : macro_deps[best_Y]) {
        if (dep != best_X) macro_deps[XY_id].insert(dep);
      }

      macro_dependents[XY_id] = macro_dependents[best_X];
      macro_dependents[XY_id].erase(best_Y);
      for (auto& child : macro_dependents[best_Y]) {
        macro_dependents[XY_id].insert(child);
      }

      for (auto& parent : macro_deps[XY_id]) {
        macro_dependents[parent].erase(best_X);
        macro_dependents[parent].erase(best_Y);
        macro_dependents[parent].insert(XY_id);
      }
      for (auto& child : macro_dependents[XY_id]) {
        macro_deps[child].erase(best_X);
        macro_deps[child].erase(best_Y);
        macro_deps[child].insert(XY_id);
      }

      macros.erase(best_X);
      macros.erase(best_Y);
      macro_deps.erase(best_X);
      macro_deps.erase(best_Y);
      macro_dependents.erase(best_X);
      macro_dependents.erase(best_Y);
    }

    std::vector<std::string> final_order;
    std::set<std::string> executed_macros;
    std::vector<std::string> ready_macros;
    for (auto& [id, deps_set] : macro_deps) {
      if (deps_set.empty()) ready_macros.push_back(id);
    }
    while (final_order.size() < subgraph_ops.size()) {
      std::string best_macro = ready_macros[0];
      int best_idx = 0;
      for (size_t i = 1; i < ready_macros.size(); ++i) {
        if (compare(macros[ready_macros[i]], macros[best_macro])) {
          best_macro = ready_macros[i];
          best_idx = static_cast<int>(i);
        }
      }
      executed_macros.insert(best_macro);
      ready_macros.erase(ready_macros.begin() + best_idx);
      for (auto& op : macros[best_macro].ops) final_order.push_back(op);
      for (auto& child : macro_dependents[best_macro]) {
        bool ready = true;
        for (auto& parent : macro_deps[child]) {
          if (!executed_macros.count(parent)) {
            ready = false;
            break;
          }
        }
        if (ready) ready_macros.push_back(child);
      }
    }
    return final_order;
  }

  std::vector<std::string> find_macro_candidate_execution_order(bool print_macros = false) {
    std::vector<std::string> op_ids;
    for (auto& [id, _] : op_nodes_) {
      op_ids.push_back(id);
    }

    auto [deps, dependents] = get_dependencies();

    struct MacroCandidate {
      std::string id;
      std::vector<std::string> ops;
      std::set<std::string> external_dependencies;
    };
    std::vector<MacroCandidate> macros;
    int macro_counter = 0;

    for (const auto& join_op_id : op_ids) {
      if (deps.at(join_op_id).size() > 1) {
        std::set<std::string> exclusive_ancestors;
        std::queue<std::string> q;
        for (const auto& dep : deps.at(join_op_id)) {
          if (dependents.at(dep).size() == 1) {
            q.push(dep);
          }
        }

        while (!q.empty()) {
          std::string curr = q.front();
          q.pop();
          exclusive_ancestors.insert(curr);
          for (const auto& p : deps.at(curr)) {
            if (dependents.at(p).size() == 1) {
              q.push(p);
            }
          }
        }

        if (!exclusive_ancestors.empty()) {
          MacroCandidate mc;
          mc.id = "macro_" + std::to_string(macro_counter++);

          mc.ops = get_simple_order_for_subgraph(exclusive_ancestors);
          mc.ops.push_back(join_op_id);

          for (const auto& op : mc.ops) {
            for (const auto& d : deps.at(op)) {
              if (!exclusive_ancestors.count(d) && d != join_op_id) {
                mc.external_dependencies.insert(d);
              }
            }
          }

          if (print_macros) {
            std::cout << "Adding macro with ops: ";
            for (const auto& op : mc.ops) {
              std::cout << op << " ";
            }
            std::cout << "\n";
          }

          macros.push_back(mc);
        }
      }
    }

    std::vector<std::string> final_order;
    std::set<std::string> executed;

    size_t prev_max = max_global_mem;
    for (auto* act : inputs_) {
      act->allocate_data();
    }
    max_global_mem = global_mem;

    while (final_order.size() < op_ids.size()) {
      std::vector<std::string> ready_ops;
      for (const auto& op : op_ids) {
        if (executed.count(op)) continue;
        bool ready = true;
        for (const auto& d : deps.at(op)) {
          if (!executed.count(d)) {
            ready = false;
            break;
          }
        }
        if (ready) ready_ops.push_back(op);
      }

      std::vector<int> ready_macros;
      for (size_t i = 0; i < macros.size(); ++i) {
        bool valid = true;
        for (const auto& op : macros[i].ops) {
          if (executed.count(op)) {
            valid = false;
            break;
          }
        }
        if (!valid) continue;
        bool ready = true;
        for (const auto& d : macros[i].external_dependencies) {
          if (!executed.count(d)) {
            ready = false;
            break;
          }
        }
        if (ready) ready_macros.push_back(static_cast<int>(i));
      }

      if (ready_ops.empty()) {
        throw std::runtime_error("Graph has a cycle or unresolved dependencies.");
      }

      long long best_delta = std::numeric_limits<long long>::max();
      long long best_spike = std::numeric_limits<long long>::max();
      std::vector<std::string> best_candidate_ops;

      auto evaluate_candidate = [&](const std::vector<std::string>& cand_ops) {
        size_t start_mem = global_mem;
        size_t saved_max_mem = max_global_mem;
        max_global_mem = global_mem;

        for (const auto& op : cand_ops) {
          op_nodes_.at(op).run();
        }

        long long spike =
            static_cast<long long>(max_global_mem) - static_cast<long long>(start_mem);
        long long delta = static_cast<long long>(global_mem) - static_cast<long long>(start_mem);

        for (auto it = cand_ops.rbegin(); it != cand_ops.rend(); ++it) {
          op_nodes_.at(*it).undo_run();
        }
        max_global_mem = saved_max_mem;

        if (delta < best_delta || (delta == best_delta && spike < best_spike)) {
          best_delta = delta;
          best_spike = spike;
          best_candidate_ops = cand_ops;
        }
      };

      for (const auto& op : ready_ops) {
        evaluate_candidate({op});
      }
      for (int mi : ready_macros) {
        evaluate_candidate(macros[mi].ops);
      }

      for (const auto& op : best_candidate_ops) {
        op_nodes_.at(op).run();
        executed.insert(op);
        final_order.push_back(op);
      }
    }

    for (auto it = final_order.rbegin(); it != final_order.rend(); ++it) {
      op_nodes_.at(*it).undo_run();
    }
    for (auto* act : inputs_) {
      act->deallocate_data();
    }
    max_global_mem = prev_max;

    return final_order;
  }

  void export_to_dot(const std::string& filename) const {
    std::ofstream out(filename);
    out << "digraph ComputationGraph {\n";
    out << "  rankdir=LR;\n";

    // Draw Operation Nodes
    out << "  node [shape=box, style=filled, fillcolor=lightgray, fontname=\"Arial\"];\n";
    for (const auto& [id, op] : op_nodes_) {
      out << "  \"" << id << "\" [label=\"" << id << "\\nWorkspace: " << op.workspace_req()
          << "\"];\n";
    }

    // Draw Tensor/Activation Nodes
    out << "  node [shape=ellipse, style=filled, fillcolor=lightblue];\n";
    std::set<std::string> drawn_tensors;
    auto draw_tensor = [&](const ActNode* t) {
      if (drawn_tensors.insert(t->uuid()).second) {
        out << "  \"" << t->uuid() << "\" [label=\"" << t->uuid() << "\\nSize: " << t->size()
            << "\"];\n";
      }
    };

    // Connect Nodes
    for (const auto& [id, op] : op_nodes_) {
      for (auto* t : op.inputs()) {
        draw_tensor(t);
        out << "  \"" << t->uuid() << "\" -> \"" << id << "\";\n";
      }
      for (auto* t : op.outputs()) {
        draw_tensor(t);
        out << "  \"" << id << "\" -> \"" << t->uuid() << "\";\n";
      }
    }
    out << "}\n";
  }
};

size_t random_act_size() {
  int rand_val = rand() % 100;
  if (rand_val < 30) return 1280;
  if (rand_val < 60) return 2560;
  if (rand_val < 80) return 5120;
  return 10240;
}

size_t random_ws_size() {
  int rand_val = rand() % 100;
  if (rand_val < 30) return 512;
  if (rand_val < 60) return 1024;
  if (rand_val < 80) return 2048;
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
    auto act = g.add_act(prefix + "_seq", random_act_size());
    g.add_node(prefix + "_seq_conv", random_ws_size(), {input}, {act});
    return build_random_diamond_dag(g, act, depth - 1, node_counter);
  } else if (structure_type == 1) {
    // Diamond
    auto left_act = g.add_act(prefix + "_left", random_act_size());
    g.add_node(prefix + "_left_conv", random_ws_size(), {input}, {left_act});
    auto left_out = build_random_diamond_dag(g, left_act, depth - 1, node_counter);

    auto right_act = g.add_act(prefix + "_right", random_act_size());
    g.add_node(prefix + "_right_conv", random_ws_size(), {input}, {right_act});
    auto right_out = build_random_diamond_dag(g, right_act, depth - 1, node_counter);

    auto merge_act = g.add_act(prefix + "_merge", random_act_size());
    g.add_node(prefix + "_add", random_ws_size(), {left_out, right_out}, {merge_act});

    return merge_act;
  } else {
    // Residual (skip connection)
    auto main_act = g.add_act(prefix + "_main", random_act_size());
    g.add_node(prefix + "_main_conv", random_ws_size(), {input}, {main_act});
    auto main_out = build_random_diamond_dag(g, main_act, depth - 1, node_counter);

    auto merge_act = g.add_act(prefix + "_res_add", random_act_size());
    g.add_node(prefix + "_res_add", random_ws_size(), {main_out, input}, {merge_act});

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

Graph random_diamond_graph(int depth, int& node_counter) {
  Graph g;
  auto input = g.add_act("input", 10000);
  g.set_input_tensors(input);
  auto output = build_random_diamond_dag(g, input, depth, node_counter);
  g.set_output_tensors(output);
  return g;
}

Graph random_m_sequences_graph(int m, int length, int& node_counter) {
  Graph g;
  for (int i = 0; i < m; ++i) {
    auto input = g.add_act("input_" + std::to_string(i), random_act_size());
    g.set_input_tensors(input);

    ActNode* curr = input;
    for (int j = 0; j < length; ++j) {
      std::string prefix = "seq_" + std::to_string(i) + "_" + std::to_string(j);
      auto next_act = g.add_act(prefix, random_act_size());
      g.add_node(prefix + "_op", random_ws_size(), {curr}, {next_act});
      curr = next_act;
    }
    g.set_output_tensors(curr);
  }
  return g;
}

void build_random_branching_dag(Graph& g, ActNode* input, int depth, int& node_counter) {
  if (depth == 0) {
    g.set_output_tensors(input);
    return;
  }

  std::string prefix = "b" + std::to_string(depth) + "_" + std::to_string(node_counter++);
  bool is_seq = (rand() % 3 > 0);

  if (is_seq) {
    auto next_act = g.add_act(prefix + "_seq", random_act_size());
    g.add_node(prefix + "_seq_conv", random_ws_size(), {input}, {next_act});
    build_random_branching_dag(g, next_act, depth - 1, node_counter);
  } else {
    int num_branches = rand() % 2 + 2;
    for (int i = 0; i < num_branches; ++i) {
      auto next_act = g.add_act(prefix + "_branch" + std::to_string(i), random_act_size());
      g.add_node(prefix + "_conv" + std::to_string(i), random_ws_size(), {input}, {next_act});
      build_random_branching_dag(g, next_act, depth - 1, node_counter);
    }
  }
}

Graph random_branching_graph(int depth, int& node_counter) {
  Graph g;
  auto input = g.add_act("input", random_act_size());
  g.set_input_tensors(input);
  build_random_branching_dag(g, input, depth, node_counter);
  return g;
}

ActNode* build_random_joining_dag(Graph& g, int depth, int& node_counter) {
  if (depth == 0) {
    std::string prefix = "j" + std::to_string(depth) + "_" + std::to_string(node_counter++);
    auto input = g.add_act(prefix + "_input", random_act_size());
    g.set_input_tensors(input);
    return input;
  }

  std::string prefix = "j" + std::to_string(depth) + "_" + std::to_string(node_counter++);
  bool is_seq = (rand() % 2 == 0);

  if (is_seq) {
    auto input = build_random_joining_dag(g, depth - 1, node_counter);
    auto output = g.add_act(prefix + "_seq_out", random_act_size());
    g.add_node(prefix + "_seq_conv", random_ws_size(), {input}, {output});
    return output;
  } else {
    int num_joining = rand() % 2 + 2;
    std::vector<ActNode*> inputs;
    for (int i = 0; i < num_joining; ++i) {
      inputs.push_back(build_random_joining_dag(g, depth - 1, node_counter));
    }
    auto output = g.add_act(prefix + "_merge", random_act_size());
    g.add_node(prefix + "_add", random_ws_size(), inputs, {output});
    return output;
  }
}

Graph random_joining_graph(int depth, int& node_counter) {
  Graph g;
  auto output = build_random_joining_dag(g, depth, node_counter);
  g.set_output_tensors(output);
  return g;
}

inline bool near(double a, double b) { return fabs(a - b) < 1e-15; }

signed main() {
  srand(static_cast<unsigned int>(time(nullptr)));

  int trials;
  std::cin >> trials;
  int original_trials = trials;
  std::map<std::string, std::vector<double>> all_efficiencies;

  while (trials--) {
    global_mem = 0;
    max_global_mem = 0;

    int node_counter = 0;
    // Graph g = random_diamond_graph(6, node_counter);
    // Graph g = random_branching_graph(3, node_counter);
    Graph g = random_joining_graph(4, node_counter);
    // int m = 4;        // Number of independent sequences
    // int length = 10;  // Length of each sequence
    // Graph g = random_m_sequences_graph(m, length, node_counter);

    auto best_order = g.find_minimum_memory_execution_order();

    auto simple_order = g.find_simple_execution_order();

    auto macro_order = g.find_macro_candidate_execution_order();

    auto effs = g.rank_execution_orders(
        {{"BEST", best_order}, {"SIMPLE", simple_order}, {"MACRO", macro_order}});

    for (auto eff : effs) {
      if (eff.first == "MACRO" && !near(eff.second, 100)) {
        g.print_graph();
        g.export_to_dot("graph.dot");
        g.find_macro_candidate_execution_order(true);
      }
    }
    for (const auto& [name, eff] : effs) {
      all_efficiencies[name].push_back(eff);
    }
  }

  std::cout << "=== Efficiency Overview (" << original_trials << " trials) ===\n";
  for (auto& [name, effs] : all_efficiencies) {
    if (effs.empty()) continue;
    std::sort(effs.begin(), effs.end());
    double sum = 0;
    for (double e : effs) sum += e;
    double avg = sum / effs.size();
    double worst = effs.front();
    double best = effs.back();

    // Percentiles (sorted ascending, so index 0 is worst)
    double p10 = effs[effs.size() * 10 / 100];
    double p50 = effs[effs.size() * 50 / 100];
    double p90 = effs[effs.size() * 90 / 100];

    std::cout << name << " Order:\n";
    std::cout << "  Average: " << std::fixed << std::setprecision(2) << avg << "%\n";
    std::cout << "  Worst:   " << std::fixed << std::setprecision(2) << worst << "%\n";
    std::cout << "  p10:     " << std::fixed << std::setprecision(2) << p10 << "%\n";
    std::cout << "  Median:  " << std::fixed << std::setprecision(2) << p50 << "%\n";
    std::cout << "  p90:     " << std::fixed << std::setprecision(2) << p90 << "%\n";
    std::cout << "  Best:    " << std::fixed << std::setprecision(2) << best << "%\n\n";
  }

  return 0;
}
