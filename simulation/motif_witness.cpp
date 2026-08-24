#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "graph.h"
#include "vision_graph_generator.h"

template <typename T>
bool contains(const std::vector<T>& vec, const T& val) {
  return std::find(vec.begin(), vec.end(), val) != vec.end();
}

void print_stats(std::vector<std::pair<std::string, std::vector<double>>>& effs_by_name) {
  for (auto& [name, effs] : effs_by_name) {
    if (effs.empty()) return;
    std::sort(effs.begin(), effs.end());
    double sum = 0;
    for (double e : effs) sum += e;
    double avg = sum / effs.size();
    double worst = effs.front();
    double best = effs.back();
    double p25 = effs[effs.size() * 25 / 100];
    double p50 = effs[effs.size() * 50 / 100];
    double p75 = effs[effs.size() * 75 / 100];

    std::cout << "Order: " << name << "\n";
    std::cout << "  Average: " << std::fixed << std::setprecision(2) << avg << "%\n";
    std::cout << "  Worst:   " << std::fixed << std::setprecision(2) << worst << "%\n";
    std::cout << "  p25:     " << std::fixed << std::setprecision(2) << p25 << "%\n";
    std::cout << "  Median:  " << std::fixed << std::setprecision(2) << p50 << "%\n";
    std::cout << "  p75:     " << std::fixed << std::setprecision(2) << p75 << "%\n";
    std::cout << "  Best:    " << std::fixed << std::setprecision(2) << best << "%\n";
  }
}

void dump_execution_order_with_memory(Graph& g, const std::vector<std::string>& order,
                                      const std::string& filename) {
  std::ofstream out(filename);
  if (order.empty()) return;
  Allocator allocator;
  GraphExecutor executor(g);

  size_t global_peak = 0;
  size_t step_peak = 0;
  allocator.subscribe("peak", [&](size_t new_mem) {
    if (new_mem > global_peak) global_peak = new_mem;
    if (new_mem > step_peak) step_peak = new_mem;
  });

  executor.init_boundaries(&allocator);
  out << "Init boundary: retained=" << allocator.allocated()
      << ", local_peak=" << allocator.allocated() << ", global_peak=" << global_peak << "\n";

  for (const auto& op_id : order) {
    step_peak = allocator.allocated();
    executor.run_op_node(&g.get_op(op_id), &allocator);
    out << op_id << ": retained=" << allocator.allocated() << ", local_peak=" << step_peak
        << ", global_peak=" << global_peak << "\n";
  }

  out << "Final global peak: " << global_peak << "\n";
  allocator.unsubscribe("peak");
}

void run_motif_witness(MotifTarget target, const std::string& name, int trials) {
  std::cout << "\nGenerating Motif: " << name << "\n";
  std::vector<double> effs;
  std::vector<double> scaled_effs;
  std::vector<std::pair<double, uint32_t>> candidate_seeds;
  std::vector<double> dfs_vs_rank;
  std::vector<double> rank_vs_linear;
  std::vector<double> linear_vs_branch;
  std::vector<double> linear_vs_join;
  std::vector<double> branch_vs_full;
  std::vector<double> join_vs_full;
  int success = 0;

  for (int i = 0; i < trials; ++i) {
    try {
      GeneratedGraph witness = generate_witness(target, rand());
      double eff = witness.peaks.oracle > 0 ? (100.0 *
                                               (static_cast<double>(witness.peaks.full) -
                                                static_cast<double>(witness.peaks.oracle)) /
                                               static_cast<double>(witness.peaks.oracle))
                                            : 0.0;
      effs.push_back(eff);

      // Now generate scaled version using the exact same seed that worked
      IRGraph scaled_ir = scale_ir(build_candidate(target, witness.seed), witness.seed);
      randomize_ids_and_materialization(scaled_ir, witness.seed);
      Graph scaled_graph = materialize(scaled_ir);
      AblationPeaks scaled_peaks = evaluate_ablations(scaled_graph);

      // For scaled graph, we compare FULL against DFS just to show the macroscopic effect
      double scaled_eff =
          scaled_peaks.dfs > 0
              ? (100.0 *
                 (static_cast<double>(scaled_peaks.dfs) - static_cast<double>(scaled_peaks.full)) /
                 static_cast<double>(scaled_peaks.dfs))
              : 0.0;
      scaled_effs.push_back(scaled_eff);

      auto gain = [](size_t baseline, size_t optimized) {
        return baseline > 0
                   ? (100.0 * (static_cast<double>(baseline) - static_cast<double>(optimized)) /
                      static_cast<double>(baseline))
                   : 0.0;
      };

      dfs_vs_rank.push_back(gain(scaled_peaks.dfs, scaled_peaks.rank));
      rank_vs_linear.push_back(gain(scaled_peaks.rank, scaled_peaks.linear));
      linear_vs_branch.push_back(gain(scaled_peaks.linear, scaled_peaks.branch));
      linear_vs_join.push_back(gain(scaled_peaks.linear, scaled_peaks.join));
      branch_vs_full.push_back(gain(scaled_peaks.branch, scaled_peaks.full));
      join_vs_full.push_back(gain(scaled_peaks.join, scaled_peaks.full));

      double target_gain = 0.0;
      switch (target) {
        case MotifTarget::Rank:
          target_gain = gain(scaled_peaks.dfs, scaled_peaks.rank);
          break;
        case MotifTarget::Linear:
          target_gain = gain(scaled_peaks.rank, scaled_peaks.linear);
          break;
        case MotifTarget::Branch:
          target_gain = gain(scaled_peaks.linear, scaled_peaks.branch);
          break;
        case MotifTarget::Join:
          target_gain = gain(scaled_peaks.linear, scaled_peaks.join);
          break;
        case MotifTarget::ForkJoin:
          target_gain = gain(std::min(scaled_peaks.branch, scaled_peaks.join), scaled_peaks.full);
          break;
      }
      candidate_seeds.push_back({target_gain, witness.seed});

      success++;
    } catch (const std::exception& e) {
      std::cout << "Failed to generate motif witness for " << name << " on attempt " << i << ": "
                << e.what() << "\n";
    }
  }

  std::cout << "Generated " << success << "/" << trials << " " << name << " witnesses.\n";
  if (success > 0) {
    std::vector<std::pair<std::string, std::vector<double>>> res = {
        {"Diff (Full vs Oracle)", effs},
        {"Gain of Rank vs DFS", dfs_vs_rank},
        {"Gain of Linear vs Rank", rank_vs_linear},
        {"Gain of Branch vs Linear", linear_vs_branch},
        {"Gain of Join vs Linear", linear_vs_join},
        {"Gain of Full vs Branch", branch_vs_full},
        {"Gain of Full vs Join", join_vs_full},
        {"Gain of Full Macro vs DFS", scaled_effs}};
    print_stats(res);
  }

  std::sort(candidate_seeds.begin(), candidate_seeds.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

  std::string safe_name = name;
  std::replace(safe_name.begin(), safe_name.end(), ' ', '_');
  std::replace(safe_name.begin(), safe_name.end(), '(', '_');
  std::replace(safe_name.begin(), safe_name.end(), ')', '_');

  int export_count = std::min(5, static_cast<int>(candidate_seeds.size()));
  for (int k = 0; k < export_count; ++k) {
    uint32_t seed = candidate_seeds[k].second;
    IRGraph scaled_ir = scale_ir(build_candidate(target, seed), seed);
    randomize_ids_and_materialization(scaled_ir, seed);
    Graph scaled_graph = materialize(scaled_ir);

    std::string base_path = "top5/top5_" + safe_name + "_" + std::to_string(k + 1);
    save_graph_to_dot(scaled_graph, base_path + ".dot");
  }
}

int main() {
  std::system("mkdir -p top5");
  srand(static_cast<unsigned int>(time(nullptr)));
  std::vector<std::string> to_checks = {"MACRO"};

  int trials = 1000;

  // Mechanism Witnesses
  run_motif_witness(MotifTarget::Rank, "V1 (Rank)", trials);
  run_motif_witness(MotifTarget::Linear, "V2 (Linear)", trials);
  run_motif_witness(MotifTarget::Branch, "V3 (Branch)", trials);
  run_motif_witness(MotifTarget::Join, "V4 (Join)", trials);
  run_motif_witness(MotifTarget::ForkJoin, "V5 (Recursive Fork-Join)", trials);

  return 0;
}
