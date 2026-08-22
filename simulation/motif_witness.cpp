#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "graph.h"
#include "vision_graph_generator.h"

template <typename T>
bool contains(const std::vector<T>& vec, const T& val) {
  return std::find(vec.begin(), vec.end(), val) != vec.end();
}

void print_stats(std::map<std::string, std::vector<double>>& effs_by_name) {
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

void run_motif_witness(MotifTarget target, const std::string& name, int trials) {
  std::cout << "\nGenerating Motif: " << name << "\n";
  std::vector<double> effs;
  std::vector<double> scaled_effs;
  int success = 0;

  for (int i = 0; i < trials; ++i) {
    try {
      GeneratedGraph witness = generate_witness(target, rand());
      double eff =
          witness.peaks.full > 0 ? (100.0 * witness.peaks.oracle / witness.peaks.full) : 0.0;
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

      success++;
    } catch (const std::exception& e) {
      std::cout << "Failed to generate motif witness for " << name << " on attempt " << i << ": "
                << e.what() << "\n";
    }
  }

  std::cout << "Generated " << success << "/" << trials << " " << name << " witnesses.\n";
  if (success > 0) {
    std::map<std::string, std::vector<double>> res = {{"Discovery (Oracle/Full)", effs},
                                                      {"Scaled (Full/DFS)", scaled_effs}};
    print_stats(res);
  }
}

int main() {
  srand(static_cast<unsigned int>(time(nullptr)));
  std::vector<std::string> to_checks = {"MACRO"};

  int trials = 20;

  // Mechanism Witnesses
  run_motif_witness(MotifTarget::Rank, "V1 (Rank)", trials);
  run_motif_witness(MotifTarget::Linear, "V2 (Linear)", trials);
  run_motif_witness(MotifTarget::Branch, "V3 (Branch)", trials);
  run_motif_witness(MotifTarget::Join, "V4 (Join)", trials);
  run_motif_witness(MotifTarget::ForkJoin, "V5 (Recursive Fork-Join)", trials);

  return 0;
}
