#include <iostream>
#include <map>
#include <vector>

#include "library/ray_tracing/occ_grid.h"

namespace rt = library::ray_tracing;

int main(int argc, char** argv) {
  printf("Merge Occ Grids\n");

  std::map<rt::Location, double> mog_map;
  std::map<rt::Location, int> counts;

  // Do the merging
  double res = 0.0;
  int num_og = argc - 1;
  for (int i = 1; i < argc; i++) {
    printf("Merging %s\n", argv[i]);

    rt::OccGrid og = rt::OccGrid::Load(argv[i]);
    res = og.GetResolution();

    const auto &locs = og.GetLocations();
    const auto &los = og.GetLogOdds();

    for (size_t j = 0; j < locs.size(); j++) {
      const auto &loc = locs[j];
      float lo = los[j];

      double p = 1 / (1 + exp(-lo));
      mog_map[loc] += p;
      counts[loc]++;
    }
    printf("Merged %d / %d\n", i, num_og);
  }

  printf("Done!\n");

  // Make occ grid and save
  std::vector<rt::Location> locs;
  std::vector<float> los;
  for (auto it = mog_map.begin(); it != mog_map.end(); it++) {
    locs.push_back(it->first);

    double sum_p_known = it->second;
    double sum_p_unknown = (num_og - counts[it->first]) * 0.5;

    double p = (sum_p_known + sum_p_unknown) / num_og;
    double l = -log(1/p - 1);
    los.push_back(l);
  }
  rt::OccGrid og(locs, los, res);
  og.Save("merged.og");

  return 0;
}
