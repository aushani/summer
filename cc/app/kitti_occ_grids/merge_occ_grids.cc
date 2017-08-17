#include <iostream>
#include <map>
#include <vector>

#include <boost/assert.hpp>
#include <boost/filesystem.hpp>

#include "library/ray_tracing/occ_grid.h"

namespace rt = library::ray_tracing;
namespace fs = boost::filesystem;

int main(int argc, char** argv) {
  printf("Merge Occ Grids\n");

  if (argc < 3) {
    printf("Usage: %s dir_name out.og\n", argv[0]);
    return 1;
  }

  std::map<rt::Location, double> mog_map;
  std::map<rt::Location, int> counts;

  fs::path p(argv[1]);
  fs::directory_iterator end_it;
  int count_og = 0;
  double res = -1.0;
  for (fs::directory_iterator it(p); it != end_it; it++) {
    if (!fs::is_regular_file(it->path())) {
      continue;
    }

    // Do the accumulating
    rt::OccGrid og = rt::OccGrid::Load(it->path().string().c_str());
    double og_res = og.GetResolution();
    if (res < 0.0) {
      res = og_res;
    }
    BOOST_ASSERT(og_res == res);

    const auto &locs = og.GetLocations();
    const auto &los = og.GetLogOdds();

    for (size_t j = 0; j < locs.size(); j++) {
      const auto &loc = locs[j];
      float lo = los[j];

      double p = 1 / (1 + exp(-lo));
      mog_map[loc] += p;
      counts[loc]++;
    }
    count_og++;
    printf("Merged %d\n", count_og);
  }

  // Make occ grid and save
  std::vector<rt::Location> locs;
  std::vector<float> los;
  for (auto it = mog_map.begin(); it != mog_map.end(); it++) {
    locs.push_back(it->first);

    double sum_p_known = it->second;
    double sum_p_unknown = (count_og - counts[it->first]) * 0.5;

    double p = (sum_p_known + sum_p_unknown) / count_og;
    double l = -log(1/p - 1);
    los.push_back(l);
  }
  rt::OccGrid og(locs, los, res);
  og.Save(argv[2]);

  return 0;
}
