#include <iostream>
#include <map>
#include <vector>

#include <boost/assert.hpp>
#include <boost/filesystem.hpp>

#include "library/ray_tracing/occ_grid.h"

#include "app/kitti_occ_grids/model.h"

namespace kog = app::kitti_occ_grids;
namespace rt = library::ray_tracing;
namespace fs = boost::filesystem;

int main(int argc, char** argv) {
  printf("Merge Occ Grids\n");

  if (argc < 3) {
    printf("Usage: %s dir_name out\n", argv[0]);
    return 1;
  }

  kog::Model *model = nullptr;

  int count = 0;

  fs::path p(argv[1]);
  fs::directory_iterator end_it;
  for (fs::directory_iterator it(p); it != end_it; it++) {
    if (!fs::is_regular_file(it->path())) {
      continue;
    }

    rt::OccGrid og = rt::OccGrid::Load(it->path().string().c_str());

    if (model == nullptr) {
      model = new kog::Model(og.GetResolution());
    }

    BOOST_ASSERT(model->GetResolution() == og.GetResolution());

    // Do the accumulating
    const auto &locs = og.GetLocations();
    const auto &los = og.GetLogOdds();

    for (size_t j = 0; j < locs.size(); j++) {
      const auto &loc = locs[j];
      float lo = los[j];

      model->MarkObservation(loc, lo);
    }

    printf("Merged %d\n", ++count);
  }

  // Save
  model->Save(argv[2]);

  delete model;

  return 0;
}
