#include <iostream>
#include <map>
#include <vector>

#include <boost/assert.hpp>
#include <boost/filesystem.hpp>

#include "library/ray_tracing/occ_grid.h"
#include "library/timer/timer.h"

#include "app/kitti_occ_grids/joint_model.h"

namespace kog = app::kitti_occ_grids;
namespace rt = library::ray_tracing;
namespace fs = boost::filesystem;

int main(int argc, char** argv) {
  printf("Merge Occ Grids\n");

  if (argc < 3) {
    printf("Usage: %s dir_name out\n", argv[0]);
    return 1;
  }

  kog::JointModel *model = nullptr;

  int count = 0;
  library::timer::Timer t;
  library::timer::Timer t_step;

  fs::path p(argv[1]);
  fs::directory_iterator end_it;
  for (fs::directory_iterator it(p); it != end_it; it++) {
    if (!fs::is_regular_file(it->path())) {
      continue;
    }

    rt::OccGrid og = rt::OccGrid::Load(it->path().string().c_str());

    if (model == nullptr) {
      model = new kog::JointModel(2.0, 2.0, og.GetResolution());
    }

    BOOST_ASSERT(model->GetResolution() == og.GetResolution());

    // Do the accumulating
    model->MarkObservations(og);
    count++;

    if (t_step.GetSeconds() > 180) {
      printf("Merged %d (%5.3f sec per og)\n", count, t.GetSeconds() / count);
      t_step.Start();
    }
  }

  // Save
  model->Save(argv[2]);

  delete model;

  return 0;
}
