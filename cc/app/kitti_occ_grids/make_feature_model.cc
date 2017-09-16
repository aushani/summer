#include <iostream>
#include <map>
#include <vector>

#include <boost/assert.hpp>
#include <boost/filesystem.hpp>

#include "library/feature/feature_model.h"
#include "library/ray_tracing/feature_occ_grid.h"
#include "library/timer/timer.h"

namespace ft = library::feature;
namespace rt = library::ray_tracing;
namespace fs = boost::filesystem;

int main(int argc, char** argv) {
  printf("Make feature model\n");

  if (argc < 3) {
    printf("Usage: %s dir_name out\n", argv[0]);
    return 1;
  }

  ft::FeatureModel *model = nullptr;

  int count = 0;
  library::timer::Timer t;
  library::timer::Timer t_step;

  fs::path p(argv[1]);
  fs::directory_iterator end_it;
  for (fs::directory_iterator it(p); it != end_it; it++) {
    if (!fs::is_regular_file(it->path())) {
      continue;
    }

    if (it->path().extension().string() != ".fog") {
      printf("skipping extension %s\n", it->path().extension().string().c_str());
      continue;
    }

    rt::FeatureOccGrid fog = rt::FeatureOccGrid::Load(it->path().string().c_str());

    if (model == nullptr) {
      model = new ft::FeatureModel(3.0, 2.0, fog.GetResolution());
    }

    BOOST_ASSERT(model->GetResolution() == fog.GetResolution());

    // Do the accumulating
    model->MarkObservations(fog);
    count++;

    if (t_step.GetSeconds() > 60) {
      printf("Merged %d (%9.5f sec per fog)\n", count, t.GetSeconds() / count);

      t_step.Start();
      model->Save(argv[2]);
      printf("Saved tmp result in %5.3f ms\n", t.GetMs());

      t_step.Start();
    }
  }

  // Save
  printf("Done! Saving to %s...\n", argv[2]);
  model->Save(argv[2]);
  printf("Saved! Goodbye\n");

  delete model;

  return 0;
}
