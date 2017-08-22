#include <iostream>
#include <map>
#include <vector>

#include <boost/assert.hpp>
#include <boost/filesystem.hpp>

#include "library/timer/timer.h"

#include "app/kitti_occ_grids/joint_model.h"
#include "app/kitti_occ_grids/chow_lui_tree.h"

namespace kog = app::kitti_occ_grids;
namespace fs = boost::filesystem;

int main(int argc, char** argv) {
  library::timer::Timer t;

  printf("Make Chow-Lui Trees\n");
  if (argc < 3) {
    printf("Usage: %s dir_name out\n", argv[0]);
    return 1;
  }

  printf("Merging Occ Grids...\n");

  kog::JointModel *model = nullptr;

  int count = 0;
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

    if (t_step.GetSeconds() > 60) {
      printf("Merged %d (%5.3f sec per og)\n", count, t.GetSeconds() / count);
      t_step.Start();
    }
  }
  printf("Have joint model, took %5.3f seconds\n", t.GetSeconds());

  t.Start();
  kog::ChowLuiTree tree(*model);
  printf("Took %5.3f ms to make Chow-Lui tree\n", t.GetMs());

  char fn[1000];
  sprintf(fn, "%s.clt", argv[2]);

  t.Start();
  tree.Save(fn);
  printf("Took %5.3f sec to save Chow-Lui tree\n", t.GetSeconds());

  sprintf(fn, "%s.jm", argv[2]);

  t.Start();
  model->Save(fn);
  printf("Took %5.3f sec to save model\n", t.GetSeconds());

  delete model;

  return 0;
}
