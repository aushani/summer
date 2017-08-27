#include <iostream>
#include <map>
#include <vector>

#include <boost/assert.hpp>
#include <boost/filesystem.hpp>

#include "library/timer/timer.h"

#include "app/kitti_occ_grids/joint_model.h"
#include "app/kitti_occ_grids/marginal_model.h"

namespace kog = app::kitti_occ_grids;
namespace fs = boost::filesystem;

int main(int argc, char** argv) {
  printf("Make Marginal Model\n");

  library::timer::Timer t;

  if (argc < 3) {
    printf("Usage: %s joint_model.jm out.mm\n", argv[0]);
    return 1;
  }

  t.Start();
  kog::JointModel jm = kog::JointModel::Load(argv[1]);
  printf("Took %5.3f sec to load joint model\n", t.GetSeconds());

  t.Start();
  kog::MarginalModel mm(jm);
  printf("Took %5.3f sec to make marginal model\n", t.GetSeconds());

  t.Start();
  mm.Save(argv[2]);
  printf("Took %5.3f sec to save marginal model\n", t.GetSeconds());

  return 0;
}
