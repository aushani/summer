#include <iostream>
#include <map>
#include <vector>

#include <boost/assert.hpp>
#include <boost/filesystem.hpp>

#include "library/timer/timer.h"

#include "app/kitti_occ_grids/joint_model.h"

namespace kog = app::kitti_occ_grids;
namespace fs = boost::filesystem;

int main(int argc, char** argv) {
  library::timer::Timer t;

  printf("Make Chow-Lui Trees\n");

  if (argc < 3) {
    printf("Usage: %s joint_model out\n", argv[0]);
    return 1;
  }

  t.Start();
  kog::JointModel model = kog::JointModel::Load(argv[1]);
  printf("Took %5.3f ms to load %s\n", t.GetMs(), argv[1]);

  return 0;
}
