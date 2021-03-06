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
  printf("Make Chow-Lui Trees\n");

  library::timer::Timer t;

  if (argc < 3) {
    printf("Usage: %s joint_model.jm out.clt\n", argv[0]);
    return 1;
  }

  t.Start();
  kog::JointModel jm = kog::JointModel::Load(argv[1]);
  printf("Took %5.3f sec to load joint model\n", t.GetSeconds());

  t.Start();
  kog::ChowLuiTree tree(jm);
  printf("Took %5.3f sec to make Chow-Lui tree\n", t.GetSeconds());

  t.Start();
  tree.Save(argv[2]);
  printf("Took %5.3f sec to save Chow-Lui tree\n", t.GetSeconds());

  return 0;
}
