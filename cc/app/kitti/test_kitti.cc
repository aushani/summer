#include <iostream>

#include "library/kitti/tracklets.h"
#include "library/kitti/velodyne_scan.h"

namespace kt = library::kitti;

int main() {
  printf("Testing KITTI...\n");

  kt::Tracklets t;

  std::string fn("/home/aushani/data/kittidata/extracted/tracklet_labels.xml");
  bool success = t.loadFromFile(fn);

  printf("Loaded tracklets from %s: %s\n", fn.c_str(), success ? "SUCCESS":"FAIL");

  printf("Have %d tracklets\n", t.numberOfTracklets());

  kt::VelodyneScan vs("/home/aushani/data/kittidata/extracted/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000000.bin");

  printf("Loaded %ld hits\n", vs.GetHits().size());

  for (const auto &h : vs.GetHits()) {
    printf("%5.3f, %5.3f, %5.3f\n", h(0), h(1), h(2));
  }
}
