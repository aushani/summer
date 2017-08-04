#include <iostream>

#include "library/kitti/tracklets.h"
#include "library/kitti/velodyne_scan.h"

namespace kt = library::kitti;

int main() {
  printf("Testing KITTI...\n");

  kt::Tracklets t;

  bool success = t.loadFromFile("/home/aushani/data/kittidata/extracted/2011_09_26/2011_09_26_drive_0001_sync/tracklet_labels.xml");

  printf("Loaded tracklets: %s\n", success ? "SUCCESS":"FAIL");
  printf("Have %d tracklets\n", t.numberOfTracklets());

  for (int i=0; i<t.numberOfTracklets(); i++) {
    auto *tt = t.getTracklet(i);
    printf("Have %s (size %5.3f x %5.3f x %5.3f) for %ld frames\n",
        tt->objectType.c_str(), tt->h, tt->w, tt->l, tt->poses.size());
  }

  kt::VelodyneScan vs("/home/aushani/data/kittidata/extracted/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000000.bin");

  printf("Loaded %ld hits\n", vs.GetHits().size());
}
