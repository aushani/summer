#include <iostream>

#include "library/kitti/tracklets.h"

int main() {
  printf("Testing KITTI...\n");

  library::kitti::Tracklets t;

  std::string fn("/home/aushani/data/kittidata/extracted/tracklet_labels.xml");
  bool success = t.loadFromFile(fn);

  printf("Loaded tracklets from %s: %s\n", fn.c_str(), success ? "SUCCESS":"FAIL");

  printf("Have %d tracklets\n", t.numberOfTracklets());
}
