#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <random>
#include <chrono>

#include <boost/archive/binary_oarchive.hpp>

#include "library/kitti/camera_cal.h"
#include "library/kitti/tracklets.h"
#include "library/kitti/velodyne_scan.h"

#include "app/kitti/model_bank_builder.h"

namespace kt = library::kitti;
namespace ut = library::util;

using namespace app::kitti;

int main() {
  printf("Bulding Model Bank from KITTI...\n");

  ModelBankBuilder mbb;
}
