#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <unistd.h>
#include <vector>

#include <sys/stat.h>

#include "library/ray_tracing/occ_grid_builder.h"
#include "library/sim_world/sim_world.h"
#include "library/sim_world/data.h"

namespace rt = library::ray_tracing;
namespace sw = library::sim_world;

namespace app {
namespace sim_world_occ_grids {

class OccGridExtractor {
 public:
  OccGridExtractor(const std::string &save_base_fn);

  void Run();

 private:
  static constexpr double kPosRes_ = 0.30;                 // 30 cm
  static constexpr double kAngleRes_ = 15.0 * M_PI / 360.0; // 15 degrees
  static constexpr int kEntriesPerObj_ = 10;

  std::vector<std::string> classes_;
  std::map<std::string, int> class_counts_;

  rt::OccGridBuilder og_builder_;
  sw::DataManager data_manager_;

  std::default_random_engine rand_engine_;

  std::string save_base_fn_;
};

} // namespace kitti
} // namespace app
