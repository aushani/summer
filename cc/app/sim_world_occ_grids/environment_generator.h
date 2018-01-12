#pragma once

#include <boost/filesystem.hpp>
#include <sys/stat.h>

#include "library/ray_tracing/occ_grid_builder.h"
#include "library/sim_world/sim_world.h"
#include "library/sim_world/data.h"

namespace rt = library::ray_tracing;
namespace sw = library::sim_world;
namespace fs = boost::filesystem;

namespace app {
namespace sim_world_occ_grids {

class EnvironmentGenerator {
 public:
  EnvironmentGenerator(const std::string &save_base_fn);

  void Run();

 private:
  static constexpr double kRes = 0.30;         // 30 cm
  static constexpr int kPixelSize_ = 100;      // 30 m

  rt::OccGridBuilder og_builder_;
  sw::DataManager data_manager_;

  fs::path save_base_path_;

  void DumpBin(const rt::OccGrid &og, const fs::path &path) const;
};

} // namespace sim_world_occ_grids
} // namespace app
