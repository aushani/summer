#include "app/sim_world_occ_grids/environment_generator.h"

namespace app {
namespace sim_world_occ_grids {

EnvironmentGenerator::EnvironmentGenerator(const std::string &save_base_fn) :
 og_builder_(10000, kRes, 400.0),
 data_manager_(6, false, false),
 save_base_path_(save_base_fn) {
  og_builder_.ConfigureSizeInPixels(kPixelSize_, kPixelSize_, 1);
}

void EnvironmentGenerator::Run() {
  int i = 0;
  while (true) {
    // Get sim data
    sw::Data *data = data_manager_.GetData();
    std::vector<ge::Point> *hits = data->GetHits();

    // Convert hits to "velodyne scan"
    std::vector<Eigen::Vector3d> scan;
    for (const auto &p : *hits) {
      scan.emplace_back(p.x, p.y, 0);
    }

    // Make occ grid
    rt::OccGrid og = og_builder_.GenerateOccGrid(scan);

    // Save Occ Grid
    //fs::path dir = save_base_path_ / fs::path(shape.GetName());
    fs::path dir = save_base_path_;
    if (!fs::exists(dir)) {
      printf("Making path: %s\n", dir.string().c_str());
      fs::create_directories(dir);
    }

    char fn[1000];
    sprintf(fn, "SIMWORLD_%06d.og", i);
    fs::path path = dir / fs::path(fn);

    //og.Save(fn);
    DumpBin(og, path);

    i++;

    delete data;
  }
}

void EnvironmentGenerator::DumpBin(const rt::OccGrid &og, const fs::path &path) const {
  std::ofstream file(path.string(), std::ios::out | std::ios::binary);

  for (int i=-kPixelSize_+1; i<kPixelSize_; i++) {
    for (int j=-kPixelSize_+1; j<kPixelSize_; j++) {
      float p = og.GetProbability(rt::Location(i, j, 0));

      // gross
      file.write((const char*)(&p), sizeof(float));
    }
  }

  file.close();
}

} // namespace sim_world_occ_grids
} // namespace app
