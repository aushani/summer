#include "app/sim_world_occ_grids/occ_grid_extractor.h"

#include "library/ray_tracing/occ_grid.h"
#include "library/timer/timer.h"

namespace rt = library::ray_tracing;
namespace sw = library::sim_world;

namespace app {
namespace sim_world_occ_grids {

OccGridExtractor::OccGridExtractor(const std::string &save_base_fn) :
 og_builder_(10000, kPosRes_, 100.0),
 data_manager_(4, false, false),
 rand_engine_(std::chrono::system_clock::now().time_since_epoch().count()),
 save_base_path_(save_base_fn) {
  og_builder_.ConfigureSizeInPixels(kPixelSize_, kPixelSize_, 1);

  classes_.push_back("BOX");
  classes_.push_back("STAR");
}

void OccGridExtractor::Run() {
  // Sampling jitter
  std::uniform_real_distribution<double> jitter_pos(-kPosRes_/2, kPosRes_/2);
  std::uniform_real_distribution<double> jitter_angle(-kAngleRes_/2, kAngleRes_/2);

  // Sampling positions
  double lower_bound = -20.0;
  double upper_bound = 20.0;
  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  double pr2 = kPosRes_ * kPosRes_;

  while (true) {
    // Get sim data
    sw::Data *data = data_manager_.GetData();
    sw::SimWorld *sim = data->GetSim();
    std::vector<ge::Point> *hits = data->GetHits();

    // Convert hits to "velodyne scan"
    std::vector<Eigen::Vector3d> scan;
    for (const auto &p : *hits) {
      scan.emplace_back(p.x, p.y, 0);
    }

    const auto &shapes = sim->GetShapes();

    // Process classes in sim
    for (const auto &shape : shapes) {
      // Set up random center point in object
      double range = kPosRes_*kPixelSize_/6;
      std::uniform_real_distribution<double> dx(-range, range);
      std::uniform_real_distribution<double> dy(-range, range);

      for (int ex=0; ex<kEntriesPerObj_; ex++) {
        double x = shape.GetCenter().x() + dx(rand_engine_);
        double y = shape.GetCenter().y() + dy(rand_engine_);
        double z = 0;
        double theta = 0;

        bool is_background = !shape.IsInside(x, y);

        og_builder_.SetPose(Eigen::Vector3d(x, y, z), theta);

        // Make occ grid
        rt::OccGrid og = og_builder_.GenerateOccGrid(scan);

        // Save Occ Grid
        //fs::path dir = save_base_path_ / fs::path(shape.GetName());
        fs::path dir = save_base_path_;
        if (!fs::exists(dir)) {
          printf("Making path: %s\n", dir.string().c_str());
          fs::create_directories(dir);
        }

        std::string label = is_background ? "BACKGROUND":shape.GetName();

        char fn[1000];
        sprintf(fn, "%s_%08d.og", label.c_str(), class_counts_[label]++);
        fs::path path = dir / fs::path(fn);

        //og.Save(fn);
        DumpBin(og, path);
      }
    }

    // Save background examples
    int n_ex = kEntriesPerObj_;
    if (class_counts_["BACKGROUND"] > class_counts_["STAR"] &&
        class_counts_["BACKGROUND"] > class_counts_["BOX"]) {
      n_ex = 1;
    }

    for (int neg=0; neg<n_ex; neg++) {
      double x = unif(rand_engine_);
      double y = unif(rand_engine_);
      double z = 0;
      double t = 0;

      // Check if too close to existing object
      bool too_close = false;
      for (auto &shape : shapes) {
        const auto &center = shape.GetCenter();

        double dx = center(0) - x;
        double dy = center(1) - y;

        if (dx*dx + dy*dy < pr2) {
          too_close = true;
          break;
        }
      }

      if (!too_close) {
        // Build og and save
        og_builder_.SetPose(Eigen::Vector3d(x, y, z), t);
        rt::OccGrid og = og_builder_.GenerateOccGrid(scan);

        // Save Occ Grid
        //fs::path dir = save_base_path_ / fs::path("BACKGROUND");
        fs::path dir = save_base_path_;
        if (!fs::exists(dir)) {
          printf("Making path: %s\n", dir.string().c_str());
          fs::create_directories(dir);
        }

        char fn[1000];
        sprintf(fn, "BACKGROUND_%08d.og", class_counts_["BACKGROUND"]++);
        fs::path path = dir / fs::path(fn);

        //og.Save(fn);
        DumpBin(og, path);
      }
    }

    // Cleanup
    delete data;

    // Check for counts
    bool keep_going = false;
    for (const auto &kv : class_counts_) {
      if (kv.second < 1000000) {
        keep_going = true;
        break;
      }
    }

    if (!keep_going) {
      break;
    }
  }
}

void OccGridExtractor::DumpCsv(const rt::OccGrid &og, const fs::path &path) const {
  std::ofstream file(path.string());

  for (int i=-kPixelSize_+1; i<kPixelSize_; i++) {
    bool first = true;

    for (int j=-kPixelSize_+1; j<kPixelSize_; j++) {
      double p = og.GetProbability(rt::Location(i, j, 0));

      if (!first) {
        file << ",";
      }

      file << p;

      first = false;
    }

    file << std::endl;
  }

  file.close();
}

void OccGridExtractor::DumpBin(const rt::OccGrid &og, const fs::path &path) const {
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
