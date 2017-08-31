#include "app/sim_world_occ_grids/occ_grid_extractor.h"

#include "library/ray_tracing/occ_grid.h"
#include "library/timer/timer.h"

namespace rt = library::ray_tracing;
namespace sw = library::sim_world;

namespace app {
namespace sim_world_occ_grids {

OccGridExtractor::OccGridExtractor(const std::string &save_base_fn) :
 og_builder_(10000, 0.30, 100.0),
 data_manager_(32, false, false),
 rand_engine_(std::chrono::system_clock::now().time_since_epoch().count()),
 save_base_fn_(save_base_fn) {
  og_builder_.ConfigureSizeInPixels(16, 16, 1);

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
      for (int ex=0; ex<kEntriesPerObj_; ex++) {
        // Jitter
        double dx = jitter_pos(rand_engine_);
        double dy = jitter_pos(rand_engine_);
        double dt = jitter_angle(rand_engine_);

        double x = shape.GetCenter()(0);
        double y = shape.GetCenter()(1);
        double z = 0;
        double t = -shape.GetAngle();

        og_builder_.SetPose(Eigen::Vector3d(x + dx, y + dy, z), t + dt);

        // Make occ grid
        rt::OccGrid og = og_builder_.GenerateOccGrid(scan);

        // Save Occ Grid
        char fn[1000];
        sprintf(fn, "%s/%s/%06d.og", save_base_fn_.c_str(), shape.GetName().c_str(), class_counts_[shape.GetName()]++);
        og.Save(fn);
      }
    }

    // Save background examples
    for (int neg=0; neg<kEntriesPerObj_; neg++) {
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
        char fn[1000];
        sprintf(fn, "%s/BACKGROUND/%06d.og", save_base_fn_.c_str(), class_counts_["BACKGROUND"]++);
        og.Save(fn);
      }
    }

    // Check for counts
    bool keep_going = false;
    for (const auto &kv : class_counts_) {
      if (kv.second < 100000) {
        keep_going = true;
        break;
      }
    }

    if (!keep_going) {
      break;
    }
  }
}


} // namespace sim_world_occ_grids
} // namespace app
