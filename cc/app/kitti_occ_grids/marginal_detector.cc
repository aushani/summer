#include "app/kitti_occ_grids/marginal_detector.h"

#include <boost/assert.hpp>

#include "library/timer/timer.h"

namespace rt = library::ray_tracing;

namespace app {
namespace kitti_occ_grids {

MarginalDetector::MarginalDetector(double resolution) : resolution_(resolution) {

}

void MarginalDetector::AddModel(const std::string &classname, const MarginalModel &mm) {
  BOOST_ASSERT(std::abs(mm.GetResolution() - resolution_) < 0.001);

  classes_.push_back(classname);
  models_.insert({classname, mm});
}

DetectionMap MarginalDetector::RunDetector(const rt::OccGrid &og) const {
  BOOST_ASSERT(std::abs(og.GetResolution() - resolution_) < 0.001);

  DetectionMap dm(100.0, 100.0, resolution_, classes_);

  const auto &locs = og.GetLocations();
  const auto &los = og.GetLogOdds();

  // For every model
  for (const auto it : models_) {
    const auto &classname = it.first;
    const auto &mm = it.second;

    // Get support size
    int min_ij = -mm.GetNXY() / 2;
    int max_ij = min_ij + mm.GetNXY();

    int min_k = -mm.GetNZ() / 2;
    int max_k = min_k + mm.GetNZ();

    // Process each observation
    for (size_t idx_og = 0; idx_og < locs.size(); idx_og++) {
      library::timer::Timer t;

      const auto &loc = locs[idx_og];
      float lo = los[idx_og];

      if (loc.k < min_k || loc.k >= max_k) {
        continue;
      }

      // Where does this observation have support?
      for (int di=min_ij; di < max_ij; di++) {
        for (int dj=min_ij; dj < max_ij; dj++) {
          // Get the location w.r.t. model
          rt::Location loc_model(di, dj, loc.k);

          double update = mm.GetLogProbability(loc_model, lo > 0);
          dm.Update(loc.i - di, loc.j - dj, classname, update);
        }
      }
    }
  }

  return dm;
}

} // namespace kitti_occ_grids
} // namespace app
