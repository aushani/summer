#include "library/ray_tracing/dense_occ_grid.h"

#include <cmath>
#include <thread>

namespace library {
namespace ray_tracing {

DenseOccGrid::DenseOccGrid(const OccGrid &og, float max_x, float max_y, float max_z) :
 nx_(2*ceil(max_x / og.GetResolution()) + 1),
 ny_(2*ceil(max_y / og.GetResolution()) + 1),
 nz_(2*ceil(max_z / og.GetResolution()) + 1),
 resolution_(og.GetResolution()),
 log_odds_(nx_*ny_*nz_, 0.0f) {

  size_t num_vals = og.GetLocations().size();
  PopulateDenseWorker(og, 0, num_vals);

  //std::vector<std::thread> threads;
  //for (size_t t_id = 0; t_id < kNumThreads; t_id++) {
  //  size_t i0 = t_id * num_vals / kNumThreads;
  //  size_t i1 = (t_id + 1) * num_vals / kNumThreads;

  //  threads.push_back(std::thread(&DenseOccGrid::PopulateDenseWorker, this, og, i0, i1));
  //}

  //for (auto &t : threads) {
  //  t.join();
  //}
}

float DenseOccGrid::GetLogOdds(const Location &loc) const {
  if (!InRange(loc)) {
    return 0.0f; // unknown
  }

  size_t idx = GetIndex(loc);
  return log_odds_[idx];
}

float DenseOccGrid::GetProbability(const Location &loc) const {
  float lo = GetLogOdds(loc);
  return 1 / (1 + exp(-lo));
}

bool DenseOccGrid::InRange(const Location &loc) const {
  return std::abs(loc.i) < nx_/2 &&
         std::abs(loc.j) < ny_/2 &&
         std::abs(loc.k) < nz_/2;
}


size_t DenseOccGrid::GetIndex(const Location &loc) const {
  size_t ix = loc.i + nx_/2;
  size_t iy = loc.j + ny_/2;
  size_t iz = loc.k + nz_/2;

  return (ix*ny_ + iy)*nz_ + iz;
}

void DenseOccGrid::PopulateDenseWorker(const OccGrid &og, size_t i0, size_t i1) {
  auto &locs = og.GetLocations();
  auto &los = og.GetLogOdds();

  for (size_t i = i0; i < i1; i++) {
    auto &loc = locs[i];

    if (!InRange(loc)) {
      continue;
    }

    size_t idx = GetIndex(loc);
    log_odds_[idx] = los[i];
  }
}

float DenseOccGrid::GetResolution() const {
  return resolution_;
}

}  // namespace ray_tracing
}  // namespace library
