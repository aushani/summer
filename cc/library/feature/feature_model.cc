#include "library/feature/feature_model.h"

#include <thread>
#include <iostream>
#include <fstream>

#include <boost/assert.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace rt = library::ray_tracing;

namespace library {
namespace feature {

FeatureModel::FeatureModel() : FeatureModel(0, 0, 0) {}

FeatureModel::FeatureModel(double range_xy, double range_z, double res) :
 resolution_(res), range_xy_(range_xy), range_z_(range_z),
 n_xy_(2*std::ceil(range_xy_ / resolution_) + 1),
 n_z_(2*std::ceil(range_z_ / resolution_) + 1),
 counters_(n_xy_ * n_xy_ * n_z_, Counter(kAngleRes)) {
  printf("Allocated %ld elements\n", counters_.size());
}

void FeatureModel::MarkObservations(const rt::FeatureOccGrid &fog) {
  BOOST_ASSERT(fog.GetResolution() == GetResolution());

  size_t sz_occu = fog.GetLocations().size();
  size_t sz_feat = fog.GetFeatureLocations().size();

  int num_threads = 24;

  std::vector<std::thread> threads;
  for (int i=0; i<num_threads; i++) {
    size_t start = i * sz_occu / num_threads;
    size_t end = (i+1) * sz_occu / num_threads;
    threads.emplace_back(&FeatureModel::MarkOccuWorker, this, fog, start, end);
  }

  for (int i=0; i<num_threads; i++) {
    size_t start = i * sz_feat / num_threads;
    size_t end = (i+1) * sz_feat/ num_threads;
    threads.emplace_back(&FeatureModel::MarkFeaturesWorker, this, fog, start, end);
  }

  for (auto &t : threads) {
    t.join();
  }
}

int FeatureModel::GetCount(const rt::Location &loc, bool occ) const {
  BOOST_ASSERT(InRange(loc));

  return counters_[GetIndex(loc)].GetCount(occ);
}

int FeatureModel::GetCount(const rt::Location &loc, float theta, float phi) const {
  BOOST_ASSERT(InRange(loc));

  return counters_[GetIndex(loc)].GetCount(theta, phi);
}

int FeatureModel::GetNumOccuObservations(const rt::Location &loc) const {
  BOOST_ASSERT(InRange(loc));
  return counters_[GetIndex(loc)].GetNumOccuObservations();
}

int FeatureModel::GetNumFeatureObservations(const rt::Location &loc) const {
  BOOST_ASSERT(InRange(loc));
  return counters_[GetIndex(loc)].GetNumFeatureObservations();
}

int FeatureModel::GetMode(const rt::Location &loc, float *theta, float *phi) const {
  BOOST_ASSERT(InRange(loc));
  return counters_[GetIndex(loc)].GetMode(theta, phi);
}

double FeatureModel::GetResolution() const {
  return resolution_;
}

size_t FeatureModel::GetNXY() const {
  return n_xy_;
}

size_t FeatureModel::GetNZ() const {
  return n_z_;
}

bool FeatureModel::InRange(const rt::Location &loc) const {
  int x = loc.i + n_xy_ / 2;
  int y = loc.j + n_xy_ / 2;
  int z = loc.k + n_z_ / 2;

  return x >= 0 && x < n_xy_ &&
         y >= 0 && y < n_xy_ &&
         z >= 0 && z < n_z_;
}

void FeatureModel::MarkOccuWorker(const rt::FeatureOccGrid &fog, size_t idx_start, size_t idx_end) {
  const auto &locs = fog.GetLocations();
  const auto &los = fog.GetLogOdds();

  for (size_t idx = idx_start; idx < idx_end; idx++) {
    const auto &loc = locs[idx];
    if (!InRange(loc)) {
      continue;
    }

    bool occ = los[idx] > 0.0;
    counters_[GetIndex(loc)].Count(occ);
  }
}

void FeatureModel::MarkFeaturesWorker(const rt::FeatureOccGrid &fog, size_t idx_start, size_t idx_end) {
  const auto &locs = fog.GetFeatureLocations();
  const auto &feats = fog.GetFeatures();

  for (size_t idx = idx_start; idx < idx_end; idx++) {
    const auto &loc = locs[idx];
    if (!InRange(loc)) {
      continue;
    }

    const rt::Feature &f = feats[idx];
    counters_[GetIndex(loc)].Count(f.theta, f.phi);
  }
}

size_t FeatureModel::GetIndex(const rt::Location &loc) const {
  int x = loc.i + n_xy_ / 2;
  int y = loc.j + n_xy_ / 2;
  int z = loc.k + n_z_ / 2;
  size_t idx = (x * n_xy_ + y) * n_z_ + z;

  BOOST_ASSERT(x >= 0 || x < n_xy_);
  BOOST_ASSERT(y >= 0 || y < n_xy_);
  BOOST_ASSERT(z >= 0 || z < n_z_);

  return idx;
}

void FeatureModel::Save(const char *fn) const {
  std::ofstream ofs(fn);
  boost::archive::binary_oarchive oa(ofs);
  oa << (*this);
}

FeatureModel FeatureModel::Load(const char *fn) {
  FeatureModel fm;

  std::ifstream ifs(fn);
  boost::archive::binary_iarchive ia(ifs);
  ia >> fm;

  return fm;
}

} // namespace chow_liu_tree
} // namespace library
