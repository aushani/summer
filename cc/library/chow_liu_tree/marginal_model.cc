#include "library/chow_liu_tree/marginal_model.h"

#include <thread>
#include <iostream>
#include <fstream>

#include <boost/assert.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace rt = library::ray_tracing;

namespace library {
namespace chow_liu_tree {

MarginalModel::MarginalModel() : MarginalModel(0, 0, 0) {
}

MarginalModel::MarginalModel(const JointModel &jm) :
 MarginalModel(jm.GetRangeXY(), jm.GetRangeZ(), jm.GetResolution()) {

  // Get all locations
  int min_ij = - (jm.GetNXY() / 2);
  int max_ij = min_ij + jm.GetNXY();

  int min_k = - (jm.GetNZ() / 2);
  int max_k = min_k + jm.GetNZ();

  for (int i=min_ij; i < max_ij; i++) {
    for (int j=min_ij; j < max_ij; j++) {
      for (int k=min_k; k < max_k; k++) {
        rt::Location loc(i, j, k);

        int c_t = jm.GetCount(loc, true);
        int c_f = jm.GetCount(loc, false);

        counts_[GetIndex(loc)].SetCount(true, c_t);
        counts_[GetIndex(loc)].SetCount(false, c_f);
      }
    }
  }
}

MarginalModel::MarginalModel(double range_xy, double range_z, double res) :
 resolution_(res), range_xy_(range_xy), range_z_(range_z),
 n_xy_(2*std::ceil(range_xy_ / resolution_) + 1),
 n_z_(2*std::ceil(range_z_ / resolution_) + 1),
 counts_(n_xy_ * n_xy_ * n_z_) {
  printf("Allocated %ld elements\n", counts_.size());
}

void MarginalModel::MarkObservations(const rt::OccGrid &og) {
  BOOST_ASSERT(og.GetResolution() == GetResolution());

  size_t sz = og.GetLocations().size();

  int num_threads = 48;

  std::vector<std::thread> threads;
  for (int i=0; i<num_threads; i++) {
    size_t start = i * sz / num_threads;
    size_t end = (i+1) * sz / num_threads;
    threads.emplace_back(&MarginalModel::MarkObservationsWorker, this, og, start, end);
  }

  for (auto &t : threads) {
    t.join();
  }
}

int MarginalModel::GetCount(const rt::Location &loc, bool occ) const {
  BOOST_ASSERT(InRange(loc));

  return counts_[GetIndex(loc)].GetCount(occ);
}

void MarginalModel::SetCount(const rt::Location &loc, bool occ, int count) {
  BOOST_ASSERT(InRange(loc));

  counts_[GetIndex(loc)].SetCount(occ, count);
}

int MarginalModel::GetNumObservations(const rt::Location &loc) const {
  BOOST_ASSERT(InRange(loc));
  return counts_[GetIndex(loc)].GetTotalCount();
}

double MarginalModel::Evaluate(const rt::OccGrid &og) const {
  const auto &locs = og.GetLocations();
  const auto &los = og.GetLogOdds();

  size_t sz = locs.size();

  double log_p = 0.0;

  for (size_t i = 0; i < sz; i++) {
    const auto &loc = locs[i];
    const bool occ = los[i] > 0;

    if (InRange(loc)) {
      size_t idx = GetIndex(loc);
      const auto &counter = counts_[idx];

      double denom = counter.GetTotalCount();
      double c = counter.GetCount(occ);

      log_p += log(c / denom);
    }
  }

  return log_p;
}

double MarginalModel::GetResolution() const {
  return resolution_;
}

size_t MarginalModel::GetNXY() const {
  return n_xy_;
}

size_t MarginalModel::GetNZ() const {
  return n_z_;
}

bool MarginalModel::InRange(const rt::Location &loc) const {
  int x = loc.i + n_xy_ / 2;
  int y = loc.j + n_xy_ / 2;
  int z = loc.k + n_z_ / 2;

  return x >= 0 && x < n_xy_ &&
         y >= 0 && y < n_xy_ &&
         z >= 0 && z < n_z_;
}

void MarginalModel::MarkObservationsWorker(const rt::OccGrid &og, size_t idx_start, size_t idx_end) {
  const auto &locs = og.GetLocations();
  const auto &los = og.GetLogOdds();

  for (size_t idx = idx_start; idx < idx_end; idx++) {
    const auto &loc = locs[idx];
    if (!InRange(loc)) {
      continue;
    }

    bool occ = los[idx] > 0.0;
    counts_[GetIndex(loc)].Count(occ);
  }
}

size_t MarginalModel::GetIndex(const rt::Location &loc) const {
  int x = loc.i + n_xy_ / 2;
  int y = loc.j + n_xy_ / 2;
  int z = loc.k + n_z_ / 2;
  size_t idx = (x * n_xy_ + y) * n_z_ + z;

  BOOST_ASSERT(x >= 0 || x < n_xy_);
  BOOST_ASSERT(y >= 0 || y < n_xy_);
  BOOST_ASSERT(z >= 0 || z < n_z_);

  return idx;
}

void MarginalModel::Save(const char *fn) const {
  std::ofstream ofs(fn);
  boost::archive::binary_oarchive oa(ofs);
  oa << (*this);
}

MarginalModel MarginalModel::Load(const char *fn) {
  MarginalModel mm;

  std::ifstream ifs(fn);
  boost::archive::binary_iarchive ia(ifs);
  ia >> mm;

  return mm;
}

} // namespace chow_liu_tree
} // namespace library
