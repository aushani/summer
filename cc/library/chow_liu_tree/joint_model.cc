#include "library/chow_liu_tree/joint_model.h"

#include <thread>
#include <iostream>
#include <fstream>

#include <boost/assert.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace rt = library::ray_tracing;

namespace library {
namespace chow_liu_tree {

JointModel::JointModel() : JointModel(0, 0, 0) {
}

JointModel::JointModel(double range_xy, double range_z, double res) :
 resolution_(res), range_xy_(range_xy), range_z_(range_z),
 n_xy_(2*std::ceil(range_xy_ / resolution_) + 1),
 n_z_(2*std::ceil(range_z_ / resolution_) + 1),
 n_loc_(n_xy_ * n_xy_ * n_z_),
 counts_(n_loc_ * n_loc_) {
  printf("Allocated %ld elements\n", counts_.size());
}

void JointModel::MarkObservations(const rt::OccGrid &og) {
  BOOST_ASSERT(og.GetResolution() == GetResolution());

  size_t sz = og.GetLocations().size();

  int num_threads = 48;

  std::vector<std::thread> threads;
  for (int i=0; i<num_threads; i++) {
    size_t start = i * sz / num_threads;
    size_t end = (i+1) * sz / num_threads;
    threads.emplace_back(&JointModel::MarkObservationsWorker, this, og, start, end);
  }

  for (auto &t : threads) {
    t.join();
  }
}

int JointModel::GetCount(const rt::Location &loc, bool occ) const {
  return GetCount(loc, occ, loc, occ);
}

void JointModel::SetCount(const rt::Location &loc, bool occ, int count) {
  SetCount(loc, occ, loc, occ, count);
}

int JointModel::GetCount(const rt::Location &loc1, bool occ1, const rt::Location &loc2, bool occ2) const {
  BOOST_ASSERT(InRange(loc1));
  BOOST_ASSERT(InRange(loc2));

  size_t idx = GetIndex(loc1, loc2);

  return counts_[idx].GetCount(occ1, occ2);
}

void JointModel::SetCount(const rt::Location &loc1, bool occ1, const rt::Location &loc2, bool occ2, int count) {
  BOOST_ASSERT(InRange(loc1));
  BOOST_ASSERT(InRange(loc2));

  size_t idx = GetIndex(loc1, loc2);

  return counts_[idx].SetCount(occ1, occ2, count);
}

int JointModel::GetNumObservations(const rt::Location &loc1) const {
  return GetNumObservations(loc1, loc1);
}

int JointModel::GetNumObservations(const rt::Location &loc1, const rt::Location &loc2) const {
  BOOST_ASSERT(InRange(loc1));
  BOOST_ASSERT(InRange(loc2));

  size_t idx = GetIndex(loc1, loc2);

  return counts_[idx].GetTotalCount();
}

double JointModel::GetMutualInformation(const rt::Location &loc1, const rt::Location &loc2) const {
  BOOST_ASSERT(InRange(loc1));
  BOOST_ASSERT(InRange(loc2));

  size_t idx = GetIndex(loc1, loc2);

  return counts_[idx].GetMutualInformation();
}

double JointModel::GetResolution() const {
  return resolution_;
}

size_t JointModel::GetNXY() const {
  return n_xy_;
}

size_t JointModel::GetNZ() const {
  return n_z_;
}

double JointModel::GetRangeXY() const {
  return range_xy_;
}

double JointModel::GetRangeZ() const {
  return range_z_;
}

bool JointModel::InRange(const rt::Location &loc) const {
  int x = loc.i + n_xy_ / 2;
  int y = loc.j + n_xy_ / 2;
  int z = loc.k + n_z_ / 2;

  return x >= 0 && x < n_xy_ &&
         y >= 0 && y < n_xy_ &&
         z >= 0 && z < n_z_;
}

void JointModel::MarkObservationsWorker(const rt::OccGrid &og, size_t idx1_start, size_t idx1_end) {
  const auto &locs = og.GetLocations();
  const auto &los = og.GetLogOdds();

  for (size_t idx1 = idx1_start; idx1 < idx1_end; idx1++) {
    const auto &loc1 = locs[idx1];
    if (!InRange(loc1)) {
      continue;
    }

    bool occ1 = los[idx1] > 0.0;

    for (size_t idx2 = 0; idx2 < locs.size(); idx2++) {
      const auto &loc2 = locs[idx2];

      if (!InRange(loc2)) {
        continue;
      }

      bool occ2 = los[idx2] > 0.0;

      counts_[GetIndex(loc1, loc2)].Count(occ1, occ2);
    }
  }
}

size_t JointModel::GetIndex(const rt::Location &loc1, const rt::Location &loc2) const {
  int x1 = loc1.i + n_xy_ / 2;
  int y1 = loc1.j + n_xy_ / 2;
  int z1 = loc1.k + n_z_ / 2;
  size_t idx1 = (x1 * n_xy_ + y1) * n_z_ + z1;

  BOOST_ASSERT(x1 >= 0 || x1 < n_xy_);
  BOOST_ASSERT(y1 >= 0 || y1 < n_xy_);
  BOOST_ASSERT(z1 >= 0 || z1 < n_z_);

  int x2 = loc2.i + n_xy_ / 2;
  int y2 = loc2.j + n_xy_ / 2;
  int z2 = loc2.k + n_z_ / 2;
  size_t idx2 = (x2 * n_xy_ + y2) * n_z_ + z2;

  BOOST_ASSERT(x2 >= 0 || x2 < n_xy_);
  BOOST_ASSERT(y2 >= 0 || y2 < n_xy_);
  BOOST_ASSERT(z2 >= 0 || z2 < n_z_);

  return idx1 * n_loc_ + idx2;
}

void JointModel::Save(const char *fn) const {
  std::ofstream ofs(fn);
  boost::archive::binary_oarchive oa(ofs);
  oa << (*this);
}

JointModel JointModel::Load(const char *fn) {
  JointModel jm;

  std::ifstream ifs(fn);
  boost::archive::binary_iarchive ia(ifs);
  ia >> jm;

  return jm;
}

} // namespace chow_liu_tree
} // namespace library
