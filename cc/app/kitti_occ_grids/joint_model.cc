#include "app/kitti_occ_grids/joint_model.h"

#include <thread>
#include <iostream>
#include <fstream>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace rt = library::ray_tracing;

namespace app {
namespace kitti_occ_grids {

JointModel::JointModel() : resolution_(0.0), range_xy_(0.0), range_z_(0.0), n_xy_(0), n_z_(0) {
}

JointModel::JointModel(double range_xy, double range_z, double res) :
 resolution_(res), range_xy_(range_xy), range_z_(range_z),
 n_xy_(2*std::ceil(range_xy_ / resolution_) + 1),
 n_z_(2*std::ceil(range_z_ / resolution_) + 1),
 counts_(n_xy_*n_xy_*n_z_ * n_xy_*n_xy_*n_z_) {
  printf("Allocated %ld elements in mi model\n", n_xy_*n_xy_*n_z_ * n_xy_*n_xy_*n_z_);
}

void JointModel::MarkObservations(const rt::OccGrid &og) {
  BOOST_ASSERT(resolution_ == og.GetResolution());

  size_t sz = og.GetLocations().size();

  int num_threads = 64;
  std::vector<std::thread> threads;
  for (int i=0; i<num_threads; i++) {
    size_t start = i * sz / num_threads;
    size_t end = (i+1) * sz / num_threads;

    threads.emplace_back(&JointModel::MarkObservatonsWorker, this, og, start, end);
  }

  for (auto &t : threads) {
    t.join();
  }
}

void JointModel::MarkObservatonsWorker(const rt::OccGrid &og, size_t idx1_start, size_t idx1_end) {
  const auto &locs = og.GetLocations();
  const auto &los = og.GetLogOdds();

  for (size_t idx1 = idx1_start; idx1 < idx1_end; idx1++) {
    const auto &loc1 = locs[idx1];
    const auto &lo1 = los[idx1];

    if (!InRange(loc1)) {
      continue;
    }

    for (size_t idx2 = 0; idx2 < locs.size(); idx2++) {
      const auto &loc2 = locs[idx2];
      const auto &lo2 = los[idx2];

      if (!InRange(loc2)) {
        continue;
      }

      counts_[GetIndex(loc1, loc2)].Count(lo1, lo2);
      BOOST_ASSERT(GetNumObservations(loc1, loc2) > 0);
    }
  }
}

int JointModel::GetNXY() const {
  return n_xy_;
}

int JointModel::GetNZ() const {
  return n_z_;
}

size_t JointModel::GetIndex(const rt::Location &loc1, const rt::Location &loc2) const {
  int x1 = loc1.i + n_xy_/2;
  int y1 = loc1.j + n_xy_/2;
  int z1 = loc1.k + n_z_/2;

  BOOST_ASSERT(x1 >= 0 && x1 < n_xy_);
  BOOST_ASSERT(y1 >= 0 && y1 < n_xy_);
  BOOST_ASSERT(z1 >= 0 && z1 < n_z_);

  size_t idx1 =  (x1 * n_xy_ + y1) * n_z_ + z1;

  int x2 = loc2.i + n_xy_/2;
  int y2 = loc2.j + n_xy_/2;
  int z2 = loc2.k + n_z_/2;

  BOOST_ASSERT(x2 >= 0 && x2 < n_xy_);
  BOOST_ASSERT(y2 >= 0 && y2 < n_xy_);
  BOOST_ASSERT(z2 >= 0 && z2 < n_z_);

  size_t idx2 =  (x2 * n_xy_ + y2) * n_z_ + z2;

  return idx1 * (n_xy_ * n_xy_ * n_z_) + idx2;
}

bool JointModel::InRange(const rt::Location &loc) const {
  //double x = loc.i * resolution_;
  //double y = loc.j * resolution_;
  //double z = loc.k * resolution_;

  //return std::abs(x) < range_xy_ && std::abs(y) < range_xy_ && std::abs(z) < range_z_;

  int x = loc.i + n_xy_/2;
  int y = loc.j + n_xy_/2;
  int z = loc.k + n_z_/2;

  return x >= 0 && x < n_xy_ &&
         y >= 0 && y < n_xy_ &&
         z >= 0 && z < n_z_;
}

double JointModel::GetResolution() const {
  return resolution_;
}

double JointModel::ComputeMutualInformation(const rt::Location &loc1, const rt::Location &loc2) const {
  if (!InRange(loc1) || !InRange(loc2)) {
    return 0.0;
  }

  size_t idx = GetIndex(loc1, loc2);

  const Counter& c = counts_[idx];

  double mi = 0.0;

  for (int i=0; i<2; i++) {
    bool occ1 = i == 0;
    for (int j=0; j<2; j++) {
      bool occ2 = j == 0;

      double p_xy = c.GetProb(occ1, occ2);
      double p_x = c.GetProb1(occ1);
      double p_y = c.GetProb2(occ2);

      // check for tolerance
      if (p_xy < 1e-10) {
        continue;
      }

      mi += p_xy * log( p_xy / (p_x * p_y));
    }
  }

  return mi;
}

int JointModel::GetNumObservations(const rt::Location &loc1, const rt::Location &loc2) const {
  if (!InRange(loc1) || !InRange(loc2)) {
    return -1;
  }

  size_t idx = GetIndex(loc1, loc2);

  const Counter& c = counts_[idx];

  return c.GetTotalCount();
}

void JointModel::Save(const char *fn) const {
  std::ofstream ofs(fn);
  boost::archive::binary_oarchive oa(ofs);
  oa << (*this);
}

JointModel JointModel::Load(const char *fn) {
  JointModel m;
  std::ifstream ifs(fn);
  boost::archive::binary_iarchive ia(ifs);
  ia >> m;

  return m;
}

} // namespace kitti_occ_grids
} // namespace app
