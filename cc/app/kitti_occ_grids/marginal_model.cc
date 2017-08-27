#include "app/kitti_occ_grids/marginal_model.h"

#include <iostream>
#include <fstream>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace rt = library::ray_tracing;

namespace app {
namespace kitti_occ_grids {

MarginalModel::MarginalModel(const JointModel &jm) :
 resolution_(jm.GetResolution()),
 n_xy_(jm.GetNXY()),
 n_z_(jm.GetNZ()),
 model_(n_xy_ * n_xy_ * n_z_) {

  int min_ij = -jm.GetNXY() / 2;
  int max_ij = min_ij + jm.GetNXY();

  int min_k = -jm.GetNZ() / 2;
  int max_k = min_k + jm.GetNZ();

  for (int i=min_ij; i < max_ij; i++) {
    for (int j=min_ij; j < max_ij; j++) {
      for (int k=min_k; k < max_k; k++) {
        rt::Location loc(i, j, k);
        int idx = GetIndex(loc);

        if (idx >= 0) {
          model_[idx] = MarginalDistribution(jm, loc);
        }
      }
    }
  }
}


double MarginalModel::GetLogProbability(const rt::Location &loc, bool occ) const {
  int idx = GetIndex(loc);
  if (idx >= 0) {
    return model_[idx].GetLogProbability(occ);
  } else {
    return 0.0;
  }
}

double MarginalModel::GetResolution() const {
  return resolution_;
}

int MarginalModel::GetNXY() const {
  return n_xy_;
}

int MarginalModel::GetNZ() const {
  return n_z_;
}

int MarginalModel::GetIndex(const rt::Location &loc) const {
  int i = loc.i + n_xy_ / 2;
  int j = loc.j + n_xy_ / 2;
  int k = loc.k + n_z_ / 2;

  if (i >= 0 || j >= 0 && k >= 0 &&
      i < n_xy_ && j < n_xy_ && k < n_z_) {
    return ((i * n_xy_) + j) * n_z_ + k;
  } else {
    return -1;
  }
}

void MarginalModel::Save(const char *fn) const {
  std::ofstream ofs(fn);
  boost::archive::binary_oarchive oa(ofs);
  oa << (*this);
}

MarginalModel MarginalModel::Load(const char *fn) {
  MarginalModel m;
  std::ifstream ifs(fn);
  boost::archive::binary_iarchive ia(ifs);
  ia >> m;

  return m;
}

} // namespace kitti_occ_grids
} // namespace app
