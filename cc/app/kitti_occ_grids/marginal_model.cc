#include "app/kitti_occ_grids/marginal_model.h"
#include <iostream>
#include <fstream>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace rt = library::ray_tracing;

namespace app {
namespace kitti_occ_grids {

MarginalModel::MarginalModel(const JointModel &jm) :
 resolution_(jm.GetResolution()) {

  int min_ij = -jm.GetNXY() / 2;
  int max_ij = min_ij + jm.GetNXY();

  int min_k = -jm.GetNZ() / 2;
  int max_k = min_k + jm.GetNZ();

  for (int i=min_ij; i < max_ij; i++) {
    for (int j=min_ij; j < max_ij; j++) {
      for (int k=min_k; k < max_k; k++) {
        rt::Location loc(i, j, k);

        model_.insert({loc, MarginalDistribution(jm, loc)});
      }
    }
  }
}


double MarginalModel::GetLogProbability(const rt::Location &loc, bool occ) const {
  auto it = model_.find(loc);
  if (it == model_.end()) {
    return 0.0;
  }

  return it->second.GetLogProbability(occ);
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
