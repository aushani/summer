#include "app/kitti_occ_grids/model.h"

#include <iostream>
#include <fstream>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace rt = library::ray_tracing;

namespace app {
namespace kitti_occ_grids {

Model::Model(double res) : resolution_(res) {

}

Model::Model() {
  resolution_ = 0.0;
}

void Model::MarkObservation(const rt::Location &loc, bool occu) {
  auto it = counts_.find(loc);
  if (it == counts_.end()) {
    counts_.insert(std::pair<rt::Location, Counter>(loc, Counter()));
    it = counts_.find(loc);
  }

  it->second.Count(occu);
}

double Model::GetProbability(const rt::Location &loc, bool occu) const {
  auto it = counts_.find(loc);
  if (it == counts_.end()) {
    return 0.0;
  }

  return it->second.GetProbability(occu);
}

void Model::MarkObservation(const rt::Location &loc, float lo) {
  MarkObservation(loc, lo > 0);
}

double Model::GetProbability(const rt::Location &loc, float lo) const {
  return GetProbability(loc, lo > 0);
}

const std::map<rt::Location, Model::Counter>& Model::GetCounts() const {
  return counts_;
}

double Model::GetResolution() const {
  return resolution_;
}

size_t Model::GetSupport(const rt::Location &loc) const {
  auto it = counts_.find(loc);
  if (it == counts_.end()) {
    return 0;
  }

  return it->second.GetTotalCount();
}

void Model::Save(const char* fn) const {
  std::ofstream ofs(fn);
  boost::archive::binary_oarchive oa(ofs);
  oa << (*this);
}

Model Model::Load(const char* fn) {
  Model m;
  std::ifstream ifs(fn);
  boost::archive::binary_iarchive ia(ifs);
  ia >> m;

  return m;
}

} // namespace kitti_occ_grids
} // namespace app
