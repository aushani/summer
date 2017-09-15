#include "library/ray_tracing/feature_occ_grid.h"

#include <thread>
#include <iostream>
#include <fstream>

#include <boost/assert.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <Eigen/Eigenvalues>

namespace library {
namespace ray_tracing {

FeatureOccGrid FeatureOccGrid::FromDevice(const DeviceFeatureOccGrid &dfog) {
  // Init
  FeatureOccGrid fog;

  // Get resolution
  fog.data_.resolution = dfog.GetResolution();

  // Init vectors
  fog.data_.locations.resize(dfog.size);
  fog.data_.log_odds.resize(dfog.size);

  fog.feature_locs_.resize(dfog.sz_features);
  fog.features_.resize(dfog.sz_features);

  // Get occ data
  cudaMemcpy(fog.data_.locations.data(), dfog.locs, sizeof(Location) * dfog.size, cudaMemcpyDeviceToHost);
  cudaMemcpy(fog.data_.log_odds.data(), dfog.los, sizeof(float) * dfog.size, cudaMemcpyDeviceToHost);
  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);

  // Get feature data
  cudaMemcpy(fog.feature_locs_.data(), dfog.feature_locs, sizeof(Location) * dfog.sz_features, cudaMemcpyDeviceToHost);
  cudaMemcpy(fog.features_.data(), dfog.features, sizeof(Feature) * dfog.sz_features, cudaMemcpyDeviceToHost);
  err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);

  return fog;
}

bool FeatureOccGrid::HasFeature(const Location &loc) const {
  std::vector<Location>::const_iterator it = std::lower_bound(feature_locs_.begin(), feature_locs_.end(), loc);
  return (it != feature_locs_.end() && (*it) == loc);
}

const std::vector<Location>& FeatureOccGrid::GetFeatureLocations() const {
  return feature_locs_;
}

const std::vector<Feature>& FeatureOccGrid::GetFeatures() const {
  return features_;
}

const Feature& FeatureOccGrid::GetFeature(const Location &loc) const {
  size_t pos = GetFeaturePos(loc);
  return features_[pos];
}

Eigen::Vector3f FeatureOccGrid::GetNormal(const Location &loc) const {
  const Feature &f = GetFeature(loc);

  float x = cos(f.phi) * cos(f.theta);
  float y = cos(f.phi) * sin(f.theta);
  float z = sin(f.phi);

  return Eigen::Vector3f(x, y, z);
  //return Eigen::Vector3f(loc.i, loc.j, loc.k).normalized();
}

size_t FeatureOccGrid::GetFeaturePos(const Location &loc) const {
  std::vector<Location>::const_iterator it = std::lower_bound(feature_locs_.begin(), feature_locs_.end(), loc);
  BOOST_ASSERT(it != feature_locs_.end() && (*it) == loc);

  size_t pos = it - feature_locs_.begin();

  return pos;
}

void FeatureOccGrid::Save(const char* fn) const {
  std::ofstream ofs(fn);
  boost::archive::binary_oarchive oa(ofs);
  oa << (*this);
}

FeatureOccGrid FeatureOccGrid::Load(const char* fn) {
  FeatureOccGrid fog;
  std::ifstream ifs(fn);
  boost::archive::binary_iarchive ia(ifs);
  ia >> fog;

  return fog;
}

}  // namespace ray_tracing
}  // namespace library
