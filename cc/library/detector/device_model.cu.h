#pragma once

#include <vector>

#include <boost/assert.hpp>

#include "library/feature/feature_model.h"
#include "library/feature/counter.h"
#include "library/ray_tracing/device_occ_grid.h"
#include "library/ray_tracing/occ_grid_location.h"

namespace ft = library::feature;
namespace rt = library::ray_tracing;

namespace library {
namespace detector {

//static constexpr int kShrinkage = 1;
static constexpr float kMinP = 0.20;
static constexpr float kMaxP = 0.80;

struct FeatureModelPoint {
  static constexpr int kMaxFeaturePoints = 1000; // This is lazy and gross!

  float log_ps_occu[2] = {0.0, 0.0};
  float log_ps_feature[kMaxFeaturePoints] = {0.0,};

  float angle_res = 0;
  int n_theta = 0;
  int n_phi = 0;

  __host__ void UpdateWith(const ft::Counter &c) {
    // Check for size
    BOOST_ASSERT(c.counts_features.size() <= kMaxFeaturePoints);

    // First do occu
    int c_t = c.GetCount(true);
    int c_f = c.GetCount(false);
    float denom = c_t + c_f;

    if (denom < 1) {
      log_ps_occu[0] = 0;
      log_ps_occu[1] = 0;
    } else {
      for (int i=0; i<2; i++) {
        bool occu = i==0;

        int count = c.GetCount(occu);
        double p = count / denom;

        if (p < kMinP) p = kMinP;
        if (p > kMaxP) p = kMaxP;

        log_ps_occu[GetIndex(occu)] = log(p);
      }
    }


    // Now do features
    for (size_t i = 0; i < c.counts_features.size(); i++) {
      if (c.total_feature_counts > 0) {
        int count = c.counts_features[i];
        double p = count / c.total_feature_counts;

        if (p < kMinP) p = kMinP;
        if (p > kMaxP) p = kMaxP;

        log_ps_feature[i] = log(p);
      } else {
        log_ps_feature[i] = 0;
      }
    }

    // Copy over params
    angle_res = c.angle_res;
    n_theta = c.n_theta;
    n_phi = c.n_phi;

    // Now we're done!
  }

  __host__ __device__ float GetLogP(bool occ) const {
    return log_ps_occu[GetIndex(occ)];
  }

  __host__ __device__ float GetLogP(float theta, float phi) const {
    return log_ps_feature[GetIndex(theta, phi)];
  }

 private:
  __host__ __device__ int GetIndex(bool occ) const {
    return occ ? 0:1;
  }

  __host__ __device__ int GetIndex(float theta, float phi) const {
    return ft::Counter::GetIndex(theta, phi, angle_res, n_theta, n_phi);
  }
};

struct DeviceModel {
 public:
  // On device
  FeatureModelPoint *fmp = nullptr;

  int n_xy = 0;
  int n_z = 0;
  int locs = 0;

  float res = 0;

  DeviceModel(const ft::FeatureModel &fm) {
    BuildModel(fm);

    size_t sz = sizeof(FeatureModelPoint) * locs;
    printf("Allocated %ld Mbytes on device for model\n", sz/(1024*1024));
  }

  void BuildModel(const ft::FeatureModel &fm) {
    n_xy = fm.GetNXY();
    n_z = fm.GetNZ();
    res = fm.GetResolution();

    locs = n_xy * n_xy * n_z;

    if (fmp == nullptr) {
      cudaMalloc(&fmp, locs * sizeof(FeatureModelPoint));
    }

    std::vector<FeatureModelPoint> h_fmp(locs);

    // Get all locations
    int min_ij = - (fm.GetNXY() / 2);
    int max_ij = min_ij + fm.GetNXY();

    int min_k = - (fm.GetNZ() / 2);
    int max_k = min_k + fm.GetNZ();

    //printf("\tFilling look up tables...\n");
    for (int i=min_ij; i < max_ij; i++) {
      for (int j=min_ij; j < max_ij; j++) {
        for (int k=min_k; k < max_k; k++) {
          rt::Location loc(i, j, k);
          int idx = GetIndex(loc);

          const ft::Counter &c = fm.GetCounter(loc);
          h_fmp[idx].UpdateWith(c);
        }
      }
    }

    // Send to device
    //printf("\tCopying to device...\n");
    cudaMemcpy(fmp, h_fmp.data(), locs*sizeof(FeatureModelPoint), cudaMemcpyHostToDevice);
  }

  void Cleanup() {
    if (fmp != nullptr) {
      cudaFree(fmp);
      fmp = nullptr;
    }
  }

  __host__ __device__ float GetLogP(const rt::Location &loc, bool occ) const {
    int idx = GetIndex(loc);
    if (idx < 0) {
      return 0.0;
    }

    return fmp[idx].GetLogP(occ);
  }

  __host__ __device__ float GetLogP(const rt::Location &loc, float theta, float phi) const {
    int idx = GetIndex(loc);
    if (idx < 0) {
      return 0.0;
    }

    return fmp[idx].GetLogP(theta, phi);
  }

  __host__ __device__ int GetIndex(const rt::Location &loc) const {
    int x = loc.i + n_xy / 2;
    int y = loc.j + n_xy / 2;
    int z = loc.k + n_z / 2;

    if (x < 0 || x >= n_xy) {
      return -1;
    }

    if (y < 0 || y >= n_xy) {
      return -1;
    }

    if (z < 0 || z >= n_z) {
      return -1;
    }

    return (x*n_xy + y)*n_z + z;
  }
};

} // namespace detector
} // namespace library
