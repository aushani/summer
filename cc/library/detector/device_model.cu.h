#pragma once

#include <vector>

#include <boost/assert.hpp>

#include "library/chow_liu_tree/marginal_model.h"
#include "library/ray_tracing/device_occ_grid.h"
#include "library/ray_tracing/occ_grid_location.h"

namespace clt = library::chow_liu_tree;
namespace rt = library::ray_tracing;

namespace library {
namespace detector {

static constexpr int kShrinkage = 1000;

struct Marginal {
  float log_ps[2] = {0.0, 0.0};

  __host__ __device__ float GetLogP(bool occ) const {
    return log_ps[GetIndex(occ)];
  }

  __host__ __device__ int GetCount(bool occ) const {
    return counts_[GetIndex(occ)] + kShrinkage;
  }

  __host__ __device__ int GetActualCount(bool occ) const {
    return counts_[GetIndex(occ)];
  }

  __host__ __device__ void SetCount(bool occ, int count) {
    counts_[GetIndex(occ)] = count;

    // Update log p
    UpdateLogP();
  }

  __host__ __device__ void IncrementCount(bool occ) {
    counts_[GetIndex(occ)]++;

    // Update log p
    UpdateLogP();
  }

 private:
  int counts_[2] = {0, 0};

  __host__ __device__ int GetIndex(bool occ) const {
    return occ ? 0:1;
  }

  __host__ __device__ void UpdateLogP() {
    float counts_total = GetCount(true) + GetCount(false);

    for (int i=0; i<2; i++) {
      bool occ = (i == 0);
      int idx = GetIndex(occ);

      log_ps[idx] = log(GetCount(occ) / counts_total);
    }
  }
};

// Forward declarations
typedef struct DeviceModel DeviceModel;
__global__ void UpdateModelKernel(DeviceModel model, const rt::DeviceOccGrid dog);

struct DeviceModel {
 public:
  // On device
  Marginal *marginals = nullptr;

  int n_xy = 0;
  int n_z = 0;
  int locs = 0;

  float res = 0;

  DeviceModel(const clt::MarginalModel &mm) {
    BuildModel(mm);

    size_t sz = sizeof(Marginal) * locs;
    printf("Allocated %ld KBytes on device for model\n", sz/(1024));
  }

  void UpdateModel(const rt::DeviceOccGrid &dog) {
    int threads = 128;
    int blocks = std::ceil(dog.size / static_cast<double>(threads));

    UpdateModelKernel<<<blocks, threads>>>((*this), dog);
    cudaError_t err = cudaDeviceSynchronize();
    BOOST_ASSERT(err == cudaSuccess);
  }

  void LoadIntoMarginalModel(clt::MarginalModel *mm) {
    // Check dimensions
    BOOST_ASSERT(n_xy == mm->GetNXY());
    BOOST_ASSERT(n_z == mm->GetNZ());

    std::vector<Marginal> h_marg(locs);

    cudaError_t err = cudaMemcpy(h_marg.data(), marginals, locs*sizeof(Marginal), cudaMemcpyDeviceToHost);
    BOOST_ASSERT(err == cudaSuccess);

    // Get all locations
    int min_ij = - (mm->GetNXY() / 2);
    int max_ij = min_ij + mm->GetNXY();

    int min_k = - (mm->GetNZ() / 2);
    int max_k = min_k + mm->GetNZ();

    for (int i=min_ij; i < max_ij; i++) {
      for (int j=min_ij; j < max_ij; j++) {
        for (int k=min_k; k < max_k; k++) {
          rt::Location loc(i, j, k);

          int idx = GetIndex(loc);
          BOOST_ASSERT(idx >= 0);

          Marginal &m = h_marg[idx];

          for (int i_occ = 0; i_occ<2; i_occ++) {
            bool occ = i_occ == 0;
            int my_count = m.GetActualCount(occ);

            mm->SetCount(loc, occ, my_count);
          }

        }
      }
    }

  }

  void BuildModel(const clt::MarginalModel &mm) {
    n_xy = mm.GetNXY();
    n_z = mm.GetNZ();
    res = mm.GetResolution();

    locs = n_xy * n_xy * n_z;

    if (marginals == nullptr) {
      cudaMalloc(&marginals, locs * sizeof(Marginal));
    }

    std::vector<Marginal> h_marg(locs);

    // Get all locations
    int min_ij = - (mm.GetNXY() / 2);
    int max_ij = min_ij + mm.GetNXY();

    int min_k = - (mm.GetNZ() / 2);
    int max_k = min_k + mm.GetNZ();

    //printf("\tFilling look up tables...\n");
    for (int i=min_ij; i < max_ij; i++) {
      for (int j=min_ij; j < max_ij; j++) {
        for (int k=min_k; k < max_k; k++) {
          rt::Location loc(i, j, k);

          // Marginal
          int idx = GetIndex(loc);
          int c_t = mm.GetCount(loc, true);
          int c_f = mm.GetCount(loc, false);
          h_marg[idx].SetCount(true, c_t);
          h_marg[idx].SetCount(false, c_f);

        }
      }
    }

    // Send to device
    //printf("\tCopying to device...\n");
    cudaMemcpy(marginals, h_marg.data(), locs*sizeof(Marginal), cudaMemcpyHostToDevice);
  }

  void Cleanup() {
    if (marginals != nullptr) {
      cudaFree(marginals);
      marginals = nullptr;
    }
  }

  __host__ __device__ float GetLogP(const rt::Location &loc, bool occ) const {
    int idx = GetIndex(loc);
    if (idx < 0) {
      return 0.0;
    }

    return marginals[idx].GetLogP(occ);
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
