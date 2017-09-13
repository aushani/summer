#pragma once

#include <vector>

#include <boost/assert.hpp>

#include "library/chow_liu_tree/joint_model.h"
#include "library/ray_tracing/device_occ_grid.h"
#include "library/ray_tracing/occ_grid_location.h"

namespace clt = library::chow_liu_tree;
namespace rt = library::ray_tracing;

namespace library {
namespace detector {

static constexpr int kShrinkage = 1;
static constexpr float kMinP = 0.20;
static constexpr float kMaxP = 0.80;

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

      float p = GetCount(occ) / counts_total;

      // Clip
      if (p > kMaxP) p = kMaxP;
      if (p < kMinP) p = kMinP;

      log_ps[idx] = log(p);
    }
  }
};

struct Conditional {
  float log_ps[4] = {0.0, 0.0, 0.0, 0.0};
  float mutual_information_ = 0;

  __host__ __device__ float GetLogP(bool occ_eval, bool occ_given) const {
    return log_ps[GetIndex(occ_eval, occ_given)];
  }

  __host__ __device__ int GetCount(bool occ_eval, bool occ_given) const {
    return counts_[GetIndex(occ_eval, occ_given)] + kShrinkage;
  }

  __host__ __device__ int GetActualCount(bool occ_eval, bool occ_given) const {
    return counts_[GetIndex(occ_eval, occ_given)];
  }

  __host__ __device__ void SetCount(bool occ_eval, bool occ_given, int count) {
    counts_[GetIndex(occ_eval, occ_given)] = count;

    // Update log p and mi
    Update();
  }

  __host__ __device__ void IncrementCount(bool occ_eval, bool occ_given) {
    counts_[GetIndex(occ_eval, occ_given)]++;

    // Update log p and mi
    Update();
  }

  __host__ __device__ float GetMutualInformation() const {
    return mutual_information_;
  }

 private:
  int counts_[4] = {0, 0, 0, 0};

  __host__ __device__ int GetIndex(bool occ_eval, bool occ_given) const {
    int idx = 0;
    if (occ_eval) {
      idx += 1;
    }

    if (occ_given) {
      idx += 2;
    }

    return idx;
  }

  __host__ __device__ void Update() {
    // Update log prob's
    for (int i=0; i<2; i++) {
      bool occ_eval = (i == 0);
      for (int j=0; j<2; j++) {
        bool occ_given = (j == 0);

        int idx = GetIndex(occ_eval, occ_given);

        int my_count = GetCount(occ_eval, occ_given);
        int other_count = GetCount(!occ_eval, occ_given);
        float denom = my_count + other_count;

        float p = my_count / denom;

        // Clip
        if (p > kMaxP) p = kMaxP;
        if (p < kMinP) p = kMinP;

        log_ps[idx] = log(p);
      }
    }

    // Update mutual information
    float counts_total = 0;
    for (int i=0; i<2; i++) {
      bool occ_eval = (i == 0);
      for (int j=0; j<2; j++) {
        bool occ_given = (j == 0);
        counts_total += GetCount(occ_eval, occ_given);
      }
    }

    mutual_information_ = 0;

    for (int i=0; i<2; i++) {
      bool occ_eval = (i == 0);
      for (int j=0; j<2; j++) {
        bool occ_given = (j == 0);

        float p_xy = GetCount(occ_eval, occ_given) / counts_total;

        float p_x = (GetCount(occ_eval, occ_given) + GetCount(occ_eval, !occ_given)) / counts_total;
        float p_y = (GetCount(occ_eval, occ_given) + GetCount(!occ_eval, occ_given)) / counts_total;

        mutual_information_ += p_xy * log(p_xy / (p_x * p_y));
      }
    }
  }
};

// Forward declarations
typedef struct DeviceModel DeviceModel;
__global__ void UpdateModelKernel(DeviceModel model, const rt::DeviceOccGrid dog);

struct DeviceModel {
 public:
  // On device
  Conditional *conditionals = nullptr;
  Marginal *marginals = nullptr;

  int n_xy = 0;
  int n_z = 0;
  int locs = 0;

  float res = 0;

  DeviceModel(const clt::JointModel &jm) {
    BuildModel(jm);

    size_t sz = sizeof(Conditional) * locs * locs + sizeof(Marginal) * locs;
    printf("Allocated %ld Mbytes on device for model\n", sz/(1024*1024));
  }

  void UpdateModel(const rt::DeviceOccGrid &dog) {
    int threads = 128;
    int blocks = std::ceil(dog.size / static_cast<double>(threads));

    UpdateModelKernel<<<blocks, threads>>>((*this), dog);
    cudaError_t err = cudaDeviceSynchronize();
    BOOST_ASSERT(err == cudaSuccess);
  }

  void LoadIntoJointModel(clt::JointModel *jm) {
    // Check dimensions
    BOOST_ASSERT(n_xy == jm->GetNXY());
    BOOST_ASSERT(n_z == jm->GetNZ());

    size_t sz = locs*locs;

    std::vector<Conditional> h_cond(sz);

    cudaMemcpy(h_cond.data(), conditionals, sz*sizeof(Conditional), cudaMemcpyDeviceToHost);

    // Get all locations
    int min_ij = - (jm->GetNXY() / 2);
    int max_ij = min_ij + jm->GetNXY();

    int min_k = - (jm->GetNZ() / 2);
    int max_k = min_k + jm->GetNZ();

    for (int i1=min_ij; i1 < max_ij; i1++) {
      for (int j1=min_ij; j1 < max_ij; j1++) {
        for (int k1=min_k; k1 < max_k; k1++) {
          rt::Location loc_eval(i1, j1, k1);

          for (int i2=min_ij; i2 < max_ij; i2++) {
            for (int j2=min_ij; j2 < max_ij; j2++) {
              for (int k2=min_k; k2 < max_k; k2++) {
                rt::Location loc_given(i2, j2, k2);

                int idx = GetIndex(loc_eval, loc_given);
                BOOST_ASSERT(idx >= 0);

                const Conditional &c = h_cond[idx];

                for (int i_eval = 0; i_eval<2; i_eval++) {
                  bool eval = i_eval == 0;
                  for (int i_given = 0; i_given<2; i_given++) {
                    bool given = i_given == 0;

                    int my_count = c.GetActualCount(eval, given);
                    jm->SetCount(loc_eval, eval, loc_given, given, my_count);
                  }
                }

              }
            }
          }

        }
      }
    }

  }

  void BuildModel(const clt::JointModel &jm) {
    n_xy = jm.GetNXY();
    n_z = jm.GetNZ();
    res = jm.GetResolution();

    locs = n_xy * n_xy * n_z;
    size_t sz = locs * locs;

    if (conditionals == nullptr) {
      cudaMalloc(&conditionals, sz * sizeof(Conditional));
    }

    if (marginals == nullptr) {
      cudaMalloc(&marginals, locs * sizeof(Marginal));
    }

    std::vector<Conditional> h_cond(sz);
    std::vector<Marginal> h_marg(locs);

    // Get all locations
    int min_ij = - (jm.GetNXY() / 2);
    int max_ij = min_ij + jm.GetNXY();

    int min_k = - (jm.GetNZ() / 2);
    int max_k = min_k + jm.GetNZ();

    //printf("\tFilling look up tables...\n");
    for (int i1=min_ij; i1 < max_ij; i1++) {
      for (int j1=min_ij; j1 < max_ij; j1++) {
        for (int k1=min_k; k1 < max_k; k1++) {
          rt::Location loc_eval(i1, j1, k1);

          // Marginal
          int idx = GetIndex(loc_eval);
          int c_t = jm.GetCount(loc_eval, true);
          int c_f = jm.GetCount(loc_eval, false);
          h_marg[idx].SetCount(true, c_t);
          h_marg[idx].SetCount(false, c_f);

          // Conditionals
          for (int i2=min_ij; i2 < max_ij; i2++) {
            for (int j2=min_ij; j2 < max_ij; j2++) {
              for (int k2=min_k; k2 < max_k; k2++) {
                rt::Location loc_given(i2, j2, k2);

                int idx = GetIndex(loc_eval, loc_given);

                for (int i_eval = 0; i_eval<2; i_eval++) {
                  bool eval = i_eval == 0;
                  for (int i_given = 0; i_given<2; i_given++) {
                    bool given = i_given == 0;

                    int my_count = jm.GetCount(loc_eval, eval, loc_given, given);
                    h_cond[idx].SetCount(eval, given, my_count);
                  }
                }
              }
            }
          }
        }
      }
    }

    // Send to device
    //printf("\tCopying to device...\n");
    cudaMemcpy(conditionals, h_cond.data(), sz*sizeof(Conditional), cudaMemcpyHostToDevice);
    cudaMemcpy(marginals, h_marg.data(), locs*sizeof(Marginal), cudaMemcpyHostToDevice);
  }

  void Cleanup() {
    if (conditionals != nullptr) {
      cudaFree(conditionals);
      conditionals = nullptr;
    }

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

  __host__ __device__ float GetLogP(const rt::Location &loc_eval, bool occ_eval, const rt::Location &loc_given, bool occ_given) const {
    int idx = GetIndex(loc_eval, loc_given);
    if (idx < 0) {
      return 0.0;
    }

    return conditionals[idx].GetLogP(occ_eval, occ_given);
  }

  __host__ __device__ float GetMutualInformation(const rt::Location &loc_eval, const rt::Location &loc_given) const {
    int idx = GetIndex(loc_eval, loc_given);
    if (idx < 0) {
      return 0.0;
    }

    return conditionals[idx].GetMutualInformation();
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

  __host__ __device__ int GetIndex(const rt::Location &loc_eval, const rt::Location &loc_given) const {
    int idx_eval = GetIndex(loc_eval);
    if (idx_eval < 0) {
      return -1;
    }

    int idx_given = GetIndex(loc_given);
    if (idx_given < 0) {
      return -1;
    }

    return idx_eval * locs + idx_given;
  }
};

} // namespace detector
} // namespace library
