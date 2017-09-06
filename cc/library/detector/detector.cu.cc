#include "library/detector/detector.h"

#include <boost/assert.hpp>

#include "library/timer/timer.h"
#include "library/ray_tracing/device_dense_occ_grid.h"

namespace library {
namespace detector {

struct Marginal {
  float log_ps[2] = {0.0, 0.0};

  __host__ __device__ float Get(bool occ) const {
    return log_ps[GetIndex(occ)];
  }

  void Set(bool occ, float log_p) {
    log_ps[GetIndex(occ)] = log_p;
  }

  __host__ __device__ int GetIndex(bool occ) const {
    return occ ? 0:1;
  }
};

struct Conditional {
  float log_ps[4] = {0.0, 0.0, 0.0, 0.0};

  __host__ __device__ float Get(bool occ_eval, bool occ_given) const {
    return log_ps[GetIndex(occ_eval, occ_given)];
  }

  void Set(bool occ_eval, bool occ_given, float log_p) {
    log_ps[GetIndex(occ_eval, occ_given)] = log_p;
  }

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
};

struct DeviceModel {
 public:
  Conditional *conditionals = nullptr;
  Marginal *marginals = nullptr;
  float *mis = nullptr;

  int n_xy = 0;
  int n_z = 0;
  int locs = 0;

  float res = 0;

  DeviceModel(const clt::JointModel &jm) {
    BuildModel(jm);
  }

  void BuildModel(const clt::JointModel &jm) {
    // Cleanup
    Cleanup();

    n_xy = jm.GetNXY();
    n_z = jm.GetNZ();
    res = jm.GetResolution();

    locs = n_xy * n_xy * n_z;
    size_t sz = locs * locs;

    cudaMalloc(&conditionals, sz * sizeof(Conditional));
    cudaMalloc(&marginals, locs * sizeof(Marginal));
    cudaMalloc(&mis, sz * sizeof(float));

    std::vector<Conditional> h_cond(sz);
    std::vector<Marginal> h_marg(locs);
    std::vector<float> h_mis(sz, 0.0);

    // Get all locations
    int min_ij = - (jm.GetNXY() / 2);
    int max_ij = min_ij + jm.GetNXY();

    int min_k = - (jm.GetNZ() / 2);
    int max_k = min_k + jm.GetNZ();

    printf("\tFilling look up tables...\n");
    for (int i1=min_ij; i1 < max_ij; i1++) {
      for (int j1=min_ij; j1 < max_ij; j1++) {
        for (int k1=min_k; k1 < max_k; k1++) {
          rt::Location loc_eval(i1, j1, k1);

          // Marginal
          int c_t = jm.GetCount(loc_eval, true);
          int c_f = jm.GetCount(loc_eval, false);
          float denom = c_t + c_f;

          int idx = GetIndex(loc_eval);
          h_marg[idx].Set(true, std::log(c_t / denom));
          h_marg[idx].Set(false, std::log(c_f / denom));

          // Conditionals
          for (int i2=min_ij; i2 < max_ij; i2++) {
            for (int j2=min_ij; j2 < max_ij; j2++) {
              for (int k2=min_k; k2 < max_k; k2++) {
                rt::Location loc_given(i2, j2, k2);

                int idx = GetIndex(loc_eval, loc_given);
                float mi = jm.GetMutualInformation(loc_eval, loc_given);

                // Check for num observations
                // TODO magic number
                if (jm.GetNumObservations(loc_eval, loc_given) < 100) {
                    mi = 0.0;
                }

                h_mis[idx] = mi;

                for (int i_eval = 0; i_eval<2; i_eval++) {
                  bool eval = i_eval == 0;
                  for (int i_given = 0; i_given<2; i_given++) {
                    bool given = i_given == 0;

                    int my_count = jm.GetCount(loc_eval, eval, loc_given, given);
                    int other_count = jm.GetCount(loc_eval, !eval, loc_given, given);

                    float denom = my_count + other_count;
                    float log_p = my_count / denom;
                    h_cond[idx].Set(eval, given, log_p);
                  }
                }
              }
            }
          }
        }
      }
    }

    // Send to device
    printf("\tCopying to device...\n");
    cudaMemcpy(conditionals, h_cond.data(), sz*sizeof(Conditional), cudaMemcpyHostToDevice);
    cudaMemcpy(marginals, h_marg.data(), locs*sizeof(Marginal), cudaMemcpyHostToDevice);
    cudaMemcpy(mis, h_mis.data(), sz*sizeof(float), cudaMemcpyHostToDevice);
  }

  void Cleanup() {
    if (conditionals != nullptr) {
      cudaFree(conditionals);
    }

    if (marginals != nullptr) {
      cudaFree(marginals);
    }

    if (mis != nullptr) {
      cudaFree(mis);
    }
  }

  __host__ __device__ float GetLogP(const rt::Location &loc, bool occ) const {
    int idx = GetIndex(loc);
    if (idx < 0) {
      return 0.0;
    }

    return marginals[idx].Get(occ);
  }

  __host__ __device__ float GetLogP(const rt::Location &loc_eval, bool occ_eval, const rt::Location &loc_given, bool occ_given) const {
    int idx = GetIndex(loc_eval, loc_given);
    if (idx < 0) {
      return 0.0;
    }

    return conditionals[idx].Get(occ_eval, occ_given);
  }

  __host__ __device__ float GetMutualInformation(const rt::Location &loc_eval, const rt::Location &loc_given) const {
    int idx = GetIndex(loc_eval, loc_given);
    if (idx < 0) {
      return 0.0;
    }

    return mis[idx];
  }

 private:
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

struct DeviceData {
  std::vector<DeviceModel> models;
  std::vector<DeviceScores> scores;

  DeviceData() {
  }

  void Cleanup() {
    for (auto &model : models) {
      model.Cleanup();
    }

    for (auto &s : scores) {
      s.Cleanup();
    }
  }

  void AddModel(const clt::JointModel &jm, float range_x, float range_y, float log_prior) {
    printf("Adding model...\n");
    models.emplace_back(jm);

    printf("Adding scores...\n");
    scores.emplace_back(jm.GetResolution(), range_x, range_y, log_prior);
  }

  void UpdateModel(int i, const clt::JointModel &jm) {
    models[i].BuildModel(jm);
  }

  size_t NumModels() const {
    return models.size();
  }
};

Detector::Detector(float res, float range_x, float range_y) :
 range_x_(range_x), range_y_(range_y),
 n_x_(2 * std::ceil(range_x / res) + 1),
 n_y_(2 * std::ceil(range_y / res) + 1),
 res_(res),
 device_data_(new DeviceData()),
 og_builder_(200000, res, sqrt(range_x*range_x + range_y * range_y)) {

}

Detector::~Detector() {
  device_data_->Cleanup();
}

void Detector::AddModel(const std::string &classname, const clt::JointModel &jm, float log_prior) {
  device_data_->AddModel(jm, range_x_, range_y_, log_prior);
  classnames_.push_back(classname);
}

void Detector::UpdateModel(const std::string &classname, const clt::JointModel &jm) {
  int idx = -1;
  for (size_t i=0; i < classnames_.size(); i++) {
    if (classnames_[i] == classname) {
      idx = i;
      break;
    }
  }

  BOOST_ASSERT(idx >= 0);

  device_data_->UpdateModel(idx, jm);
}

struct LocsBuffer {
  static constexpr int kNumLocs = 2197;

  rt::Location locs[kNumLocs];
  int idx_at = 0;
  bool all_valid = false;

  __device__ void Mark(const rt::Location &loc) {
    locs[idx_at] = loc;

    idx_at++;
    if (idx_at >= kNumLocs) {
      idx_at = 0;
      all_valid = true;
    }
  }

  __device__ int NumValid() const {
    return all_valid ? kNumLocs:idx_at;
  }

  __device__ const rt::Location& Get(int i) const {
    return locs[i];
  }
};

__global__ void Evaluate(const rt::DeviceDenseOccGrid ddog, const DeviceModel model, const DeviceScores scores) {
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int threads = blockDim.x;

  const int idx = tidx + bidx * threads;
  if (!scores.InRange(idx)) {
    return;
  }

  ObjectState os = scores.GetState(idx);

  int min_ij = -(model.n_xy/2);
  int max_ij = min_ij + model.n_xy;

  int min_k = -(model.n_z/2);
  int max_k = min_k + model.n_z;

  float log_p = 0;
  float total_mi = 0;

  LocsBuffer locs_seen;

  for (int i=min_ij; i < max_ij; i++) {
    for (int j=min_ij; j < max_ij; j++) {
      for (int k=min_k; k < max_k; k++) {
        rt::Location loc(i, j, k);

        // TODO transform
        rt::Location loc_global(loc.i + os.x/model.res, loc.j + os.y/model.res, loc.k);

        if (!ddog.IsKnown(loc_global)) {
          continue;
        }

        // Search through locs for best dependency
        rt::Location best_loc;
        float best_mi = 0;
        for (int i=0; i<locs_seen.NumValid(); i++) {
          const auto &loc_prev = locs_seen.Get(i);

          // Get mi
          float mi = model.GetMutualInformation(loc, loc_prev);

          if (mi > best_mi) {
            best_mi = mi;
            best_loc = loc_prev;
          }
        }

        bool occ = ddog.IsOccu(loc_global);

        if (best_mi > 0) {
          // TODO transform
          rt::Location loc_prev_global(best_loc.i + os.x/model.res, best_loc.j + os.y/model.res, best_loc.k);
          bool occ_prev = ddog.IsOccu(loc_prev_global);

          log_p += model.GetLogP(loc, occ, best_loc, occ_prev);
          total_mi += best_mi;
        } else {
          log_p += model.GetLogP(loc, occ);
        }

        locs_seen.Mark(loc);
      }
    }
  }

  scores.d_scores[idx] = log_p;

  //printf("total mi: %5.3f\n", total_mi);
}

void Detector::Run(const std::vector<Eigen::Vector3d> &hits) {
  library::timer::Timer t;

  t.Start();
  auto dog = og_builder_.GenerateOccGridDevice(hits);
  printf("Took %5.3f ms to build device occ grid\n", t.GetMs());

  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);

  t.Start();
  rt::DeviceDenseOccGrid ddog(*dog, 50.0, 2.0);
  printf("Made Device Dense Occ Grid in %5.3f ms\n", t.GetMs());

  err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);

  for (int i=0; i<device_data_->NumModels(); i++) {
    printf("\tApplying Model %d\n", i);

    auto &model = device_data_->models[i];
    auto &scores = device_data_->scores[i];
    scores.Reset();

    int threads = kThreadsPerBlock_;
    int blocks = scores.Size() / threads + 1;

    Evaluate<<<blocks, threads>>>(ddog, model, scores);

    err = cudaDeviceSynchronize();
    BOOST_ASSERT(err == cudaSuccess);

    scores.CopyToHost();
  }

  ddog.Cleanup();
  dog->Cleanup();
}

const DeviceScores& Detector::GetScores(const std::string &classname) const {
  int idx = -1;
  for (size_t i=0; i < classnames_.size(); i++) {
    if (classnames_[i] == classname) {
      idx = i;
      break;
    }
  }

  BOOST_ASSERT(idx >= 0);

  return device_data_->scores[idx];
}

float Detector::GetScore(const std::string &classname, const ObjectState &os) const {
  auto &scores = GetScores(classname);
  return scores.GetScore(os);
}

float Detector::GetProb(const std::string &classname, const ObjectState &os) const {
  auto &my_scores = GetScores(classname);
  if (!my_scores.InRange(os)) {
    return 0.0;
  }

  float my_score = GetScore(classname, os);
  float max_score = my_score;

  for (const auto &class_scores : device_data_->scores) {
    float score = class_scores.GetScore(os);

    if (score > max_score) {
      max_score = score;
    }
  }

  float sum = 0;

  for (const auto &class_scores : device_data_->scores) {
    float score = class_scores.GetScore(os);
    sum += exp(score - max_score);
  }

  float prob = exp(my_score - max_score) / sum;
  return prob;
}

float Detector::GetLogOdds(const std::string &classname, const ObjectState &os) const {
  double prob = GetProb(classname, os);
  double lo = -std::log(1/prob - 1);

  return lo;
}

float Detector::GetRangeX() const {
  return range_x_;
}

float Detector::GetRangeY() const {
  return range_y_;
}

float Detector::GetResolution() const {
  return res_;
}

} // namespace ray_tracing
} // namespace library
