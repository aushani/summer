#include "library/detector/detector.h"

#include <boost/assert.hpp>

#include "library/timer/timer.h"
#include "library/ray_tracing/device_dense_occ_grid.h"

namespace library {
namespace detector {

struct DeviceModel {
  float *log_p_occu = nullptr;
  float *log_p_free = nullptr;

  int n_xy = 0;
  int n_z = 0;

  float res = 0;

  DeviceModel(const clt::MarginalModel &mm) {
    n_xy = mm.GetNXY();
    n_z = mm.GetNZ();
    res = mm.GetResolution();

    size_t sz = n_xy * n_xy * n_z;

    cudaMalloc(&log_p_occu, sz * sizeof(float));
    cudaMalloc(&log_p_free, sz * sizeof(float));

    std::vector<float> h_lpo(sz, 0.0);
    std::vector<float> h_lpf(sz, 0.0);

    // Get all locations
    int min_ij = - (mm.GetNXY() / 2);
    int max_ij = min_ij + mm.GetNXY();

    int min_k = - (mm.GetNZ() / 2);
    int max_k = min_k + mm.GetNZ();

    for (int i=min_ij; i < max_ij; i++) {
      for (int j=min_ij; j < max_ij; j++) {
        for (int k=min_k; k < max_k; k++) {
          rt::Location loc(i, j, k);

          int c_t = mm.GetCount(loc, true);
          int c_f = mm.GetCount(loc, false);
          float denom = c_t + c_f;

          h_lpo[GetIndex(loc)] = log(c_t / denom);
          h_lpf[GetIndex(loc)] = log(c_f / denom);
        }
      }
    }

    // Send to device
    cudaMemcpy(log_p_occu, h_lpo.data(), sz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(log_p_free, h_lpf.data(), sz*sizeof(float), cudaMemcpyHostToDevice);
  }

  void Cleanup() {
    cudaFree(log_p_occu);
    cudaFree(log_p_free);
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

    return (x*n_xy + y)* n_z + z;
  }

  __host__ __device__ float GetLogP(const rt::Location &loc, bool occ) const {
    int idx = GetIndex(loc);
    if (idx < 0) {
      return 0.0;
    }

    return occ ? log_p_occu[idx]:log_p_free[idx];
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

  void AddModel(const clt::MarginalModel &mm, float range_x, float range_y, float log_prior) {
    models.emplace_back(mm);
    scores.emplace_back(mm.GetResolution(), range_x, range_y, log_prior);
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
 device_data_(new DeviceData()) {

}

Detector::~Detector() {
  device_data_->Cleanup();
}

void Detector::AddModel(const std::string &classname, const clt::MarginalModel &mm, float log_prior) {
  device_data_->AddModel(mm, range_x_, range_y_, log_prior);
  classnames_.push_back(classname);
}

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

  for (int i=min_ij; i < max_ij; i++) {
    for (int j=min_ij; j < max_ij; j++) {
      for (int k=min_k; k < max_k; k++) {
        rt::Location loc(i, j, k);

        // TODO transform
        rt::Location loc_global(i + os.x/model.res, j + os.y/model.res, k);

        if (!ddog.IsKnown(loc_global)) {
          continue;
        }

        bool occ = ddog.IsOccu(loc_global);
        log_p += model.GetLogP(loc, occ);
      }
    }
  }

  scores.d_scores[idx] = log_p;
}

void Detector::Run(const rt::DeviceOccGrid &dog) {
  library::timer::Timer t;

  t.Start();
  rt::DeviceDenseOccGrid ddog(dog, 50.0, 3.0);
  printf("Made Device Dense Occ Grid in %5.3f ms.\n", t.GetMs());

  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);

  for (int i=0; i<device_data_->NumModels(); i++) {
    printf("\tApplying Model %d\n", i);

    auto &model = device_data_->models[i];
    auto &scores = device_data_->scores[i];
    scores.Reset();

    int threads = kThreadsPerBlock_;
    int blocks = scores.Size() / threads + 1;

    Evaluate<<<blocks, threads>>>(ddog, model, scores);

    cudaError_t err = cudaDeviceSynchronize();
    BOOST_ASSERT(err == cudaSuccess);

    scores.CopyToHost();
  }

  ddog.Cleanup();
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
