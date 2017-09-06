#include "library/detector/detector.h"

#include "library/ray_tracing/device_dense_occ_grid.h"

namespace library {
namespace detector {

struct DeviceModel {
  float *log_p_occu = nullptr;
  float *log_p_free = nullptr;

  int n_xy = 0;
  int n_z = 0;

  DeviceModel(const clt::MarginalModel &mm) {
    n_xy = mm.GetNXY();
    n_z = mm.GetNZ();

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

struct DeviceScores {
  float *scores = nullptr;
  int n_x = 0;
  int n_y = 0;
  float res = 0;

  DeviceScores(float r, double range_x, double range_y)  :
   n_x(2 * ceil(range_x / r) + 1),
   n_y(2 * ceil(range_y / r) + 1),
   res(r) {
    cudaMalloc(&scores, sizeof(float)*n_x*n_y);
    cudaMemset(scores, 0, sizeof(float)*n_x*n_y);
  }

  void Cleanup() {
    cudaFree(scores);
  }

  __host__ __device__ bool InRange(int idx) const {
    return idx < n_x * n_y;
  }


  __host__ __device__ ObjectState GetState(int idx) const {
    int ix = idx / n_y;
    int iy = idx % n_y;

    // int instead of size_t because could be negative
    int dix = ix - n_x/2;
    int diy = iy - n_y/2;

    float x = dix * res;
    float y = diy * res;

    return ObjectState(x, y, 0);
  }

  __host__ __device__ int GetIndex(const ObjectState &os) const {
    int ix = os.x / res + n_x / 2;
    int iy = os.y / res + n_y / 2;

    if (ix >= n_x || iy >= n_y) {
      return -1;
    }

    size_t idx = ix * n_y + iy;
    return idx;
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

    for (auto &score : scores) {
      score.Cleanup();
    }
  }

  void AddModel(const clt::MarginalModel &mm) {
    models.emplace_back(mm);
    scores.emplace_back(0.3, 50.0, 50.0);
  }
};

Detector::Detector(double res, double range_x, double range_y) :
 range_x_(range_x), range_y_(range_y),
 n_x_(2 * std::ceil(range_x / res) + 1),
 n_y_(2 * std::ceil(range_y / res) + 1),
 res_(res),
 device_data_(new DeviceData()) {

}

Detector::~Detector() {
  device_data_->Cleanup();
}

void Detector::AddModel(const std::string &classname, const clt::MarginalModel &mm) {
  device_data_->AddModel(mm);
  class_names_.push_back(classname);
}

__global__ void Evaluate(const rt::DeviceDenseOccGrid &ddog, const DeviceModel &model, const DeviceScores &scores) {
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int threads = blockDim.x;

  const int idx = tidx + bidx * threads;
  if (!scores.InRange(idx)) {
    return;
  }

  ObjectState os = scores.GetState(idx);
}

DetectionMap Detector::Run(const rt::DeviceOccGrid &dog) {
  rt::DeviceDenseOccGrid ddog(dog, 50.0, 3.0);

  int threads = 1024;
  int blocks = 1024;

  Evaluate<<<blocks, threads>>>(ddog, device_data_->models[0], device_data_->scores[0]);

  return DetectionMap();
}

} // namespace ray_tracing
} // namespace library
