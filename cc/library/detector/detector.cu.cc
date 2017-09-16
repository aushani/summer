#include "library/detector/detector.h"

#include <boost/assert.hpp>

#include "library/timer/timer.h"
#include "library/ray_tracing/device_dense_occ_grid.h"

#include "library/detector/device_model.cu.h"

namespace library {
namespace detector {

struct DeviceData {
  //std::vector<DeviceModel> models;
  //std::vector<DeviceScores> scores;
  typedef std::pair<DeviceModel, DeviceScores> Data;
  std::map<ModelKey, Data> data;

  DeviceData() {
  }

  void Cleanup() {
    for (auto &kv : data) {
      auto &d = kv.second;
      d.first.Cleanup();
      d.second.Cleanup();
    }
  }

  void AddModel(const ModelKey &key, const ft::FeatureModel &fm, const Dim &dim, float lp) {
    Data d(DeviceModel(fm), DeviceScores(dim, lp));
    data.insert({key, d});
  }

  const DeviceScores& GetScores(const ModelKey &key) const {
    auto it = data.find(key);
    BOOST_ASSERT(it != data.end());

    return it->second.second;
  }

  const DeviceModel& GetModel(const ModelKey &key) const {
    auto it = data.find(key);
    BOOST_ASSERT(it != data.end());

    return it->second.first;
  }
};

Detector::Detector(const Dim &d) :
 dim_(d),
 device_data_(new DeviceData()),
 og_builder_(200000, dim_.res, 100.0) {

}

Detector::~Detector() {
  device_data_->Cleanup();
}

void Detector::AddModel(const std::string &classname, int angle_bin, const ft::FeatureModel &fm, float log_prior) {
  ModelKey key(classname, angle_bin);

  device_data_->AddModel(key, fm, dim_, log_prior);
  classnames_.push_back(classname);
}

int Detector::GetModelIndex(const std::string &classname) const {
  int idx = -1;
  for (size_t i=0; i < classnames_.size(); i++) {
    if (classnames_[i] == classname) {
      idx = i;
      break;
    }
  }

  return idx;
}

__global__ void Evaluate(const rt::DeviceDenseFeatureOccGrid grid, const DeviceModel model, const DeviceScores scores, bool use_features) {
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int threads = blockDim.x;

  const int idx = tidx + bidx * threads;
  if (!scores.dim.InRange(idx)) {
    return;
  }

  ObjectState os = scores.dim.GetState(idx);

  int min_ij = -(model.n_xy/2);
  int max_ij = min_ij + model.n_xy;

  int min_k = -(model.n_z/2);
  int max_k = min_k + model.n_z;

  float log_p = 0;

  for (int i=min_ij; i < max_ij; i++) {
    for (int j=min_ij; j < max_ij; j++) {
      for (int k=min_k; k < max_k; k++) {
        rt::Location loc(i, j, k);

        rt::Location loc_global(loc.i + os.x/model.res, loc.j + os.y/model.res, loc.k);
        if (!grid.IsKnown(loc_global)) {
          continue;
        }

        bool occ = grid.IsOccu(loc_global);
        log_p += model.GetLogP(loc, occ);

        // Do we have a feature vector too?
        if (use_features && grid.HasFeature(loc_global)) {
          const rt::Feature &f = grid.GetFeature(loc_global);

          // evaluate!
          log_p += model.GetLogP(loc, f.theta, f.phi);
        }
      }
    }
  }

  scores.d_scores[idx] = log_p;
}

void Detector::Run(const std::vector<Eigen::Vector3d> &hits, const std::vector<float> &intensities) {
  library::timer::Timer t;

  t.Start();
  auto dfog = og_builder_.GenerateDeviceFeatureOccGrid(hits, intensities);
  printf("Took %5.3f ms to build device occ grid\n", t.GetMs());

  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);

  t.Start();
  rt::DeviceDenseFeatureOccGrid grid(dfog, 100.0, 4.0);
  printf("Made Device Dense Occ Grid in %5.3f ms\n", t.GetMs());

  err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);

  for (auto &kv : device_data_->data) {
    auto &d = kv.second;
    //printf("\tApplying Model %d\n", i);

    DeviceModel &model = d.first;
    DeviceScores &scores = d.second;
    scores.Reset();

    int threads = kThreadsPerBlock_;
    int blocks = scores.dim.Size() / threads + 1;

    Evaluate<<<blocks, threads>>>(grid, model, scores, use_features);

    err = cudaDeviceSynchronize();
    BOOST_ASSERT(err == cudaSuccess);

    scores.CopyToHost();
  }

  dfog.Cleanup();
  grid.Cleanup();
}

std::vector<Detection> Detector::GetDetections(double thresh) const {
  // This is something really stupid just to get started

  std::vector<Detection> detections;

  for (int i = 0; i < dim_.n_x; i++) {
    for (int j = 0; j < dim_.n_y; j++) {
      for (int angle_bin = 0; angle_bin < kAngleBins; angle_bin++) {
        ObjectState os = dim_.GetState(i, j);
        os.angle_bin = angle_bin;

        float lo_car = GetLogOdds("Car", os);
        float my_score = GetScore("Car", os);

        if (lo_car > thresh) {
          // Check for max
          bool max = true;
          for (int di = -2/dim_.res; di <= 2/dim_.res && max; di++) {
            for (int dj = -2/dim_.res; dj <= 2/dim_.res && max; dj++) {
              for (int b=0; b<kAngleBins && max; b++) {
                if (di == 0 && dj == 0 && b == angle_bin) {
                  continue;
                }

                ObjectState n_os = dim_.GetState(i + di, j + dj);
                n_os.angle_bin = b;
                float lo = GetLogOdds("Car", n_os);

                if (lo > lo_car) {
                  max = false;
                } else if (lo == lo_car) {
                  double score = GetScore("Car", n_os);
                  if (score > my_score) {
                    max = false;
                  }
                }
              }
            }
          }

          if (max) {
            float confidence = lo_car;
            detections.emplace_back("Car", os, confidence);
          }
        }
      }
    }
  }

  return detections;
}

const DeviceScores& Detector::GetScores(const std::string &classname, int angle_bin) const {
  ModelKey key(classname, angle_bin);
  return device_data_->GetScores(key);
}

float Detector::GetScore(const std::string &classname, const ObjectState &os) const {
  auto &scores = GetScores(classname, os.angle_bin);
  return scores.GetScore(os);
}

float Detector::GetProb(const std::string &classname, double x, double y) const {
  // TODO Hacky
  double p = 0;
  for (int angle_bin = 0; angle_bin < kAngleBins; angle_bin++) {
    ObjectState os(x, y, angle_bin);
    p += GetProb(classname, os);
  }

  return p;
}

float Detector::GetLogOdds(const std::string &classname, double x, double y) const {
  double prob = GetProb(classname, x, y);
  double lo = -std::log(1/prob - 1);

  if (lo > 40) {
    lo = 40;
  }

  if (lo < -40) {
    lo = -40;
  }

  return lo;
}

float Detector::GetProb(const std::string &classname, const ObjectState &os) const {
  auto &my_scores = GetScores(classname, os.angle_bin);
  if (!my_scores.dim.InRange(os)) {
    return 0.0;
  }

  float my_score = GetScore(classname, os);
  float max_score = my_score;

  // Normalize over classes and orientations
  for (const auto &kv : device_data_->data) {
    const auto &class_scores = kv.second.second;
    float score = class_scores.GetScore(os);

    if (score > max_score) {
      max_score = score;
    }
  }

  float sum = 0;

  for (const auto &kv : device_data_->data) {
    const auto &class_scores = kv.second.second;
    float score = class_scores.GetScore(os);
    sum += exp(score - max_score);
  }

  float prob = exp(my_score - max_score) / sum;
  return prob;
}

float Detector::GetLogOdds(const std::string &classname, const ObjectState &os) const {
  auto &my_scores = GetScores(classname, os.angle_bin);
  if (!my_scores.dim.InRange(os)) {
    return 0.0;
  }

  ModelKey my_key(classname, os.angle_bin);

  float my_score = GetScore(classname, os);
  float max_score = my_score;
  float max_other = -100000;

  // Normalize over classes and orientations
  for (const auto &kv : device_data_->data) {
    const auto &class_scores = kv.second.second;
    float score = class_scores.GetScore(os);

    if (score > max_score) {
      max_score = score;
    }

    if (score > max_other && kv.first != my_key) {
      max_other = score;
    }
  }

  float sum = 0;

  for (const auto &kv : device_data_->data) {
    const auto &class_scores = kv.second.second;
    float score = class_scores.GetScore(os);
    sum += exp(score - max_score);
  }
  float prob = exp(my_score - max_score) / sum;

  //double prob = GetProb(os);
  double lo = -std::log(1/prob - 1);

  if (lo > 10) {
    //printf("%5.3f vs %5.3f, %5.3f\n", lo, my_score, max_other);
    //lo = 10;
    lo = my_score - max_other; // approx for multiclass
  }

  if (lo < -10) {
    //printf("%5.3f vs %5.3f, %5.3f\n", lo, my_score, max_other);
    //lo = -10;
    lo = my_score - max_other; // approx for multiclass
  }

  return lo;
}

const Dim& Detector::GetDim() const {
  return dim_;
}

const std::vector<std::string>& Detector::GetClasses() const {
  return classnames_;
}

} // namespace ray_tracing
} // namespace library
