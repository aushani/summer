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

  void AddModel(const ModelKey &key, const clt::JointModel &jm, const Dim &dim, float lp) {
    Data d(DeviceModel(jm), DeviceScores(dim, lp));
    data.insert({key, d});
  }

  void AddModel(const ModelKey &key, const clt::MarginalModel &mm, const Dim &dim, float lp) {
    Data d(DeviceModel(mm), DeviceScores(dim, lp));
    data.insert({key, d});
  }

  void UpdateModel(const ModelKey &key, const clt::JointModel &jm) {
    auto it = data.find(key);
    BOOST_ASSERT(it != data.end());

    Data &d = it->second;
    d.first.BuildModel(jm);
  }

  void UpdateModel(const ModelKey &key, const clt::MarginalModel &mm) {
    auto it = data.find(key);
    BOOST_ASSERT(it != data.end());

    Data &d = it->second;
    d.first.BuildModel(mm);
  }

  void UpdateModel(const ModelKey &key, const rt::DeviceOccGrid &dog) {
    auto it = data.find(key);
    BOOST_ASSERT(it != data.end());

    Data &d = it->second;
    d.first.UpdateModel(dog);
  }

  void LoadIntoJointModel(const ModelKey &key, clt::JointModel *jm) {
    auto it = data.find(key);
    BOOST_ASSERT(it != data.end());

    Data &d = it->second;
    d.first.LoadIntoJointModel(jm);
  }

  void LoadIntoMarginalModel(const ModelKey &key, clt::MarginalModel *mm) {
    auto it = data.find(key);
    BOOST_ASSERT(it != data.end());

    Data &d = it->second;
    d.first.LoadIntoMarginalModel(mm);
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

void Detector::AddModel(const std::string &classname, int angle_bin, const clt::JointModel &jm, float log_prior) {
  ModelKey key(classname, angle_bin);

  device_data_->AddModel(key, jm, dim_, log_prior);
  classnames_.push_back(classname);
}

void Detector::AddModel(const std::string &classname, int angle_bin, const clt::MarginalModel &mm, float log_prior) {
  ModelKey key(classname, angle_bin);

  device_data_->AddModel(key, mm, dim_, log_prior);
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


void Detector::UpdateModel(const std::string &classname, int angle_bin, const clt::JointModel &jm) {
  ModelKey key(classname, angle_bin);
  device_data_->UpdateModel(key, jm);
}

void Detector::UpdateModel(const std::string &classname, int angle_bin, const clt::MarginalModel &mm) {
  ModelKey key(classname, angle_bin);
  device_data_->UpdateModel(key, mm);
}

void Detector::UpdateModel(const std::string &classname,  int angle_bin, const rt::DeviceOccGrid &dog) {
  ModelKey key(classname, angle_bin);
  device_data_->UpdateModel(key, dog);
}

void Detector::LoadIntoJointModel(const std::string &classname, int angle_bin, clt::JointModel *jm) const {
  ModelKey key(classname, angle_bin);
  device_data_->LoadIntoJointModel(key, jm);
}

void Detector::LoadIntoMarginalModel(const std::string &classname, int angle_bin, clt::MarginalModel *mm) const {
  ModelKey key(classname, angle_bin);
  device_data_->LoadIntoMarginalModel(key, mm);
}


struct LocsBuffer {
  static constexpr int kNumLocs = 10;

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

__global__ void Evaluate(const rt::DeviceDenseOccGrid ddog, const DeviceModel model, const DeviceModel bg_model, const DeviceScores scores) {
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

        if (model.conditionals != nullptr) {
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
        } else {
          // Marginal only
          bool occ = ddog.IsOccu(loc_global);
          float update = model.GetLogP(loc, occ);
          log_p += update;
        }
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
  //printf("Took %5.3f ms to build device occ grid\n", t.GetMs());

  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);

  t.Start();
  rt::DeviceDenseOccGrid ddog(*dog, 100.0, 4.0);
  //printf("Made Device Dense Occ Grid in %5.3f ms\n", t.GetMs());

  err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);

  for (auto &kv : device_data_->data) {
    auto &d = kv.second;
    //printf("\tApplying Model %d\n", i);

    DeviceModel &model = d.first;
    DeviceScores &scores = d.second;
    scores.Reset();

    const DeviceModel &bg_model = device_data_->GetModel(ModelKey("Background", 0));

    int threads = kThreadsPerBlock_;
    int blocks = scores.dim.Size() / threads + 1;

    Evaluate<<<blocks, threads>>>(ddog, model, bg_model, scores);

    err = cudaDeviceSynchronize();
    BOOST_ASSERT(err == cudaSuccess);

    scores.CopyToHost();
  }

  ddog.Cleanup();
  dog->Cleanup();
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
