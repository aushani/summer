#include "library/detector/detector.h"

#include <boost/assert.hpp>

#include "library/timer/timer.h"
#include "library/ray_tracing/device_dense_occ_grid.h"

#include "library/detector/device_model.cu.h"

namespace library {
namespace detector {

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
    //printf("Adding model...\n");
    models.emplace_back(jm);

    //printf("Adding scores...\n");
    scores.emplace_back(jm.GetResolution(), range_x, range_y, log_prior);
  }

  void UpdateModel(int i, const clt::JointModel &jm) {
    models[i].BuildModel(jm);
  }

  void UpdateModel(int i, const rt::DeviceOccGrid &dog) {
    models[i].UpdateModel(dog);
  }

  void LoadIntoJointModel(int i, clt::JointModel *jm) {
    models[i].LoadIntoJointModel(jm);
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


void Detector::UpdateModel(const std::string &classname, const clt::JointModel &jm) {
  int idx = GetModelIndex(classname);
  BOOST_ASSERT(idx >= 0);

  device_data_->UpdateModel(idx, jm);
}

void Detector::UpdateModel(const std::string &classname, const rt::DeviceOccGrid &dog) {
  int idx = GetModelIndex(classname);
  BOOST_ASSERT(idx >= 0);

  device_data_->UpdateModel(idx, dog);
}

void Detector::LoadIntoJointModel(const std::string &classname, clt::JointModel *jm) const {
  int idx = GetModelIndex(classname);
  BOOST_ASSERT(idx >= 0);

  device_data_->LoadIntoJointModel(idx, jm);
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
  //printf("Took %5.3f ms to build device occ grid\n", t.GetMs());

  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);

  t.Start();
  rt::DeviceDenseOccGrid ddog(*dog, 50.0, 2.0);
  //printf("Made Device Dense Occ Grid in %5.3f ms\n", t.GetMs());

  err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);

  for (int i=0; i<device_data_->NumModels(); i++) {
    //printf("\tApplying Model %d\n", i);

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

  if (lo > 40) {
    lo = 40;
  }

  if (lo < -40) {
    lo = -40;
  }

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

float Detector::GetNX() const {
  return n_x_;
}

float Detector::GetNY() const {
  return n_y_;
}

const std::vector<std::string>& Detector::GetClasses() const {
  return classnames_;
}

} // namespace ray_tracing
} // namespace library
