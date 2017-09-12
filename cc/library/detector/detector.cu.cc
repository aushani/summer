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

  void AddModel(const clt::MarginalModel &mm, float range_x, float range_y, float log_prior, int threads) {
    //printf("Adding model...\n");
    models.emplace_back(mm);

    //printf("Adding scores...\n");
    scores.emplace_back(mm.GetResolution(), range_x, range_y, log_prior, threads);
  }

  void UpdateModel(int i, const clt::MarginalModel &mm) {
    models[i].BuildModel(mm);
  }

  void UpdateModel(int i, const rt::DeviceOccGrid &dog) {
    models[i].UpdateModel(dog);
  }

  void LoadIntoMarginalModel(int i, clt::MarginalModel *mm) {
    models[i].LoadIntoMarginalModel(mm);
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

void Detector::AddModel(const std::string &classname, const clt::MarginalModel &mm, float log_prior) {
  device_data_->AddModel(mm, range_x_, range_y_, log_prior, kScoringThreads_);
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


void Detector::UpdateModel(const std::string &classname, const clt::MarginalModel &mm) {
  int idx = GetModelIndex(classname);
  BOOST_ASSERT(idx >= 0);

  device_data_->UpdateModel(idx, mm);
}

void Detector::UpdateModel(const std::string &classname, const rt::DeviceOccGrid &dog) {
  int idx = GetModelIndex(classname);
  BOOST_ASSERT(idx >= 0);

  device_data_->UpdateModel(idx, dog);
}

void Detector::LoadIntoMarginalModel(const std::string &classname, clt::MarginalModel *mm) const {
  int idx = GetModelIndex(classname);
  BOOST_ASSERT(idx >= 0);

  device_data_->LoadIntoMarginalModel(idx, mm);
}

__global__ void Evaluate(const rt::DeviceOccGrid dog, const DeviceModel model, const DeviceScores scores) {
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  const int scoring_thread_idx = bidx;
  if (scoring_thread_idx >= scores.scoring_threads) {
    return;
  }

  int min_ij = -(model.n_xy/2);
  int max_ij = min_ij + model.n_xy;

  int min_k = -(model.n_z/2);
  int max_k = min_k + model.n_z;

  for (int voxel_idx = scoring_thread_idx; voxel_idx < dog.size; voxel_idx+=scores.scoring_threads) {

    // Get the voxel we're looking at
    rt::Location &loc_global = dog.locs[voxel_idx];
    bool occ = dog.los[voxel_idx] > 0;

    // Check k
    if (loc_global.k < min_k || loc_global.k >= max_k) {
      continue;
    }

    // Update
    int di = tidx + min_ij;
    int dj = tidy + min_ij;

    if (di < max_ij && dj < max_ij) {
      int i = loc_global.i + di;
      int j = loc_global.j + dj;

      ObjectState os(i*model.res, j*model.res, 0);
      size_t idx = scores.GetThreadIndex(os, scoring_thread_idx);

      if (idx > 0) {
        rt::Location loc_model(-di, -dj, loc_global.k);
        float update = model.GetLogP(loc_model, occ);

        scores.d_scores_thread[idx] += update;
      }
    }

    // Sync
    __syncthreads();
  }
}

void Detector::Run(const std::vector<Eigen::Vector3d> &hits) {
  //library::timer::Timer t;

  //t.Start();
  auto dog = og_builder_.GenerateOccGridDevice(hits);
  //printf("\tTook %5.3f ms to build device occ grid with %d voxels\n", t.GetMs(), dog->size);

  cudaError_t err = cudaDeviceSynchronize();
  BOOST_ASSERT(err == cudaSuccess);

  for (int i=0; i<device_data_->NumModels(); i++) {
    auto &model = device_data_->models[i];
    auto &scores = device_data_->scores[i];
    scores.Reset();

    int blocks = kScoringThreads_;
    dim3 blockDim;
    blockDim.x = 32;
    blockDim.y = 32;
    BOOST_ASSERT(model.n_xy <= 32);

    //t.Start();
    Evaluate<<<blocks, blockDim>>>(*dog, model, scores);
    err = cudaDeviceSynchronize();
    //printf("\t\tTook %5.3f ms to apply %dx%dx%d model %s with %dx%d threads and %d blocks\n",
    //    t.GetMs(), model.n_xy, model.n_xy, model.n_z, classnames_[i].c_str(), blockDim.x, blockDim.y, blocks);

    BOOST_ASSERT(err == cudaSuccess);

    //t.Start();
    scores.Reduce();
    //printf("\t\tTook %5.3f ms to reduce scores\n", t.GetMs());

    scores.CopyToHost();
  }

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
