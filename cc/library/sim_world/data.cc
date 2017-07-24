#include "library/sim_world/data.h"

#include <random>
#include <chrono>

namespace ge = library::geometry;

namespace library {
namespace sim_world {

Data::Data(bool gen_random, bool gen_occluded) :
 sim_(new SimWorld(3)),
 points_(new std::vector<ge::Point>()),
 labels_(new std::vector<float>()),
 hits_(new std::vector<ge::Point>()),
 origins_(new std::vector<ge::Point>()),
 occluded_points_(new std::vector<ge::Point>()),
 occluded_labels_(new std::vector<float>()),
 obs_points_(new std::vector<ge::Point>()),
 obs_labels_(new std::vector<float>()) {

  if (gen_random) {
    int trials = 10000;
    sim_->GenerateAllSamples(trials, points_, labels_);
  }

  if (gen_occluded) {
    int trials = 10000;
    sim_->GenerateOccludedSamples(trials, occluded_points_, occluded_labels_);
  }

  //sim_->GenerateGrid(10.0, points_, labels_);
  sim_->GenerateSimData(hits_, origins_);
  sim_->GenerateSimData(obs_points_, obs_labels_);
}

Data::~Data() {
  delete sim_;
  delete points_;
  delete labels_;
  delete hits_;
  delete origins_;

  delete occluded_points_;
  delete occluded_labels_;

  delete obs_points_;
  delete obs_labels_;
}

SimWorld* Data::GetSim() {
  return sim_;
}

std::vector<ge::Point>* Data::GetPoints() {
  return points_;
}

std::vector<float>* Data::GetLabels() {
  return labels_;
}

std::vector<ge::Point>* Data::GetHits() {
  return hits_;
}

std::vector<ge::Point>* Data::GetOrigins() {
  return origins_;
}

std::vector<ge::Point>* Data::GetOccludedPoints() {
  return occluded_points_;
}

std::vector<float>* Data::GetOccludedLabels() {
  return occluded_labels_;
}

std::vector<ge::Point>* Data::GetObsPoints() {
  return obs_points_;
}

std::vector<float>* Data::GetObsLabels() {
  return obs_labels_;
}

DataManager::DataManager(int threads, bool gen_random, bool gen_occluded) :
 gen_random_(gen_random), gen_occluded_(gen_occluded) {
  for (int i=0; i<threads; i++) {
    threads_.push_back(std::thread(&DataManager::GenerateData, this));
  }
}

DataManager::~DataManager() {
  Finish();

  for (size_t i=0; i<threads_.size(); i++) {
    //printf("joining thread %ld\n", i);
    threads_[i].join();
  }

  for (size_t i=0; i<data_.size(); i++) {
    delete data_[i];
  }
}

void DataManager::Finish() {
  done_ = true;
}

Data* DataManager::GetData() {
  Data *res = NULL;
  while(res == NULL) {
    mutex_.lock();
    if (!data_.empty()) {
      res = data_.front();
      data_.pop_front();
      //printf("\tHave %ld data left\n", data_.size());
    }
    mutex_.unlock();

    if (res == NULL) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      //printf("\tNeed to wait...\n");
    }
  }

  return res;
}

void DataManager::GenerateData() {
  while (!done_) {
    Data *d = new Data(gen_random_, gen_occluded_);

    mutex_.lock();
    size_t sz = data_.size();
    mutex_.unlock();

    if (sz < kMaxData) {
      mutex_.lock();
      data_.push_back(d);
      mutex_.unlock();
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
}

}
}
