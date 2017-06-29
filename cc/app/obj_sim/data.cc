#include "app/obj_sim/data.h"

#include <random>
#include <chrono>

Data::Data() :
 sim_(new SimWorld()),
 points_(new std::vector<hm::Point>()),
 labels_(new std::vector<float>()),
 hits_(new std::vector<hm::Point>()),
 origins_(new std::vector<hm::Point>()) {

  int trials = 10000;
  sim_->GenerateSamples(trials, points_, labels_);

  //sim_->GenerateGrid(10.0, points_, labels_);
  sim_->GenerateSimData(hits_, origins_);
}

Data::~Data() {
  delete sim_;
  delete points_;
  delete labels_;
  delete hits_;
  delete origins_;
}

SimWorld* Data::GetSim() {
  return sim_;
}

std::vector<hm::Point>* Data::GetPoints() {
  return points_;
}

std::vector<float>* Data::GetLabels() {
  return labels_;
}

std::vector<hm::Point>* Data::GetHits() {
  return hits_;
}

std::vector<hm::Point>* Data::GetOrigins() {
  return origins_;
}

DataManager::DataManager(int threads) {
  for (int i=0; i<threads; i++) {
    threads_.push_back(std::thread(&DataManager::GenerateData, this));
  }
}

DataManager::~DataManager() {
  Finish();

  for (size_t i=0; i<threads_.size(); i++) {
    printf("joining thread %ld\n", i);
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
      printf("\tHave %ld data left\n", data_.size());
    }
    mutex_.unlock();
  }

  return res;
}

void DataManager::GenerateData() {
  while (!done_) {
    Data *d = new Data();

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
