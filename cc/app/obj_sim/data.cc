#include "app/obj_sim/data.h"

#include <random>
#include <chrono>

Data::Data() :
 sim_(new SimWorld()),
 points_(new std::vector<hm::Point>()),
 labels_(new std::vector<float>()) {

  int trials = 5000;

  sim_->GenerateSamples(trials, points_, labels_);
}

Data::~Data() {
  delete sim_;
  delete points_;
  delete labels_;
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

DataManager::DataManager(int threads) {
  for (int i=0; i<threads; i++) {
    threads_.push_back(std::thread(&DataManager::GenerateData, this));
  }
}

DataManager::~DataManager() {
  done_ = true;

  for (size_t i=0; i<threads_.size(); i++) {
    printf("Killing thread %ld\n", i);
    threads_[i].join();
  }

  printf("delete deque\n");
  for (size_t i=0; i<data_.size(); i++) {
    delete data_[i];
  }

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
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
}
