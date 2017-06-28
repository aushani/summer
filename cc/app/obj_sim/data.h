#pragma once

#include <thread>
#include <mutex>
#include <deque>

#include "library/hilbert_map/hilbert_map.h"

#include "app/obj_sim/sim_world.h"

namespace hm = library::hilbert_map;

class Data {
 public:
  Data();
  ~Data();

  SimWorld* GetSim();
  std::vector<hm::Point>* GetPoints();
  std::vector<float>* GetLabels();

 private:
  SimWorld *sim_ = NULL;
  std::vector<hm::Point> *points_ = NULL;
  std::vector<float> *labels_ = NULL;
};

class DataManager {
 public:
  DataManager(int threads);
  ~DataManager();

  Data* GetData();

 private:
  const size_t kMaxData = 100;

  std::vector<std::thread> threads_;
  std::deque<Data*> data_;
  std::mutex mutex_;

  volatile bool done_ = false;

  void GenerateData();
};
