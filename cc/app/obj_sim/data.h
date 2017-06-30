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
  std::vector<hm::Point>* GetHits();
  std::vector<hm::Point>* GetOrigins();

  std::vector<hm::Point>* GetOccludedPoints();
  std::vector<float>* GetOccludedLabels();

 private:
  SimWorld *sim_ = NULL;
  std::vector<hm::Point> *points_ = NULL;
  std::vector<float> *labels_ = NULL;
  std::vector<hm::Point> *hits_ = NULL;
  std::vector<hm::Point> *origins_ = NULL;

  std::vector<hm::Point> *occluded_points_ = NULL;
  std::vector<float> *occluded_labels_ = NULL;
};

class DataManager {
 public:
  DataManager(int threads);
  ~DataManager();

  Data* GetData();
  void Finish();

 private:
  const size_t kMaxData = 100;

  std::vector<std::thread> threads_;
  std::deque<Data*> data_;
  std::mutex mutex_;

  volatile bool done_ = false;

  void GenerateData();
};
