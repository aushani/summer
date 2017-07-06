#pragma once

#include <thread>
#include <mutex>
#include <deque>

#include "library/hilbert_map/hilbert_map.h"
#include "library/sim_world/sim_world.h"

namespace ge = library::geometry;
namespace sw = library::sim_world;

namespace app {
namespace kernel_learning {

class Data {
 public:
  Data();
  ~Data();

  sw::SimWorld* GetSim();
  std::vector<ge::Point>* GetPoints();
  std::vector<float>* GetLabels();
  std::vector<ge::Point>* GetHits();
  std::vector<ge::Point>* GetOrigins();

  std::vector<ge::Point>* GetOccludedPoints();
  std::vector<float>* GetOccludedLabels();

 private:
  sw::SimWorld *sim_ = NULL;
  std::vector<ge::Point> *points_ = NULL;
  std::vector<float> *labels_ = NULL;
  std::vector<ge::Point> *hits_ = NULL;
  std::vector<ge::Point> *origins_ = NULL;

  std::vector<ge::Point> *occluded_points_ = NULL;
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

}
}
