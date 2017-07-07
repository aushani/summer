#pragma once

#include <thread>
#include <mutex>
#include <deque>

#include "library/geometry/point.h"
#include "library/sim_world/sim_world.h"

namespace ge = library::geometry;

namespace library {
namespace sim_world {

class Data {
 public:
  Data(bool gen_random, bool gen_occluded);
  ~Data();

  SimWorld* GetSim();
  std::vector<ge::Point>* GetPoints();
  std::vector<float>* GetLabels();

  std::vector<ge::Point>* GetHits();
  std::vector<ge::Point>* GetOrigins();

  std::vector<ge::Point>* GetOccludedPoints();
  std::vector<float>* GetOccludedLabels();

  std::vector<ge::Point>* GetObsPoints();
  std::vector<float>* GetObsLabels();

 private:
  SimWorld *sim_ = NULL;
  std::vector<ge::Point> *points_ = NULL;
  std::vector<float> *labels_ = NULL;
  std::vector<ge::Point> *hits_ = NULL;
  std::vector<ge::Point> *origins_ = NULL;

  std::vector<ge::Point> *occluded_points_ = NULL;
  std::vector<float> *occluded_labels_ = NULL;

  std::vector<ge::Point> *obs_points_ = NULL;
  std::vector<float> *obs_labels_ = NULL;
};

class DataManager {
 public:
  DataManager(int threads, bool gen_random=true, bool gen_occluded=true);
  ~DataManager();

  Data* GetData();
  void Finish();

 private:
  const size_t kMaxData = 100;

  std::vector<std::thread> threads_;
  std::deque<Data*> data_;
  std::mutex mutex_;

  bool gen_random_;
  bool gen_occluded_;

  volatile bool done_ = false;

  void GenerateData();
};

}
}
