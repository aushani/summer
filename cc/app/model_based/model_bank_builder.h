#pragma once

#include <thread>

#include <boost/thread/shared_mutex.hpp>
//#include <boost/noncopyable.hpp>

#include "library/timer/timer.h"
#include "library/sim_world/sim_world.h"
#include "library/sim_world/data.h"

#include "model_bank.h"

namespace sw = library::sim_world;

class ModelBankBuilder {
 public:
  ModelBankBuilder();
  ~ModelBankBuilder();

  void Finish();

  void SaveModelBank(const std::string &fn);

 private:
  const double kPosRes_ = 0.3; // 30 cm
  const double kAngleRes_ = 15.0 * M_PI/180.0; // 15 deg

  const int kEntriesPerObj_ = 100;

  std::map<std::string, int> samples_per_class_;

  sw::DataManager data_manager_;

  ModelBank model_bank_;

  std::vector<std::thread> threads_;
  boost::shared_mutex mb_mutex_;
  volatile bool done_ = false;

  void RunClassTrials(const std::string &cn);
  void RunNegativeMiningTrials();
};
