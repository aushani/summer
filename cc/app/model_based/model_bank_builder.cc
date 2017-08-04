#include "model_bank_builder.h"

#include <boost/archive/binary_oarchive.hpp>

#include <iostream>
#include <fstream>

#include "library/timer/timer.h"

ModelBankBuilder::ModelBankBuilder() :
  data_manager_(32, false, false), model_bank_() {

  double area = (20*20 - 10*10)*360;
  double n_shapes = 1.5;

  double prior = kPosRes_*kPosRes_*kAngleRes_*n_shapes / area;

  model_bank_.AddRayModel("BOX", 5.0, prior);
  model_bank_.AddRayModel("STAR", 5.0, prior);
  model_bank_.AddRayModel("NOOBJ", 5.0, 2*M_PI, 0.1, 1-2*prior);

  for (const auto &cn : model_bank_.GetClasses()) {
    if (cn == "NOOBJ") {
      threads_.emplace_back(&ModelBankBuilder::RunNegativeMiningTrials, this);
    } else {
      threads_.emplace_back(&ModelBankBuilder::RunClassTrials, this, cn);
    }
  }
}

ModelBankBuilder::~ModelBankBuilder() {
  Finish();

  for (auto &t : threads_) {
    t.join();
  }
}

void ModelBankBuilder::RunClassTrials(const std::string &cn) {
  library::timer::Timer t;

  // Sampling jitter
  std::uniform_real_distribution<double> jitter_pos(-kPosRes_/2, kPosRes_/2);
  std::uniform_real_distribution<double> jitter_angle(-kAngleRes_/2, kAngleRes_/2);

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine re(seed);

  while (!done_) {
    // Get sim data
    sw::Data *data = data_manager_.GetData();
    sw::SimWorld *sim = data->GetSim();
    std::vector<ge::Point> *hits = data->GetHits();

    // Convert hits to observations
    std::vector<Observation> observations;
    for (const auto& h : *hits) {
      observations.emplace_back(Eigen::Vector2d(h.x, h.y));
    }

    auto shapes = sim->GetShapes();

    for (auto &shape : shapes) {
      if (shape.GetName() != cn) {
        continue;
      }

      for (int i=0; i<kEntriesPerObj_; i++) {
        double dx = jitter_pos(re);
        double dy = jitter_pos(re);
        double dt = jitter_angle(re);

        ObjectState os(shape.GetCenter()(0) + dx, shape.GetCenter()(1) + dy, shape.GetAngle() + dt, shape.GetName());

        // Each thread has it's own object class, so lock_shared is OK
        mb_mutex_.lock_shared();
        model_bank_.MarkObservations(os, observations);
        samples_per_class_[cn]++;
        mb_mutex_.unlock_shared();
      }

    }

    delete data;
  }

  printf("\tFinished training for %s for %5.3f sec\n", cn.c_str(), t.GetMs()/1e3);
}

void ModelBankBuilder::RunNegativeMiningTrials() {
  library::timer::Timer t;

  double pr2 = kPosRes_ * kPosRes_;

  // Sampling positions
  double lower_bound = -20.0;
  double upper_bound = 20.0;
  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine re(seed);

  while (!done_) {
    // Get sim data
    sw::Data *data = data_manager_.GetData();
    sw::SimWorld *sim = data->GetSim();
    std::vector<ge::Point> *hits = data->GetHits();

    // Convert hits to observations
    std::vector<Observation> observations;
    for (const auto& h : *hits) {
      observations.emplace_back(Eigen::Vector2d(h.x, h.y));
    }

    auto shapes = sim->GetShapes();

    for (int neg=0; neg<kEntriesPerObj_; neg++) {
      double x = unif(re);
      double y = unif(re);
      double object_angle = 0;

      ObjectState os(x, y, object_angle, "NOOBJ");

      bool too_close = false;

      for (auto &shape : shapes) {
        const auto &center = shape.GetCenter();

        if ((os.GetPos() - center).squaredNorm() < pr2) {
          too_close = true;
          break;
        }
      }

      if (!too_close) {
        // Each thread has it's own object class, so lock_shared is OK
        mb_mutex_.lock_shared();
        model_bank_.MarkObservations(os, observations);
        samples_per_class_["NOOBJ"]++;
        mb_mutex_.unlock_shared();
      }
    }
    delete data;
  }

  printf("\tFinished negative mining for %5.3f sec\n", t.GetMs()/1e3);
}

void ModelBankBuilder::SaveModelBank(const std::string &fn) {
  library::timer::Timer t;

  // Make a copy of the model bank we have so far
  mb_mutex_.lock();
  t.Start();
  ModelBank mb_cp(model_bank_);
  mb_mutex_.unlock();
  printf("Took %5.3f sec to make a copy of the model bank\n", t.GetMs());

  // Save model bank copy
  t.Start();
  std::ofstream ofs(fn);
  boost::archive::binary_oarchive oa(ofs);
  oa << mb_cp;
  printf("\tTook %5.3f sec to save\n", t.GetMs()/1e3);

  // Print some stats
  for (auto it = samples_per_class_.begin(); it != samples_per_class_.end(); it++) {
    printf("\t\tClass %s has %d samples\n", it->first.c_str(), it->second);
  }

  mb_cp.PrintStats();
}

void ModelBankBuilder::Finish() {
  done_ = true;
}
