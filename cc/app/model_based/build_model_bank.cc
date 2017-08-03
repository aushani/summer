#include <iostream>
#include <fstream>

#include <boost/archive/binary_oarchive.hpp>

#include "library/timer/timer.h"
#include "library/sim_world/sim_world.h"
#include "library/sim_world/data.h"

#include "model_bank.h"

namespace ge = library::geometry;
namespace sw = library::sim_world;

void SaveModelBank(const ModelBank &model_bank, const std::string &fn) {
  library::timer::Timer t;
  std::ofstream ofs(fn);
  boost::archive::binary_oarchive oa(ofs);
  oa << model_bank;
  printf("Took %5.3f sec to save\n", t.GetMs()/1e3);
}

void RunTrials(ModelBank *model_bank, sw::DataManager *data_manager, int n_trials) {
  // Resolution we car about for the model
  double pos_res = 0.3; // 30 cm
  double pr2 = pos_res * pos_res;

  double angle_res = 15.0 * M_PI/180.0; // 15 deg

  // Sampling positions
  double lower_bound = -20.0;
  double upper_bound = 20.0;
  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  //std::uniform_real_distribution<double> rand_angle(-M_PI, M_PI);
  std::uniform_real_distribution<double> rand_angle(-0.01, 0.01);

  std::uniform_real_distribution<double> jitter_pos(-pos_res/2, pos_res/2);
  std::uniform_real_distribution<double> jitter_angle(-angle_res/2, angle_res/2);

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine re(seed);

  int n_entries_per_obj = 100;

  for (int trial = 0; trial < n_trials; trial++) {
    // Get sim data
    sw::Data *data = data_manager->GetData();
    sw::SimWorld *sim = data->GetSim();
    std::vector<ge::Point> *hits = data->GetHits();

    // Convert hits to observations
    std::vector<Observation> observations;
    for (const auto& h : *hits) {
      observations.emplace_back(Eigen::Vector2d(h.x, h.y));
    }

    auto shapes = sim->GetShapes();

    for (auto &shape : shapes) {
      for (int i=0; i<n_entries_per_obj; i++) {
        double dx = jitter_pos(re);
        double dy = jitter_pos(re);
        double dt = jitter_angle(re);

        ObjectState os(shape.GetCenter()(0) + dx, shape.GetCenter()(1) + dy, shape.GetAngle() + dt, shape.GetName());
        model_bank->MarkObservations(os, observations);
      }
    }

    // Negative mining
    for (size_t neg=0; neg<n_entries_per_obj*shapes.size(); neg++) {
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
        model_bank->MarkObservations(os, observations);
      }
    }

    delete data;
  }
}

ModelBank LearnModelBank(int n_trials, const char *base) {
  sw::DataManager data_manager(16, false, false);

  library::timer::Timer t;

  double area = (20*20 - 10*10)*360;
  double pos_res = 0.3; // 30 cm
  double angle_res = 15.0; // 15 deg
  double n_shapes = 1.5;

  double prior = pos_res*pos_res*angle_res*n_shapes / area;

  ModelBank model_bank;
  model_bank.AddRayModel("BOX", 5.0, prior);
  model_bank.AddRayModel("STAR", 5.0, prior);
  model_bank.AddRayModel("NOOBJ", 5.0, 2*M_PI, 0.1, 1-2*prior);

  std::string bs(base);

  char fn[1000];
  int step = 0;

  while (true) {
    t.Start();
    RunTrials(&model_bank, &data_manager, n_trials);
    printf("Took %5.3f sec to run %d trials, saving step %d result \n", t.GetMs()/1e3, n_trials, step);

    printf("\n");
    model_bank.PrintStats();
    printf("\n");

    sprintf(fn, "%s_%06d", base, step++);
    SaveModelBank(model_bank, std::string(fn));
  }
  data_manager.Finish();

  return model_bank;
}

int main(int argc, char** argv) {
  printf("Building model bank...\n");

  if (argc != 3) {
    printf("Usage: build_model_bank n_trials filename\n");
    return 1;
  }

  int n_trials = strtol(argv[1], NULL, 10);
  ModelBank model_bank = LearnModelBank(n_trials, argv[2]);
}
