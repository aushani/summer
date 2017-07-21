#include <iostream>
#include <fstream>

#include "library/timer/timer.h"
#include "library/sim_world/sim_world.h"
#include "library/sim_world/data.h"

#include "model_bank.h"

namespace ge = library::geometry;
namespace sw = library::sim_world;

void SaveModelBank(const ModelBank &model_bank, const std::string &fn) {
  std::ofstream ofs(fn);
  boost::archive::text_oarchive oa(ofs);
  oa << model_bank;
}

void RunTrials(ModelBank *model_bank, sw::DataManager *data_manager, int n_trials) {
  // Resolution we car about for the model
  double pos_res = 0.3;
  double angle_res = 10.0 * M_PI/180.0;

  // Sampling positions
  double lower_bound = -8.0;
  double upper_bound = 8.0;
  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  std::uniform_real_distribution<double> jitter(-pos_res/2.0, pos_res/2.0);
  //std::uniform_real_distribution<double> rand_angle(-M_PI, M_PI);
  std::uniform_real_distribution<double> rand_angle(-0.01, 0.01);
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine re(seed);

  int step = fmax(n_trials/100, 100);

  for (int trial = 0; trial < n_trials; trial++) {
    if (trial % step == 0) {
      //printf("\tTrial %d / %d\n", trial, n_trials);
    }

    // Get sim data
    sw::Data *data = data_manager->GetData();
    sw::SimWorld *sim = data->GetSim();
    std::vector<ge::Point> *hits = data->GetHits();

    auto shapes = sim->GetShapes();

    for (auto &shape : shapes) {
      Eigen::Vector2d x_sensor_object = shape.GetCenter();
      x_sensor_object(0) += jitter(re);
      x_sensor_object(1) += jitter(re);
      double object_angle = shape.GetAngle();

      for (size_t i=0; i<hits->size(); i++) {
        Eigen::Vector2d x_hit;
        x_hit(0) = hits->at(i).x;
        x_hit(1) = hits->at(i).y;

        model_bank->MarkObservation(shape.GetName(), x_sensor_object, object_angle, x_hit);
      }
    }

    // Negative mining
    for (int neg=0; neg<10; neg++) {
      Eigen::Vector2d x_sensor_noobj;
      x_sensor_noobj(0) = unif(re);
      x_sensor_noobj(1) = unif(re);

      double object_angle = rand_angle(re);

      bool too_close = false;

      for (auto &shape : shapes) {
        const auto &center = shape.GetCenter();

        if ((x_sensor_noobj - center).norm() < pos_res) {
          too_close = true;
          break;
        }
      }

      if (!too_close) {
        for (size_t i=0; i<hits->size(); i++) {
          Eigen::Vector2d x_hit;
          x_hit(0) = hits->at(i).x;
          x_hit(1) = hits->at(i).y;

          model_bank->MarkObservation("NOOBJ", x_sensor_noobj, object_angle, x_hit);
        }
      }
    }

    delete data;
  }
}

ModelBank LearnModelBank(int n_trials, const char *base) {
  sw::DataManager data_manager(16, false, false);

  library::timer::Timer t;

  ModelBank model_bank;
  model_bank.AddRayModel("BOX", 10.0, 0.25);
  model_bank.AddRayModel("STAR", 10.0, 0.25);
  model_bank.AddRayModel("NOOBJ", 10.0, 0.5);

  std::string bs(base);

  char fn[1000];
  int step = 0;

  while (true) {
    t.Start();
    RunTrials(&model_bank, &data_manager, n_trials);
    printf("Took %5.3f sec to run %d trials, saving step %d result \n", t.GetMs()/1e3, n_trials, step);

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
