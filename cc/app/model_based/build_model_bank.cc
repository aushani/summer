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
  double pos_res = 0.3; // 30 cm
  double angle_res = 45.0 * M_PI/180.0; // 45 deg

  // Sampling positions
  double lower_bound = -20.0;
  double upper_bound = 20.0;
  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  //std::uniform_real_distribution<double> rand_angle(-M_PI, M_PI);
  std::uniform_real_distribution<double> rand_angle(-0.01, 0.01);

  std::uniform_real_distribution<double> jitter_pos(-pos_res, pos_res);
  std::uniform_real_distribution<double> jitter_angle(-angle_res, angle_res);

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

    // Convert hits to observations
    std::vector<Observation> observations;
    for (const auto& h : *hits) {
      observations.emplace_back(Eigen::Vector2d(h.x, h.y));
    }

    auto shapes = sim->GetShapes();

    for (auto &shape : shapes) {
      for (int i=0; i<10; i++) {
        double dx = jitter_pos(re);
        double dy = jitter_pos(re);
        double dt = jitter_angle(re);

        ObjectState os(shape.GetCenter()(0) + dx, shape.GetCenter()(1) + dy, shape.GetAngle() + dt, shape.GetName());

        for (const auto& obs : observations) {
          model_bank->MarkObservation(os, obs);
        }
      }
    }

    // Negative mining
    for (size_t neg=0; neg<shapes.size(); neg++) {
      double x = unif(re);
      double y = unif(re);
      double object_angle = rand_angle(re);

      ObjectState os(x, y, object_angle, "NOOBJ");

      if (std::abs(x) < 10 && std::abs(y) < 10) {
        continue;
      }

      bool too_close = false;

      for (auto &shape : shapes) {
        const auto &center = shape.GetCenter();

        if ((os.GetPos() - center).norm() < pos_res) {
          double d_angle = shape.GetAngle() - object_angle;
          while (d_angle < -M_PI) d_angle += 2*M_PI;
          while (d_angle > M_PI) d_angle -= 2*M_PI;

          if (std::abs(d_angle) < angle_res) {
            too_close = true;
            break;
          }
        }
      }

      if (!too_close) {
        for (size_t i=0; i<hits->size(); i++) {
          Observation x_hit(Eigen::Vector2d(hits->at(i).x, hits->at(i).y));
          model_bank->MarkObservation(os, x_hit);
        }
      }
    }

    delete data;
  }
}

ModelBank LearnModelBank(int n_trials, const char *base) {
  sw::DataManager data_manager(16, false, false);

  library::timer::Timer t;

  double area = 20*20 - 10*10;
  double pos_res = 0.3;
  double n_shapes = 1.5;

  double prior = pos_res*pos_res*n_shapes / area;

  ModelBank model_bank;
  model_bank.AddRayModel("BOX", 5.0, prior);
  model_bank.AddRayModel("STAR", 5.0, prior);
  model_bank.AddRayModel("NOOBJ", 5.0, 1 - 2*prior);

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
