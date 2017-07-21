#include <iostream>
#include <fstream>

#include "library/timer/timer.h"
#include "library/sim_world/sim_world.h"
#include "library/sim_world/data.h"

#include "model_bank.h"

namespace ge = library::geometry;
namespace sw = library::sim_world;

ModelBank LearnModelBank(int n_trials) {
  sw::DataManager data_manager(16, false, false);

  library::timer::Timer t;
  t.Start();

  ModelBank model_bank;
  model_bank.AddRayModel("BOX", 10.0, 0.25);
  model_bank.AddRayModel("STAR", 10.0, 0.25);
  model_bank.AddRayModel("NOOBJ", 10.0, 0.5);

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
      printf("\tTrial %d / %d\n", trial, n_trials);
    }

    // Get sim data
    sw::Data *data = data_manager.GetData();
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

        model_bank.MarkObservation(shape.GetName(), x_sensor_object, object_angle, x_hit);
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
          //printf("%5.3f, %5.3f is too close to %s at %5.3f, %5.3f\n",
          //    x_sensor_noobj(0), x_sensor_noobj(1),
          //    shape.GetName().c_str(), center(0), center(1));
          break;
        }
      }

      if (!too_close) {
        for (size_t i=0; i<hits->size(); i++) {
          Eigen::Vector2d x_hit;
          x_hit(0) = hits->at(i).x;
          x_hit(1) = hits->at(i).y;

          model_bank.MarkObservation("NOOBJ", x_sensor_noobj, object_angle, x_hit);
        }
      }
    }

    delete data;
  }
  data_manager.Finish();
  printf("Took %5.3f ms to build model\n", t.GetMs());

  return model_bank;
}

int main(int argc, char** argv) {
  printf("Building model bank...\n");

  if (argc != 3) {
    printf("Usage: build_model_bank n_trials filename\n");
    return 1;
  }

  int n_trials = strtol(argv[1], NULL, 10);
  ModelBank model_bank = LearnModelBank(n_trials);

  std::ofstream ofs(argv[2]);
  boost::archive::text_oarchive oa(ofs);
  oa << model_bank;
}
