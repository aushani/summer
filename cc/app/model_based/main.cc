#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <sstream>
#include <vector>
#include <map>
#include <iterator>
#include <math.h>
#include <random>

#include "library/timer/timer.h"
#include "library/sim_world/sim_world.h"
#include "library/sim_world/data.h"

#include "model_bank.h"
#include "ray_model.h"
#include "detection_map.h"

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

DetectionMap BuildMap(const std::vector<ge::Point> &hits, const ModelBank model_bank) {
  DetectionMap detection_map(20.0, 0.3, model_bank);

  library::timer::Timer t;

  t.Start();
  detection_map.ProcessObservations(hits);
  printf("Took %5.3f ms to build detection map\n", t.GetMs());

  return detection_map;
}


void GenerateSyntheticScans(const ModelBank &model_bank) {
  auto models = model_bank.GetModels();
  for (auto it = models.begin(); it != models.end(); it++) {
    printf("Generating synethic scan for %s\n", it->first.c_str());

    std::ofstream model_file;
    model_file.open(it->first + std::string(".csv"));

    Eigen::Vector2d x_sensor_object;
    x_sensor_object(0) = 0.0;
    x_sensor_object(1) = 50.0;
    double object_angle = (2*M_PI) / 20;

    for (double sensor_angle = M_PI/2 * 0.9; sensor_angle < M_PI/2 * 1.1; sensor_angle += 0.001) {
      for (double percentile = 0.01; percentile <= 0.99; percentile+=0.01) {
        //if (it->first == "NOOBJ" && std::abs(percentile-0.5)<0.01) printf("\tangle: %5.3f ", sensor_angle);
        double range = it->second.GetExpectedRange(x_sensor_object, object_angle, sensor_angle, percentile);
        double x = cos(sensor_angle)*range;
        double y = sin(sensor_angle)*range;
        model_file << x << "," << y << "," << percentile << std::endl;
      }
    }
    model_file.close();
  }
}

int main(int argc, char** argv) {
  printf("Model based detector\n");

  int n_trials = 1000;
  if (argc > 1)
    n_trials = strtol(argv[1], NULL, 10);

  int n_exp = 3;
  if (argc > 2)
    n_exp = strtol(argv[2], NULL, 10);

  ModelBank model_bank = LearnModelBank(n_trials);

  GenerateSyntheticScans(model_bank);

  char fn[100];

  for (int experiment = 0; experiment < n_exp; experiment++) {
    printf("Experiment %d / %d\n", experiment, n_exp);

    sw::SimWorld sim(1);

    std::vector<ge::Point> hits, origins;
    sim.GenerateSimData(&hits, &origins);

    std::vector<Eigen::Vector2d> x_hits;
    for (ge::Point hit : hits) {
      Eigen::Vector2d x;
      x(0) = hit.x;
      x(1) = hit.y;
      x_hits.push_back(x);
    }

    DetectionMap detection_map = BuildMap(hits, model_bank);

    // At object?
    for (auto &s : sim.GetShapes()) {
      auto c = s.GetCenter();
      double angle = s.GetAngle() * 180.0/M_PI;

      printf("\tShape %s at %5.3f, %5.3f with angle %5.3f\n", s.GetName().c_str(), c(0), c(1), angle);

      library::timer::Timer t;

      for (const auto &cn : model_bank.GetClasses()) {
        ObjectState os(c(0), c(1), angle, cn);
        t.Start();
        double score = detection_map.EvaluateObservationsForState(x_hits, os);
        double t_ms = t.GetMs();
        printf("\t\tShape %s (actually %s) at %5.3f, %5.3f with angle %5.3f --> %5.3f (%5.3f ms)\n",
            cn.c_str(), s.GetName().c_str(), os.pos.x, os.pos.y, os.angle, score, t_ms);
      }
    }

    // Write out data
    printf("Saving data...\n");
    std::ofstream data_file;
    sprintf(fn, "data_%03d.csv", experiment);
    data_file.open(fn);
    for (size_t i=0; i<hits.size(); i++) {
      data_file << hits[i].x << "," << hits[i].y << std::endl;
    }
    data_file.close();

    // Write out result
    const std::vector<std::string> classes = detection_map.GetClasses();
    for (const std::string &cn : classes) {
      std::ofstream res_file;
      sprintf(fn, "result_%s_%03d.csv", cn.c_str(), experiment);
      res_file.open(fn);

      printf("Saving %s map...\n", cn.c_str());

      auto map = detection_map.GetScores();
      for (auto it = map.begin(); it != map.end(); it++) {
        const ObjectState &os = it->first;

        // Check class
        if (os.classname != cn) {
          continue;
        }

        float x = os.pos.x;
        float y = os.pos.y;

        double angle = os.angle;
        double score = it->second;

        double prob = detection_map.GetProb(os);
        //double prob = 0.5;

        res_file << x << "," << y << "," << angle << "," << score << ", " << prob << std::endl;
      }
      res_file.close();
    }

    // Write out ground truth
    printf("Saving ground truth...\n");
    std::ofstream gt_file;
    sprintf(fn, "ground_truth_%03d.csv", experiment);
    gt_file.open(fn);

    std::vector<ge::Point> query_points;
    std::vector<float> gt_labels;
    sim.GenerateGrid(10.0, &query_points, &gt_labels);

    for (size_t i=0; i<query_points.size(); i++) {
      float x = query_points[i].x;
      float y = query_points[i].y;
      float p_gt = gt_labels[i];

      gt_file << x << ", " << y << ", " << p_gt << std::endl;
    }
    gt_file.close();
  }

  printf("Done\n");

  return 0;
}
