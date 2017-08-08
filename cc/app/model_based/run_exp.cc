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

#include <boost/archive/binary_iarchive.hpp>

#include "library/timer/timer.h"
#include "library/sim_world/sim_world.h"
#include "library/sim_world/data.h"

#include "app/model_based/model_bank.h"
#include "app/model_based/ray_model.h"
#include "app/model_based/detection_map.h"
#include "app/model_based/observation.h"

namespace ge = library::geometry;
namespace sw = library::sim_world;

namespace mb = app::model_based;

mb::ModelBank LoadModelBank(const char *fn) {
  mb::ModelBank model_bank;
  std::ifstream ifs(fn);
  boost::archive::binary_iarchive ia(ifs);
  ia >> model_bank;

  return model_bank;
}

mb::DetectionMap BuildMap(const std::vector<ge::Point> &hits, const mb::ModelBank &model_bank) {
  mb::DetectionMap detection_map(25.0, 0.3, model_bank);

  library::timer::Timer t;

  t.Start();
  detection_map.ProcessObservations(hits);
  printf("Took %5.3f ms to build detection map\n", t.GetMs());

  return detection_map;
}

void SaveShapes(const sw::SimWorld &sim, const char *fn) {
  std::ofstream shapes_file(fn);
  for (auto &s : sim.GetShapes()) {
    auto c = s.GetCenter();
    shapes_file << c(0) << ", " << c(1) << ", " << s.GetAngle() << ", " << s.GetName() << std::endl;
  }
  shapes_file.close();
}

void SaveHits(const std::vector<ge::Point> &hits, const char *fn) {
  printf("Saving hits...\n");
  std::ofstream data_file(fn);
  for (const auto &hit : hits) {
    data_file << hit.x << "," << hit.y << std::endl;
  }
  data_file.close();
}

void SaveGroundTruth(const sw::SimWorld &sim, const char *fn) {
  printf("Saving ground truth...\n");

  std::ofstream gt_file(fn);

  std::vector<ge::Point> query_points;
  std::vector<float> gt_labels;
  sim.GenerateGrid(25.0, &query_points, &gt_labels, 0.3);

  for (size_t i=0; i<query_points.size(); i++) {
    float x = query_points[i].x;
    float y = query_points[i].y;
    float p_gt = gt_labels[i];

    gt_file << x << "," << y << "," << p_gt << std::endl;
  }
  gt_file.close();
}

void SaveResult(const mb::DetectionMap &detection_map, const std::string &cn, const char *fn) {
  std::ofstream res_file(fn);

  printf("Saving %s map...\n", cn.c_str());

  auto map = detection_map.GetScores();
  for (auto it = map.begin(); it != map.end(); it++) {
    const mb::ObjectState &os = it->first;

    // Check class
    if (os.GetClassname() != cn) {
      continue;
    }

    float x = os.GetPos()(0);
    float y = os.GetPos()(1);

    double angle = os.GetTheta();

    double score = detection_map.GetScore(os);
    //double logodds = detection_map.GetLogOdds(os);
    double prob = detection_map.GetProb(os);

    double p = prob;
    if (p < 1e-16)
      p = 1e-16;
    if (p > (1 - 1e-16))
      p = 1 - 1e-16;
    double logodds = -log(1.0/p - 1);

    res_file << x << "," << y << "," << angle << "," << score << "," << logodds << "," << prob << std::endl;
  }
  res_file.close();
}

void GenerateSyntheticScans(const mb::ModelBank &model_bank) {

  std::vector<double> angles;
  for (double sensor_angle = M_PI/4; sensor_angle < 3*M_PI/4; sensor_angle += 0.01) {
    angles.push_back(sensor_angle);
  }

  auto models = model_bank.GetModels();
  for (auto it = models.begin(); it != models.end(); it++) {
    printf("Generating synthetic scan for %s\n", it->first.c_str());

    const mb::RayModel &model = it->second;

    double x = 0.0;
    double y = 15.0;
    double object_angle = 0;
    mb::ObjectState os(x, y, object_angle, it->first);

    std::ofstream model_file(it->first + std::string(".csv"));
    for (double x = -10; x<10; x+=0.05) {
      for (double y = 5; y<25; y+=0.05) {
        mb::Observation x_hit(Eigen::Vector2d(x, y));
        double prob = model.GetProbability(os, x_hit);
        model_file << x_hit.GetX() << "," << x_hit.GetY() << "," << prob << std::endl;
      }
    }
    model_file.close();

    for (int i=0; i<4; i++) {
      auto hits = model.SampleObservations(os, angles);

      std::ofstream model_file;
      std::ostringstream ss;
      ss << it->first;
      ss << "_sample_" << i << ".csv";
      model_file.open(ss.str());

      for (const auto& hit : hits) {
        if (hit.GetRange() < 100) {
          double x = hit.GetX();
          double y = hit.GetY();

          model_file << x << "," << y << ",1" << std::endl;
        }
      }
      model_file.close();
    }

    for (int i=0; i<4; i++) {
      std::vector<int> n_grams;
      auto hits = model.SampleObservations(os, angles, &n_grams);

      std::ofstream model_file;
      std::ostringstream ss;
      ss << it->first;
      ss << "_sample_dependent_" << i << ".csv";
      model_file.open(ss.str());

      for (size_t i=0; i<hits.size(); i++) {
        const auto &hit = hits[i];

        if (hit.GetRange() < 100) {
          double x = hit.GetX();
          double y = hit.GetY();

          model_file << x << "," << y << "," << n_grams[i] << std::endl;
        }
      }
      model_file.close();
    }
  }
}

int main(int argc, char** argv) {
  printf("Model based detector\n");

  if (argc != 3) {
    printf("Usage: detector filename n_exp\n");
    return 1;
  }

  int n_exp = strtol(argv[2], NULL, 10);

  library::timer::Timer t;
  t.Start();
  mb::ModelBank model_bank = LoadModelBank(argv[1]);
  printf("Took %5.3f sec to load model bank\n", t.GetMs()/1e3);

  model_bank.PrintStats();

  GenerateSyntheticScans(model_bank);

  char fn[100];

  for (int experiment = 0; experiment < n_exp; experiment++) {
    printf("Experiment %d / %d\n", experiment, n_exp);

    sw::SimWorld sim(3);

    sprintf(fn, "shapes_%03d.csv", experiment);
    SaveShapes(sim, fn);

    std::vector<ge::Point> hits, origins;
    sim.GenerateSimData(&hits, &origins);

    printf("Have %ld hits\n", hits.size());

    // Write out data
    sprintf(fn, "data_%03d.csv", experiment);
    SaveHits(hits, fn);

    // Write out ground truth
    sprintf(fn, "ground_truth_%03d.csv", experiment);
    SaveGroundTruth(sim, fn);

    std::vector<Eigen::Vector2d> x_hits;
    for (ge::Point hit : hits) {
      x_hits.emplace_back(hit.x, hit.y);
    }

    for (int n_gram = 1; n_gram <= 2; n_gram++) {
      printf("Running with %d-gram\n", n_gram);

      // Build Map
      model_bank.UseNGram(n_gram);
      mb::DetectionMap detection_map = BuildMap(hits, model_bank);
      std::vector<std::string> classes = detection_map.GetClasses();

      // Eval at object locations
      for (auto &s : sim.GetShapes()) {
        auto c = s.GetCenter();
        double angle = s.GetAngle() * 180.0/M_PI;

        printf("\tShape %s at %5.3f, %5.3f with angle %5.3f\n", s.GetName().c_str(), c(0), c(1), angle);

        library::timer::Timer t;

        for (const auto &cn : classes) {
          mb::ObjectState os(c(0), c(1), angle, cn);
          t.Start();
          double score = detection_map.EvaluateObservationsForState(x_hits, os);
          double t_ms = t.GetMs();
          printf("\t\tShape %s (actually %s) at %5.3f, %5.3f with angle %5.3f --> %5.3f (%5.3f ms)\n",
              cn.c_str(), s.GetName().c_str(), os.GetPos()(0), os.GetPos()(1), os.GetTheta(), score, t_ms);
        }
      }

      // Write out result
      for (const std::string &cn : classes) {
        sprintf(fn, "result_%s_%02dgram_%03d.csv", cn.c_str(), n_gram, experiment);
        SaveResult(detection_map, cn, fn);
      }
    }
  }

  printf("Done\n");

  return 0;
}
