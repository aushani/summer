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

#include "model_bank.h"
#include "ray_model.h"
#include "detection_map.h"
#include "observation.h"

namespace ge = library::geometry;
namespace sw = library::sim_world;

ModelBank LoadModelBank(const char *fn) {
  ModelBank model_bank;
  std::ifstream ifs(fn);
  boost::archive::binary_iarchive ia(ifs);
  ia >> model_bank;

  return model_bank;
}

DetectionMap BuildMap(const std::vector<ge::Point> &hits, const ModelBank model_bank) {
  DetectionMap detection_map(25.0, 0.3, model_bank);

  library::timer::Timer t;

  t.Start();
  detection_map.ProcessObservations(hits);
  printf("Took %5.3f ms to build detection map\n", t.GetMs());

  return detection_map;
}


void GenerateSyntheticScans(const ModelBank &model_bank) {

  std::vector<double> angles;
  for (double sensor_angle = M_PI/4; sensor_angle < 3*M_PI/4; sensor_angle += 0.01) {
    angles.push_back(sensor_angle);
  }

  auto models = model_bank.GetModels();
  for (auto it = models.begin(); it != models.end(); it++) {
    printf("Generating synthetic scan for %s\n", it->first.c_str());

    const RayModel &model = it->second;

    double x = 0.0;
    double y = 15.0;
    double object_angle = 0;
    ObjectState os(x, y, object_angle, it->first);

    std::ofstream model_file(it->first + std::string(".csv"));
    for (double x = -10; x<10; x+=0.05) {
      for (double y = 10; y<20; y+=0.05) {
        Observation x_hit(Eigen::Vector2d(x, y));
        double likelihood = model.GetLikelihood(os, x_hit);
        if (likelihood < 0) {
          likelihood = 0;
        }

        model_file << x_hit.GetX() << "," << x_hit.GetY() << "," << likelihood << std::endl;
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
  ModelBank model_bank = LoadModelBank(argv[1]);
  printf("Took %5.3f sec to load model bank\n", t.GetMs()/1e3);

  model_bank.PrintStats();

  GenerateSyntheticScans(model_bank);

  char fn[100];

  for (int experiment = 0; experiment < n_exp; experiment++) {
    printf("Experiment %d / %d\n", experiment, n_exp);

    sw::SimWorld sim(3);
    //sw::SimWorld sim(0);
    //sim.AddShape(sw::Shape::CreateStar(0, 20, 1.5));
    //sim.AddShape(sw::Shape::CreateBox(0, 30, 3, 6));
    //sim.AddShape(sw::Shape::CreateBox(30, 0, 3, 6));
    //sim.AddShape(sw::Shape::CreateBox(0, -30, 3, 6));
    //sim.AddShape(sw::Shape::CreateBox(-30, 0, 3, 6));

    std::vector<ge::Point> hits, origins;
    sim.GenerateSimData(&hits, &origins);

    printf("Have %ld hits\n", hits.size());

    //std::vector<ge::Point> mod_hits;
    //mod_hits.push_back(hits[hits.size()*0.00]);
    //mod_hits.push_back(hits[hits.size()*0.25]);
    //mod_hits.push_back(hits[hits.size()*0.50]);
    //mod_hits.push_back(hits[hits.size()*0.75]);
    //hits = mod_hits;

    std::vector<Eigen::Vector2d> x_hits;
    for (ge::Point hit : hits) {
      Eigen::Vector2d x;
      x(0) = hit.x;
      x(1) = hit.y;
      x_hits.push_back(x);
    }

    DetectionMap detection_map = BuildMap(hits, model_bank);

    std::vector<std::string> classes = detection_map.GetClasses();

    //library::timer::Timer t;
    //t.Start();
    //auto detections = detection_map.GetMaxDetections(0);
    //double ms = t.GetMs();
    //printf("Took %5.3f ms for max detectinos\n", ms);
    //for (auto it = detections.begin(); it != detections.end(); it++) {
    //  auto d = it->first;
    //  printf("\tMax Detection: %s, %5.3f, %5.3f at angle %5.3f ==> %5.3f\n",
    //      d.GetClassname().c_str(), d.GetPos()(0), d.GetPos()(1), d.GetTheta(), it->second);
    //}

    // At object?
    for (auto &s : sim.GetShapes()) {
      auto c = s.GetCenter();
      double angle = s.GetAngle() * 180.0/M_PI;

      printf("\tShape %s at %5.3f, %5.3f with angle %5.3f\n", s.GetName().c_str(), c(0), c(1), angle);

      library::timer::Timer t;

      for (const auto &cn : classes) {
        ObjectState os(c(0), c(1), angle, cn);
        t.Start();
        double score = detection_map.EvaluateObservationsForState(x_hits, os);
        double t_ms = t.GetMs();
        printf("\t\tShape %s (actually %s) at %5.3f, %5.3f with angle %5.3f --> %5.3f (%5.3f ms)\n",
            cn.c_str(), s.GetName().c_str(), os.GetPos()(0), os.GetPos()(1), os.GetTheta(), score, t_ms);
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
    for (const std::string &cn : classes) {
      std::ofstream res_file;
      sprintf(fn, "result_%s_%03d.csv", cn.c_str(), experiment);
      res_file.open(fn);

      printf("Saving %s map...\n", cn.c_str());

      auto map = detection_map.GetScores();
      for (auto it = map.begin(); it != map.end(); it++) {
        const ObjectState &os = it->first;

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
        if (p < 1e-15)
          p = 1e-15;
        if (p > (1 - 1e-15))
          p = 1 - 1e-15;
        double logodds = -log(1.0/p - 1);

        res_file << x << "," << y << "," << angle << "," << score << "," << logodds << "," << prob << std::endl;
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
    sim.GenerateGrid(30.0, &query_points, &gt_labels, 0.2);

    for (size_t i=0; i<query_points.size(); i++) {
      float x = query_points[i].x;
      float y = query_points[i].y;
      float p_gt = gt_labels[i];

      gt_file << x << "," << y << "," << p_gt << std::endl;
    }
    gt_file.close();
  }

  printf("Done\n");

  return 0;
}
