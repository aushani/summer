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

#include "object_model.h"

namespace ge = library::geometry;
namespace sw = library::sim_world;

struct comp {
  bool operator() (const ge::Point p1, const ge::Point p2) {
    if (p1.x != p2.x) return p1.x < p2.x;
    return p1.y < p2.y;
  }
};

double EvalProb(const std::vector<ge::Point> &points, const std::vector<float> &labels, const ObjectModel &model, const ObjectModel &no_obj, const ge::Point &p) {
  double log_l_obj = 0;
  double log_l_noobj = 20;

  int updates = 0;

  for (size_t p_i=0; p_i<points.size(); p_i++) {
    double x = points[p_i].x - p.x;
    double y = points[p_i].y - p.y;
    ge::Point point(x, y);

    if (!model.InBounds(point))
      continue;

    double l_zo = model.EvaluateLikelihood(point, labels[p_i]);
    if (l_zo < 1e-15) l_zo = 1e-15;

    double update = log(l_zo);
    log_l_obj += update;

    double l_zno = no_obj.EvaluateLikelihood(point, labels[p_i]);
    if (l_zno < 1e-15) l_zno = 1e-15;
    log_l_noobj += log(l_zno);

    updates++;
  }

  //printf("%f vs %f (%d)\n", log_l_obj, log_l_noobj, updates);

  return log_l_obj - log_l_noobj;

  //double min = fmin(log_l_obj, log_l_noobj);
  //log_l_obj -= min;
  //log_l_noobj -= min;

  //if (log_l_obj > 100)
  //  return 1.0;
  //if (log_l_noobj > 100)
  //  return 0.0;

  //double sum = exp(log_l_obj) + exp(log_l_noobj);
  //double prob = exp(log_l_obj)/sum;

  //return prob;
}

std::map<ge::Point, double, comp> Detect(const std::vector<ge::Point> &points, const std::vector<float> &labels, const ObjectModel &model, const ObjectModel &no_obj) {

  double res = 0.25;

  std::map<ge::Point, double, comp> map;
  for (double x = -15; x<15; x+=res) {
    for (double y = -15; y<15; y+=res) {
      ge::Point p(x, y);
      map[p] = EvalProb(points, labels, model, no_obj, p);
    }
  }

  printf("Done\n");

  return map;
}

int main(int argc, char** argv) {
  printf("Model based detector\n");

  sw::DataManager data_manager(64, false, false);

  printf("Started data manager\n");

  library::timer::Timer t;

  ObjectModel model(2.0, 0.25);
  ObjectModel no_obj(2.0, 1.0);

  t.Start();
  for (int trial = 0; trial<1000; trial++) {
    if (trial % 100 == 0) {
      printf("Trial %d\n", trial);
    }

    sw::Data *data = data_manager.GetData();

    sw::SimWorld *sim = data->GetSim();
    std::vector<ge::Point> *points = data->GetObsPoints();
    std::vector<float> *labels = data->GetObsLabels();

    auto locs = sim->GetObjectLocations();

    for (size_t i=0; i<points->size(); i++) {
      for (auto p_center : locs) {
        ge::Point p_obj(points->at(i).x - p_center.x, points->at(i).y - p_center.y);
        model.Build(p_obj, labels->at(i));
      }
    }

    // Find a random location
    for (int neg=0; neg<5; neg++) {
      double lower_bound = -8;
      double upper_bound = 8;
      std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
      std::uniform_real_distribution<double> rand_size(1.0, 2.0);
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::default_random_engine re(seed);
      double x = unif(re);
      double y = unif(re);
      if (sim->IsOccupied(x, y)) {
        neg--;
        continue;
      }
      for (size_t i=0; i<points->size(); i++) {
        ge::Point p(points->at(i).x - x, points->at(i).y - y);
        no_obj.Build(p, labels->at(i));
      }
    }

    delete data;
  }
  printf("Took %5.3f ms to build model\n", t.GetMs());
  data_manager.Finish();

  sw::SimWorld sim(5);
  //sim.AddShape(sw::Shape::CreateStar(0, 0, 2.0));
  std::vector<ge::Point> points;
  std::vector<float> labels;

  t.Start();
  sim.GenerateSimData(&points, &labels);
  //sim.GenerateGrid(10.0, &points, &labels, 0.25);
  printf("Took %5.3f ms to generate %ld samples\n", t.GetMs(), points.size());

  for (auto p_center : sim.GetObjectLocations()) {
    printf("Score at obj is: %5.3f\n", EvalProb(points, labels, model, no_obj, p_center));
    printf("Score at obj + (2, 2) is: %5.3f\n", EvalProb(points, labels, model, no_obj, ge::Point(p_center.x + 2, p_center.y + 2)));
  }


  t.Start();
  auto map = Detect(points, labels, model, no_obj);
  printf("Took %5.3f ms to detect\n", t.GetMs());

  // Write out data
  printf("Saving data...\n");
  std::ofstream points_file;
  points_file.open("data.csv");
  for (size_t i=0; i<points.size(); i++) {
    points_file << points[i].x << ", " << points[i].y << ", " << labels[i] << std::endl;
  }
  points_file.close();

  // Write out model
  std::ofstream model_file;
  model_file.open("model.csv");

  for (double x = -2.0; x <= 2.0; x+=0.01) {
    for (double y = -2.0; y <= 2.0; y+=0.01) {
      float p_o = model.EvaluateLikelihood(ge::Point(x, y), 1.0);
      float p_f = model.EvaluateLikelihood(ge::Point(x, y), -1.0);

      model_file << x << ", " << y << ", " << p_o << ", " << p_f << std::endl;
    }
  }
  model_file.close();

  std::ofstream noobj_file;
  noobj_file.open("no_obj.csv");

  for (double x = -2.0; x <= 2.0; x+=0.01) {
    for (double y = -2.0; y <= 2.0; y+=0.01) {
      float p_o = no_obj.EvaluateLikelihood(ge::Point(x, y), 1.0);
      float p_f = no_obj.EvaluateLikelihood(ge::Point(x, y), -1.0);

      noobj_file << x << ", " << y << ", " << p_o << ", " << p_f << std::endl;
    }
  }
  noobj_file.close();

  // Write out result
  std::ofstream res_file;
  res_file.open("result.csv");

  for (auto it = map.begin(); it != map.end(); it++) {
    float x = it->first.x;
    float y = it->first.y;

    double p_m = it->second;
    double p = p_m > 0.5 ? 1:0;

    res_file << x << ", " << y << ", " << p << ", " << p_m << std::endl;
  }
  res_file.close();

  // Write out ground truth
  printf("Saving ground truth...\n");
  std::ofstream gt_file;
  gt_file.open("ground_truth.csv");

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

  printf("Done\n");

  return 0;
}
