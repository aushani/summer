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
#include "ray_model.h"
#include "detection_map.h"

namespace ge = library::geometry;
namespace sw = library::sim_world;

int main(int argc, char** argv) {
  printf("Model based detector\n");

  sw::DataManager data_manager(64, false, false);

  printf("Started data manager\n");

  library::timer::Timer t;

  RayModel model(3.0);

  t.Start();
  double lower_bound = -8;
  double upper_bound = 8;
  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  std::uniform_real_distribution<double> rand_size(1.0, 2.0);
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine re(seed);

  for (int trial = 0; trial<10000; trial++) {
    if (trial % 100 == 0) {
      printf("Trial %d\n", trial);
    }

    sw::Data *data = data_manager.GetData();

    sw::SimWorld *sim = data->GetSim();
    std::vector<ge::Point> *hits = data->GetHits();

    auto shapes = sim->GetShapes();

    for (auto &shape : shapes) {
      Eigen::Vector2d x_sensor_object = shape.GetCenter();
      double object_angle = shape.GetAngle();

      for (size_t i=0; i<hits->size(); i++) {
        Eigen::Vector2d x_hit;
        x_hit(0) = hits->at(i).x;
        x_hit(1) = hits->at(i).y;

        model.MarkObservationWorldFrame(x_sensor_object, object_angle, x_hit);
      }
    }

    // Negative mining
    for (int neg_ex = 0; neg_ex < 5; neg_ex ++) {
      double x = unif(re);
      double y = unif(re);
      if (sim->IsOccupied(x, y))
        continue;

      Eigen::Vector2d x_sensor_object;
      x_sensor_object << x, y;

      double object_angle = 0.0;

      for (size_t i=0; i<hits->size(); i++) {
        Eigen::Vector2d x_hit;
        x_hit(0) = hits->at(i).x;
        x_hit(1) = hits->at(i).y;

        model.MarkNegativeObservationWorldFrame(x_sensor_object, object_angle, x_hit);
      }
    }

    delete data;
  }
  printf("Took %5.3f ms to build model\n", t.GetMs());
  data_manager.Finish();

  sw::SimWorld sim(5);
  //auto star1 = sw::Shape::CreateStar(0.0, 3.0, 2.0);
  //auto star2 = sw::Shape::CreateStar(0.0, -3.0, 2.0);
  //star2.Rotate((2*M_PI/10));
  //sim.AddShape(star1);
  //sim.AddShape(star2);
  //sim.AddShape(sw::Shape::CreateBox(0.0, 3.0, 2.0, 2.0));
  for (auto &s : sim.GetShapes()) {
    auto c = s.GetCenter();
    double x = c(0);
    double y = c(1);
    double angle = s.GetAngle();
    double sym = 72.0;
    printf("Shape at %5.3f, %5.3f with angle %5.3f (ie, %5.3f)\n", x, y, angle*180.0/M_PI, fmod(angle*180.0/M_PI, sym));
  }

  std::vector<ge::Point> hits, origins;
  std::vector<ge::Point> points;
  std::vector<float> labels;

  DetectionMap detection_map(20.0, 0.3, model);

  t.Start();
  sim.GenerateSimData(&hits, &origins);
  sim.GenerateSimData(&points, &labels);
  printf("Took %5.3f ms to generate %ld samples\n", t.GetMs(), hits.size());

  t.Start();
  detection_map.ProcessObservations(hits);
  printf("Took %5.3f ms to detect\n", t.GetMs());

  // Non max suppression
  t.Start();
  detection_map.ListMaxDetections();
  printf("Took %5.3f ms to do non-max suppression\n", t.GetMs());

  // Write out data
  printf("Saving data...\n");
  std::ofstream data_file;
  data_file.open("data.csv");
  for (size_t i=0; i<hits.size(); i++) {
    data_file << hits[i].x << "," << hits[i].y << std::endl;
  }
  data_file.close();

  // Write out result
  std::ofstream res_file;
  res_file.open("result.csv");

  auto map = detection_map.GetScores();
  for (auto it = map.begin(); it != map.end(); it++) {
    float x = it->first.pos.x;
    float y = it->first.pos.y;

    double angle = it->first.angle;
    //if ( fabs(angle) > 0.1 )
    //  continue;

    double score = it->second;

    double prob = 0.0;
    if (score < -100)
      prob = 0.0;
    if (score > 100)
      prob = 1.0;
    prob = 1/(1+exp(-score));

    res_file << x << "," << y << "," << angle << "," << score << ", " << prob << std::endl;
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

  // Model...
  std::ofstream model_file;
  model_file.open("model.csv");

  Eigen::Vector2d x_sensor_object;
  x_sensor_object(0) = 0.0;
  x_sensor_object(1) = 5.0;
  double object_angle = (2*M_PI) / 20;

  printf("Simulating object at %5.3f, %5.3f\n", x_sensor_object(0), x_sensor_object(1));

  //double exp_range = model.GetExpectedRange(x_sensor_object, object_angle, M_PI/2, 0.50);
  //printf("Expected range = %5.3f\n", exp_range);

  //Eigen::Vector2d x_hit;

  //for (double y = 1.0; y < 10.0; y+=0.25) {
  //  x_hit << 0.0, y;
  //  printf("Prob hit at %5.3f, %5.3f = %5.3f\n", x_hit(0), x_hit(1), model.GetProbability(x_sensor_object, object_angle, x_hit));
  //}

  for (double sensor_angle = -M_PI; sensor_angle < M_PI; sensor_angle += 0.01) {
    for (double percentile = 0.2; percentile <= 0.8; percentile+=0.2) {
      double range = model.GetExpectedRange(x_sensor_object, object_angle, sensor_angle, percentile);
      double x = cos(sensor_angle)*range;
      double y = sin(sensor_angle)*range;
      model_file << x << "," << y << "," << percentile << std::endl;
    }
  }
  model_file.close();

  printf("Done\n");

  return 0;
}
