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
  for (int trial = 0; trial<500; trial++) {
    if (trial % 100 == 0) {
      printf("Trial %d\n", trial);
    }

    sw::Data *data = data_manager.GetData();

    sw::SimWorld *sim = data->GetSim();
    std::vector<ge::Point> *hits = data->GetHits();

    auto locs = sim->GetObjectLocations();

    for (auto p_center : locs) {
      Eigen::Vector2d x_sensor_object;
      x_sensor_object(0) = p_center.x;
      x_sensor_object(1) = p_center.y;

      double object_angle = 0.0;

      for (size_t i=0; i<hits->size(); i++) {
        Eigen::Vector2d x_hit;
        x_hit(0) = hits->at(i).x;
        x_hit(1) = hits->at(i).y;

        model.MarkObservationWorldFrame(x_sensor_object, object_angle, x_hit);
      }
    }

    delete data;
  }
  printf("Took %5.3f ms to build model\n", t.GetMs());
  data_manager.Finish();

  sw::SimWorld sim(0);
  sim.AddShape(sw::Shape::CreateStar(3.0, 3.0, 2.0));

  std::vector<ge::Point> hits, origins;
  std::vector<ge::Point> points;
  std::vector<float> labels;

  DetectionMap detection_map(10.0, 0.25, model);

  t.Start();
  sim.GenerateSimData(&hits, &origins);
  sim.GenerateSimData(&points, &labels);
  printf("Took %5.3f ms to generate %ld samples\n", t.GetMs(), hits.size());

  t.Start();
  detection_map.ProcessObservations(hits);
  printf("Took %5.3f ms to detect\n", t.GetMs());

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

    double score = it->second;

    res_file << x << "," << y << "," << score << std::endl;
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
