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
  std::uniform_real_distribution<double> rand_angle(-M_PI, M_PI);
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
    for (int neg=0; neg<5; neg++) {
      Eigen::Vector2d x_sensor_noobj;
      x_sensor_noobj(0) = unif(re);
      x_sensor_noobj(1) = unif(re);

      double object_angle = rand_angle(re);

      for (auto &shape : shapes) {
        auto center = shape.GetCenter();

        if ((x_sensor_noobj - center).norm() < 6) {
          continue;
        }
      }

      for (size_t i=0; i<hits->size(); i++) {
        Eigen::Vector2d x_hit;
        x_hit(0) = hits->at(i).x;
        x_hit(1) = hits->at(i).y;

        model.MarkNegativeObservationWorldFrame(x_sensor_noobj, object_angle, x_hit);
      }
    }

    delete data;
  }
  printf("Took %5.3f ms to build model\n", t.GetMs());
  data_manager.Finish();

  char fn[100];

  int n_experiments = 100;

  std::vector<std::pair<double, bool>> results;

  for (int experiment = 0; experiment < n_experiments; experiment++) {
    printf("Experiment %d / %d\n", experiment, n_experiments);

    sw::SimWorld sim(5);
    for (auto &s : sim.GetShapes()) {
      auto c = s.GetCenter();
      double x = c(0);
      double y = c(1);
      double angle = s.GetAngle();
      double sym = 72.0;
      printf("\tShape %s at %5.3f, %5.3f with angle %5.3f (ie, %5.3f)\n",
          s.GetName().c_str(), x, y, angle*180.0/M_PI, fmod(angle*180.0/M_PI, sym));
    }

    std::vector<ge::Point> hits, origins;
    std::vector<ge::Point> points;
    std::vector<float> labels;

    DetectionMap detection_map(20.0, 0.15, model);

    t.Start();
    sim.GenerateSimData(&hits, &origins);
    sim.GenerateSimData(&points, &labels);
    printf("Took %5.3f ms to generate %ld samples\n", t.GetMs(), hits.size());

    t.Start();
    detection_map.ProcessObservations(hits);
    printf("Took %5.3f ms to detect\n", t.GetMs());

    // Non max suppression
    t.Start();
    auto detections = detection_map.GetMaxDetections(10);
    printf("Took %5.3f ms to do non-max suppression\n", t.GetMs());

    for (auto it = detections.begin(); it != detections.end(); it++) {
      const ObjectState &os = it->first;
      printf("\tDetected max at %5.3f, %5.3f at %5.3f deg (val = %7.3f)\n",
          os.pos.x, os.pos.y, os.angle, it->second);
    }

    // Evaluate detections
    std::vector<sw::Shape> shapes(sim.GetShapes());

    int count_good = 0;
    int count_total = detections.size();

    while (!detections.empty()) {
      if (shapes.empty()) {
        for (auto it = detections.begin(); it != detections.end(); it++) {
          results.push_back(std::pair<double, bool>(it->second, false));
        }
        break;
      }

      // Find max detection
      ObjectState os = detections.begin()->first;
      double score = detections.begin()->second;
      for (auto it = detections.begin(); it != detections.end(); it++) {
        if (it->second > score) {
          os = it->first;
          score = it->second;
        }
      }

      detections.erase(os);

      // Is it valid or not?
      bool valid = false;
      for (auto shape = shapes.begin(); shape != shapes.end(); shape++) {
        if (!shape->IsInside(os.pos.x, os.pos.y))
          continue;

        double object_angle = shape->GetAngle();
        double sym = 72.0;
        object_angle = fmod(object_angle*180.0/M_PI, sym);

        double detection_angle = os.angle;
        detection_angle = fmod(detection_angle*180.0/M_PI, sym);

        double diff = object_angle - detection_angle;
        while (diff < -180.0) diff += 360.0;
        while (diff > 180.0) diff -= 360.0;

        if (diff < 9) {
          valid = true;
          shapes.erase(shape);
          count_good++;
          break;
        }
      }

      results.push_back(std::pair<double, bool>(score, valid));
    }

    printf("%d / %d detections good\n", count_good, count_total);

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
    std::ofstream res_file;
    sprintf(fn, "result_%03d.csv", experiment);
    res_file.open(fn);

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

  // Save scores and good/bad results
  std::ofstream pr_file;
  pr_file.open("pr_scores.csv");
  for (auto &res : results) {
    pr_file << res.first << ", " << res.second << std::endl;
  }
  pr_file.close();

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
    for (double percentile = 0.01; percentile <= 0.99; percentile+=0.01) {
      double range = model.GetExpectedRange(x_sensor_object, object_angle, sensor_angle, percentile);
      double x = cos(sensor_angle)*range;
      double y = sin(sensor_angle)*range;
      model_file << x << "," << y << "," << percentile << std::endl;
    }
  }
  model_file.close();

  std::ofstream noobj_file;
  noobj_file.open("noobj.csv");

  printf("Simulating no object at %5.3f, %5.3f\n", x_sensor_object(0), x_sensor_object(1));

  //double exp_range = noobj.GetExpectedRange(x_sensor_object, object_angle, M_PI/2, 0.50);
  //printf("Expected range = %5.3f\n", exp_range);

  //Eigen::Vector2d x_hit;

  //for (double y = 1.0; y < 10.0; y+=0.25) {
  //  x_hit << 0.0, y;
  //  printf("Prob hit at %5.3f, %5.3f = %5.3f\n", x_hit(0), x_hit(1), noobj.GetProbability(x_sensor_object, object_angle, x_hit));
  //}

  for (double sensor_angle = -M_PI; sensor_angle < M_PI; sensor_angle += 0.01) {
    for (double percentile = 0.01; percentile <= 0.99; percentile+=0.01) {
      double range = model.GetExpectedRangeNoObject(x_sensor_object, object_angle, sensor_angle, percentile);
      double x = cos(sensor_angle)*range;
      double y = sin(sensor_angle)*range;
      noobj_file << x << "," << y << "," << percentile << std::endl;
    }
  }
  noobj_file.close();

  printf("Done\n");

  return 0;
}
