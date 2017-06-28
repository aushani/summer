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
#include <chrono>

#include "library/hilbert_map/hilbert_map.h"

#include "app/obj_sim/sim_world.h"
#include "app/obj_sim/learned_kernel.h"
#include "app/obj_sim/data.h"

namespace hm = library::hilbert_map;

int main(int argc, char** argv) {
  printf("Object sim\n");

  // Start data threads
  DataManager data_manager(16);

  std::vector<float> w;
  int num_epochs = 1000;

  hm::Opt opt;
  opt.min = -2.0;
  opt.max = 2.0;
  opt.inducing_points_n_dim = 40;
  opt.learning_rate = 0.1;

  opt.l1_reg = 0.001;

  float decay_rate = 0.9999;

  for (int epoch = 0; epoch<num_epochs; epoch++) {
    printf("\n--- EPOCH %02d / %02d (learning rate = %7.5f) ---\n", epoch, num_epochs, opt.learning_rate);

    // Get data
    printf("\tGetting data...\n");
    Data *data = data_manager.GetData();
    printf("\tHave %ld points\n", data->GetPoints()->size());

    // This kernel is actually w
    printf("\tSetting up kernel...\n");
    LearnedKernel w_kernel(20.0, 0.5);
    for (size_t i = 0; i<w_kernel.GetDimSize(); i++) {
      for (size_t j = 0; j<w_kernel.GetDimSize(); j++) {
        w_kernel.SetPixel(i, j, 0.0f);
      }
    }

    for (const Shape& shape : data->GetSim()->GetObjects()) {
      Eigen::Vector2d center = shape.GetCenter();

      w_kernel.SetLocation(center(0), center(1), 1.0f);
    }
    printf("\tHave %ldx%ld kernel\n", w_kernel.GetDimSize(), w_kernel.GetDimSize());

    // Make a map with this "kernel"
    auto tic_map = std::chrono::steady_clock::now();
    hm::HilbertMap w_map(*data->GetPoints(), *data->GetLabels(), w_kernel, opt, epoch>0 ? NULL:w.data());
    auto toc_map = std::chrono::steady_clock::now();
    auto t_ms_map = std::chrono::duration_cast<std::chrono::milliseconds>(toc_map - tic_map);
    printf("\tLearned map in %ld ms\n", t_ms_map.count());

    w = w_map.GetW();

    opt.learning_rate *= decay_rate;
    if (opt.learning_rate < 0.01)
      break;

    delete data;
  }

  // Now actually make a HM with the kernel we learned
  LearnedKernel kernel(opt.max - opt.min, (opt.max - opt.min)/opt.inducing_points_n_dim);
  for (int i=0; i<opt.inducing_points_n_dim; i++) {
    for (int j=0; j<opt.inducing_points_n_dim; j++) {
      int idx = i * opt.inducing_points_n_dim + j;
      float val = w[idx];
      kernel.SetPixel(i, j, val);
    }
  }

  std::vector<hm::Point> hits, origins;

  SimWorld sim;
  sim.GenerateSimData(&hits, &origins);

  printf("Making actual map\n");
  hm::HilbertMap map(hits, origins, kernel);
  printf("Done, writing files...\n");

  // Write out data
  std::ofstream points_file;
  points_file.open("points.csv");
  for (size_t i=0; i<hits.size(); i++) {
    points_file << hits[i].x << ", " << hits[i].y << std::endl;
  }
  points_file.close();

  // Evaluate
  std::vector<hm::Point> query_points;
  std::vector<float> gt_labels;
  sim.GenerateGrid(&query_points, &gt_labels);

  // Write out ground truth
  std::ofstream gt_file;
  gt_file.open("ground_truth.csv");
  for (size_t i=0; i<query_points.size(); i++) {
    float x = query_points[i].x;
    float y = query_points[i].y;
    float p = gt_labels[i];

    gt_file << x << ", " << y << ", " << p << std::endl;
  }
  gt_file.close();

  auto tic = std::chrono::steady_clock::now();
  std::vector<float> probs = map.GetOccupancy(query_points);
  auto toc = std::chrono::steady_clock::now();
  auto t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);
  printf("Evaluated grid in %ld ms (%5.3f ms / call)\n", t_ms.count(), ((double)t_ms.count())/query_points.size());

  // Write grid to file
  std::ofstream grid_file;
  grid_file.open("grid.csv");
  for (size_t i=0; i<query_points.size(); i++) {
    float x = query_points[i].x;
    float y = query_points[i].y;
    float p = probs[i];

    grid_file << x << ", " << y << ", " << p << std::endl;
  }
  grid_file.close();

  // Write kernel to file
  std::ofstream kernel_file;
  kernel_file.open("kernel.csv");
  for (float x = -kernel.MaxSupport(); x<=kernel.MaxSupport(); x+=0.05) {
    for (float y = -kernel.MaxSupport(); y<=kernel.MaxSupport(); y+=0.05) {
      float val = kernel.Evaluate(x, y);
      kernel_file << x << ", " << y << ", " << val << std::endl;
    }
  }
  kernel_file.close();

  printf("Done\n");

  return 0;
}
