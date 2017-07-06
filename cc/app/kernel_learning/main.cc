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
#include "library/sim_world/sim_world.h"

#include "app/kernel_learning/learned_kernel.h"
#include "app/kernel_learning/data.h"

namespace hm = library::hilbert_map;
namespace sw = library::sim_world;
using namespace app::kernel_learning;

void SaveKernel(const hm::IKernel &kernel, const char *fn) {
  std::ofstream kernel_file;
  kernel_file.open(fn);
  for (float x = -kernel.MaxSupport(); x<=kernel.MaxSupport(); x+=0.05) {
    for (float y = -kernel.MaxSupport(); y<=kernel.MaxSupport(); y+=0.05) {
      float val = kernel.Evaluate(x, y);
      kernel_file << x << ", " << y << ", " << val << std::endl;
    }
  }
  kernel_file.close();
}

int main(int argc, char** argv) {
  printf("Object sim\n");

  // Start data threads
  DataManager data_manager(64);

  std::vector<float> kernel_vector;
  int num_epochs = 1;
  if (argc > 1)
    num_epochs = strtol(argv[1], NULL, 10);

  hm::Opt opt_kernel;
  opt_kernel.min = -2.0;
  opt_kernel.max = 2.0;
  opt_kernel.inducing_points_n_dim = 20;
  opt_kernel.learning_rate = 0.002;
  opt_kernel.l1_reg = 0.001;

  hm::Opt opt_w;
  opt_w.min = -10.0;
  opt_w.max = 10.0;
  opt_w.inducing_points_n_dim = 100;
  opt_w.learning_rate = 0.1;
  opt_w.l1_reg = 0.001;

  float decay_rate = 0.9999;

  LearnedKernel kernel(opt_kernel.max - opt_kernel.min, (opt_kernel.max - opt_kernel.min)/opt_kernel.inducing_points_n_dim);
  kernel.CopyFrom(hm::SparseKernel(3.0));

  for (int epoch = 1; epoch<=num_epochs; epoch++) {
    printf("\n--- EPOCH %02d / %02d (learning rate = %7.5f) ---\n", epoch, num_epochs, opt_kernel.learning_rate);

    // Get data
    printf("\tGetting data...\n");
    Data *data = data_manager.GetData();
    printf("\tHave %ld sample points\n", data->GetPoints()->size());
    printf("\tHave %ld occluded points\n", data->GetOccludedPoints()->size());
    printf("\tHave %ld data observations\n", data->GetHits()->size());

    // This kernel is actually w
    LearnedKernel w_kernel(opt_w.max - opt_w.min, (opt_w.max - opt_w.min)/opt_w.inducing_points_n_dim);

    std::vector<hm::IKernel*> kernels;
    kernels.push_back(&kernel);
    //hm::HilbertMap map_w(*data->GetPoints(), *data->GetLabels(), kernels, opt_w);
    hm::HilbertMap map_w(*data->GetHits(), *data->GetOrigins(), kernels, opt_w);
    std::vector<float> w_vector = map_w.GetW();
    for (int i=0; i<opt_w.inducing_points_n_dim; i++) {
      for (int j=0; j<opt_w.inducing_points_n_dim; j++) {
        int idx = i * opt_w.inducing_points_n_dim + j;
        float val = w_vector[idx];
        w_kernel.SetPixel(i, j, val);
      }
    }

    /*
    printf("\tSetting up kernel...\n");
    for (size_t i = 0; i<w_kernel.GetDimSize(); i++) {
      for (size_t j = 0; j<w_kernel.GetDimSize(); j++) {
        w_kernel.SetPixel(i, j, 0.0f);
      }
    }

    for (const Shape& shape : data->GetSim()->GetObjects()) {
      Eigen::Vector2d center = shape.GetCenter();

      w_kernel.SetLocation(center(0), center(1), 1.0f);
    }
    */

    printf("\tLearning kernel...\n");
    std::vector<hm::IKernel*> w_kernels;
    w_kernels.push_back(&w_kernel);
    auto tic_map = std::chrono::steady_clock::now();
    hm::HilbertMap map_kernel(*data->GetOccludedPoints(), *data->GetOccludedLabels(), w_kernels, opt_kernel, kernel.GetData().data());
    //hm::HilbertMap map_kernel(*data->GetHits(), *data->GetOrigins(), w_kernels, opt_kernel, kernel.GetData().data());
    auto toc_map = std::chrono::steady_clock::now();
    auto t_ms_map = std::chrono::duration_cast<std::chrono::milliseconds>(toc_map - tic_map);
    printf("\tLearned map in %ld ms\n", t_ms_map.count());

    kernel_vector = map_kernel.GetW();

    // Copy kernel out
    for (int i=0; i<opt_kernel.inducing_points_n_dim; i++) {
      for (int j=0; j<opt_kernel.inducing_points_n_dim; j++) {
        int idx = i * opt_kernel.inducing_points_n_dim + j;
        float val = kernel_vector[idx];
        kernel.SetPixel(i, j, val);
      }
    }

    // Save kernel sometimes
    if (true || epoch % 10 == 0) {
      printf("\tSaving kernel...\n");
      char fn[1024];
      sprintf(fn, "kernel_%04d.csv", epoch);
      SaveKernel(kernel, fn);
    }

    // Update opt
    opt_kernel.learning_rate *= decay_rate;
    //if (opt_kernel.learning_rate < 0.001)
    //  break;

    // Cleanup
    delete data;
  }
  data_manager.Finish();

  // Write kernel to file
  SaveKernel(kernel, "kernel.csv");

  printf("\n\nGenerating examples...\n");

  for (int example = 0; example < 3; example++) {
    printf("\tMaking example %d\n", example);

    // Now actually make a HM with the kernel we learned
    std::vector<hm::Point> hits, origins;

    sw::SimWorld sim;
    sim.GenerateSimData(&hits, &origins);

    std::vector<hm::IKernel*> kernels;
    kernels.push_back(&kernel);
    hm::HilbertMap map(hits, origins, kernels, opt_w);

    char fn[1024];

    // Write out data
    std::ofstream points_file;
    sprintf(fn, "points_%02d.csv", example);
    points_file.open(fn);
    for (size_t i=0; i<hits.size(); i++) {
      points_file << hits[i].x << ", " << hits[i].y << std::endl;
    }
    points_file.close();

    // Evaluate
    std::vector<hm::Point> query_points;
    std::vector<float> gt_labels;
    sim.GenerateGrid(10.0, &query_points, &gt_labels);
    std::vector<float> probs = map.GetOccupancy(query_points);

    // Write out ground truth
    std::ofstream gt_file, hm_file;
    sprintf(fn, "ground_truth_%02d.csv", example);
    gt_file.open(fn);
    sprintf(fn, "hilbert_map_%02d.csv", example);
    hm_file.open(fn);

    for (size_t i=0; i<query_points.size(); i++) {
      float x = query_points[i].x;
      float y = query_points[i].y;
      float p_gt = gt_labels[i];
      float p_hm = probs[i];

      gt_file << x << ", " << y << ", " << p_gt << std::endl;
      hm_file << x << ", " << y << ", " << p_hm << std::endl;
    }
    gt_file.close();
    hm_file.close();
  }

  printf("Done\n");

  return 0;
}
