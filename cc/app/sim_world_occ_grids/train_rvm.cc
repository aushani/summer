#include <iostream>
#include <map>
#include <vector>

#include <boost/assert.hpp>
#include <boost/filesystem.hpp>

#include "library/bayesian_inference/rvm.h"
#include "library/ray_tracing/occ_grid.h"
#include "library/timer/timer.h"

namespace bi = library::bayesian_inference;
namespace rt = library::ray_tracing;
namespace fs = boost::filesystem;
namespace tr = library::timer;

std::vector<rt::OccGrid> LoadDirectory(char *dir, int samples) {
  std::vector<rt::OccGrid> res;

  int count = 0;

  fs::path p(dir);
  fs::directory_iterator end_it;
  for (fs::directory_iterator it(p); it != end_it; it++) {
    if (!fs::is_regular_file(it->path())) {
      continue;
    }

    if (it->path().extension().string() != ".og") {
      printf("skipping extension %s\n", it->path().extension().string().c_str());
      continue;
    }

    rt::OccGrid og = rt::OccGrid::Load(it->path().string().c_str());
    res.push_back(og);

    count++;
    if (count >= samples) {
      break;
    }
  }

  return res;
}

class OccGridKernelMi : public bi::IKernel<rt::OccGrid> {
  virtual double Compute(const rt::OccGrid &sample, const rt::OccGrid &x_m) const {
    std::map<std::pair<bool, bool>, float> joint_histogram;
    std::map<bool, float> sample_histogram;
    std::map<bool, float> xm_histogram;

    int count = 0;
    for (int i=-8; i<8; i++) {
      for (int j=-8; j<8; j++) {
        rt::Location loc(i, j, 0);
        count++;

        float p_occ_sample = sample.GetProbability(loc);
        float p_occ_xm = x_m.GetProbability(loc);

        float p_free_sample = 1 - p_occ_sample;
        float p_free_xm = 1 - p_occ_xm;

        joint_histogram[std::make_pair(true, true)] += p_occ_sample * p_occ_sample;
        joint_histogram[std::make_pair(true, false)] += p_occ_sample * p_free_sample;

        joint_histogram[std::make_pair(false, true)] += p_free_sample * p_occ_sample;
        joint_histogram[std::make_pair(false, false)] += p_free_sample * p_free_sample;

        sample_histogram[true] += p_occ_sample;
        sample_histogram[false] += p_free_sample;

        xm_histogram[true] += p_occ_xm;
        xm_histogram[false] += p_free_xm;
      }
    }

    // Compute mutual informations
    float mi = 0;

    for (int i=0; i<2; i++) {
      bool x_occu = i==0;
      float p_x = sample_histogram[x_occu] / count;

      for (int j=0; j<2; j++) {
        bool y_occu = j==0;

        float p_y = xm_histogram[x_occu] / count;
        float p_xy = joint_histogram[std::make_pair(x_occu, y_occu)] / count;

        mi += p_xy * std::log(p_xy / (p_x * p_y));
      }
    }

    return mi;
  }
};

class OccGridKernelCorr : public bi::IKernel<rt::OccGrid> {
  virtual double Compute(const rt::OccGrid &sample, const rt::OccGrid &x_m) const {
    int count = 0;
    double corr = 0.0;

    for (int i=-8; i<8; i++) {
      for (int j=-8; j<8; j++) {
        rt::Location loc(i, j, 0);
        count++;

        float p_occ_sample = sample.GetProbability(loc);
        float p_occ_xm = x_m.GetProbability(loc);

        float p_free_sample = 1 - p_occ_sample;
        float p_free_xm = 1 - p_occ_xm;

        corr += p_occ_sample*p_occ_xm + p_free_sample*p_free_xm;
      }
    }

    corr /= count;

    //printf("cor is %5.3f\n", corr);

    return corr - 0.5;
  }
};

int main(int argc, char** argv) {
  printf("Make RVM model\n");

  tr::Timer t;

  if (argc < 4) {
    printf("Usage: %s dir_name_pos dir_name_neg samples_per_class out\n", argv[0]);
    return 1;
  }

  int n_samples_per_class = atoi(argv[3]);

  t.Start();
  auto samples_pos = LoadDirectory(argv[1], n_samples_per_class);
  printf("Loaded pos\n");

  auto samples_neg = LoadDirectory(argv[2], n_samples_per_class);
  printf("Loaded neg\n");

  std::vector<rt::OccGrid> samples;
  samples.insert(samples.end(), samples_pos.begin(), samples_pos.end());
  samples.insert(samples.end(), samples_neg.begin(), samples_neg.end());

  printf("Took %5.3f ms to load %ld samples\n", t.GetMs(), samples.size());

  std::vector<int> labels;
  for (int i=0; i<2*n_samples_per_class; i++) {
    labels.push_back(i < n_samples_per_class ? 1:0);
  }

  OccGridKernelCorr kernel;

  printf("Making model...\n");
  t.Start();
  bi::Rvm<rt::OccGrid> model = bi::Rvm<rt::OccGrid>(samples, labels, &kernel);
  printf("Took %5.3f ms to make model\n", t.GetMs());

  printf("Solving model...\n");
  t.Start();
  model.Solve(1000);
  printf("Took %5.3f ms to solve model\n", t.GetMs());

  // Predict and test
  t.Start();
  std::vector<double> pred_labels = model.PredictLabels(samples);
  printf("Took %5.3f ms to predict %ld og's\n", t.GetMs(), samples.size());

  int correct = 0;
  for (size_t i=0; i<pred_labels.size(); i++) {
    //printf("%5.3f vs %d\n", pred_labels[i], labels[i]);
    if ( (pred_labels[i] > 0.5) == (labels[i] > 0.5)) {
      correct++;
    }
  }
  printf("%d / %ld = %5.3f %% correct\n", correct, pred_labels.size(), 100.0 * correct / pred_labels.size());

  auto rvs = model.GetRelevanceVectors();
  printf("Have %ld relevance vectors\n", rvs.size());

  // Save
  printf("Done! Saving to %s...\n", argv[4]);

  for (size_t i=0; i<rvs.size(); i++) {
    const auto &rv = rvs[i];
    char fn[1000];
    sprintf(fn, "%s/%06ld.og", argv[4], i);

    rv.Save(fn);
  }

  return 0;
}
