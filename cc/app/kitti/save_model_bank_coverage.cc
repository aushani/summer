#include "app/kitti/model_bank.h"

#include <iostream>
#include <fstream>

#include "library/timer/timer.h"

using namespace app::kitti;

int main(int argc, char** argv) {
  printf("Save model bank coverage\n");

  if (argc < 2) {
    printf("Usage: save_model_bank_coverage input_file [prefix]\n");
    return 1;
  }

  const char *prefix = "";
  if (argc > 2) {
    prefix = argv[2];
  }

  library::timer::Timer t;
  ModelBank mb = ModelBank::LoadModelBank(argv[1]);
  printf("Took %5.3f sec to load model bank\n", t.GetSeconds());

  auto models = mb.GetModels();
  for (auto it = models.begin(); it != models.end(); it++) {
    const auto& model = it->second;

    auto counts = model.GetHistogramFillinByAngle();
    auto median_weight = model.GetHistogramMedianWeight();

    char fn[1000];
    sprintf(fn, "%s%s%s.csv",
        prefix, strlen(prefix) > 0 ? "_":"", it->first.c_str());
    std::ofstream stats_file(fn);
    for (auto it = counts.begin(); it != counts.end(); it++) {
      double theta = it->first.first;
      double phi = it->first.second;
      double mw = median_weight[it->first];
      stats_file << theta << ", " << phi << ", " << it->second << ", " << mw << std::endl;
    }
    stats_file.close();
  }

  return 0;
}
