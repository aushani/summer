#include "library/kitti/tracklets.h"
#include "library/kitti/velodyne_scan.h"
#include "library/timer/timer.h"

#include "app/kitti/model_bank.h"

namespace kt = library::kitti;

using namespace app::kitti;

void GenerateSampleScans(const ObjectState &os, const RayModel &model) {
  char fn[1000];

  for (int i=0; i<10; i++) {
    sprintf(fn, "%s_%02d.csv", os.classname.c_str(), i);
    std::ofstream sample_file(fn);

    for (double sensor_theta = -M_PI; sensor_theta < M_PI; sensor_theta += 0.01) {
      for (double sensor_phi = -M_PI; sensor_phi < M_PI; sensor_phi += 0.01) {

        double range = model.SampleRange(os, sensor_theta, sensor_phi);

        if (range < 0 || range >= 100.0) {
          continue;
        }

        double x = range * cos(sensor_phi) * cos(sensor_theta);
        double y = range * cos(sensor_phi) * sin(sensor_theta);
        double z = range * sin(sensor_phi);

        sample_file << x << ", " << y << ", " << z << std::endl;
      }
    }
  }
}

int main(int argc, char** argv) {
  printf("Generate KITTI sample scans...\n");

  if (argc < 2) {
    printf("Usage: generate_sample_scans model_bank_file\n");
    return 1;
  }

  const char *model_bank_file = argv[1];

  library::timer::Timer t;
  ModelBank mb = ModelBank::LoadModelBank(model_bank_file);
  printf("Took %5.3f sec to load model bank\n", t.GetMs()/1e3);

  auto models = mb.GetModels();
  for (auto it = models.begin(); it != models.end(); it++) {
    const auto& model = it->second;
    printf("Model %s\n", it->first.c_str());

    ObjectState os(Eigen::Vector2d(25.213, 8.603), -3.184, it->first);

    GenerateSampleScans(os, model);
  }
}
