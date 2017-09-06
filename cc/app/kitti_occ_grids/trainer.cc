#include "app/kitti_occ_grids/trainer.h"

namespace app {
namespace kitti_occ_grids {

Trainer::Trainer(const std::string &save_base_fn) :
 save_base_path_(save_base_fn),
 detector_(kRes_, 50, 50),
 og_builder_(200000, kRes_, 100.0),
 camera_cal_("/home/aushani/data/kittidata/extracted/2011_09_26/") {

  // Configure occ grid builder size
  og_builder_.ConfigureSizeInPixels(10, 10, 10); // +- 5 meters

  models_.insert({"Car", clt::JointModel(3.0, 2.0, kRes_)});
  models_.insert({"Cyclist", clt::JointModel(3.0, 2.0, kRes_)});
  models_.insert({"Pedestrian", clt::JointModel(3.0, 2.0, kRes_)});
  models_.insert({"Background", clt::JointModel(3.0, 2.0, kRes_)});

  for (const auto &kv : models_) {
    detector_.AddModel(kv.first, kv.second);
  }

  printf("Initialized all models\n");
}

void Trainer::Run() {
  //library::timer::Timer t;
  //t.Start();
  //detector_.Run(scan.GetHits());
  //printf("Took %5.3f ms to run detector\n", t.GetMs());

  // Find where it's most wrong

  // Update joint models

  // Rinse and repeat

}

} // namespace kitti
} // namespace app
