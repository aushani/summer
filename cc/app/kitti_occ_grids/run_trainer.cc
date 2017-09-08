#include "app/kitti_occ_grids/trainer.h"

namespace kog = app::kitti_occ_grids;

int main(int argc, char** argv) {
  printf("Running KITTI Trainer\n");

  if (argc < 2) {
    printf("Usage: %s save_dir [load_dir] [starting epoch] [starting log num]\n", argv[0]);
    return 1;
  }

  std::shared_ptr<kog::Trainer> trainer;

  if (argc == 2) {
    trainer = std::make_shared<kog::Trainer>(argv[1]);
  } else {
    trainer = std::make_shared<kog::Trainer>(argv[1], argv[2]);
  }

  int epoch = 0;
  int log_num = 0;

  if (argc > 3) {
    epoch = atoi(argv[3]);
  }

  if (argc > 4) {
    log_num = atoi(argv[4]);
  }

  printf("Starting from epoch %d log %d\n", epoch, log_num);

  trainer->Run(epoch, log_num);

  return 0;
}
