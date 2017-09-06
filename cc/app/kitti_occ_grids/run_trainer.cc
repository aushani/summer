#include "app/kitti_occ_grids/trainer.h"

namespace kog = app::kitti_occ_grids;

int main(int argc, char** argv) {
  printf("Running KITTI Trainer\n");

  if (argc < 2) {
    printf("Usage: %s save_dir\n", argv[0]);
    return 1;
  }

  kog::Trainer oge(argv[1]);
  oge.Run();

  return 0;
}
