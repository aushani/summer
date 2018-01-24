#include "app/sim_world_occ_grids/occ_grid_extractor.h"

namespace swog = app::sim_world_occ_grids;

int main(int argc, char** argv) {
  printf("Running Occ Grid Extractor with better sampling\n");

  if (argc < 2) {
    printf("Usage: %s save_dir\n", argv[0]);
    return 1;
  }

  swog::OccGridExtractor oge(argv[1]);
  oge.Run();

  return 0;
}
