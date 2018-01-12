#include "app/sim_world_occ_grids/environment_generator.h"

namespace swog = app::sim_world_occ_grids;

int main(int argc, char** argv) {
  printf("Running Gen Environments\n");

  if (argc < 2) {
    printf("Usage: %s save_dir\n", argv[0]);
    return 1;
  }

  swog::EnvironmentGenerator eg(argv[1]);
  eg.Run();

  printf("Done\n");

  return 0;
}
