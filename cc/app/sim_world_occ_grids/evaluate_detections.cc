#include <iostream>
#include <thread>

#include "library/timer/timer.h"

#include "app/sim_world_occ_grids/evaluator.h"

namespace kog = app::sim_world_occ_grids;

int main(int argc, char** argv) {
  printf("Evaluate Detections\n");

  if (argc < 3) {
    printf("Usage: %s training_data/ testing_data/ \n", argv[0]);
    return 1;
  }

  kog::Evaluator evaluator(argv[1], argv[2]);

  for (const std::string &cn : evaluator.GetClasses()) {
    evaluator.QueueClass(cn);
  }

  evaluator.Start();

  while (evaluator.HaveWork()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    evaluator.PrintResults();
    printf("\n");
  }

  printf("\nDone!\n");

  evaluator.PrintResults();

  return 0;
}
