#include <iostream>
#include <thread>

#include "library/timer/timer.h"

#include "app/sim_world_occ_grids/evaluator.h"

namespace swog = app::sim_world_occ_grids;

int main(int argc, char** argv) {
  printf("Evaluate Detections\n");

  if (argc < 2) {
    printf("Usage: %s training_data/ testing_data/ \n", argv[0]);
    return 1;
  }

  swog::Evaluator evaluator(argv[1], argv[2]);

  for (const std::string &cn : evaluator.GetClasses()) {
    evaluator.QueueClass(cn);
  }

  evaluator.Start();

  while (evaluator.HaveWork()) {
    std::this_thread::sleep_for(std::chrono::seconds(10));
    evaluator.PrintResults();
    printf("\n----\n");
  }

  printf("\nDone!\n");

  evaluator.PrintResults();

  return 0;
}
