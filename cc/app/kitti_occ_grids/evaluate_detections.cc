#include <iostream>
#include <thread>

#include "library/timer/timer.h"

#include "app/kitti_occ_grids/evaluator.h"

namespace kog = app::kitti_occ_grids;

int main(int argc, char** argv) {
  printf("Evaluate Detections\n");

  if (argc < 2) {
    printf("Usage: %s gen_data/\n", argv[0]);
    return 1;
  }

  kog::Evaluator evaluator(argv[1]);

  for (const std::string &cn : evaluator.GetClasses()) {
    evaluator.QueueClass(cn);
  }

  evaluator.Start();

  while (evaluator.HaveWork()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    evaluator.PrintConfusionMatrix();
    printf("\n");
  }

  printf("\nDone!\n");

  evaluator.PrintConfusionMatrix();

  return 0;
}
