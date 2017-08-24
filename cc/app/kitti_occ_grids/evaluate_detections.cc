#include <iostream>
#include <thread>

#include "library/timer/timer.h"

#include "app/kitti_occ_grids/evaluator.h"

namespace kog = app::kitti_occ_grids;

int main(int argc, char** argv) {
  printf("Evaluate Detections\n");

  if (argc < 4) {
    printf("Usage: %s training_data/ testing_data/ eval_type\n", argv[0]);
    printf("                                                \n");
    printf("      eval_type   :=    [ LOTP | SC | MARGINAL ]\n");
    return 1;
  }

  kog::ChowLuiTree::EvalType type = kog::ChowLuiTree::EvalType::MARGINAL;
  std::string string_type(argv[3]);
  if (string_type == "LOTP") {
    type = kog::ChowLuiTree::EvalType::LOTP;
  } else if (string_type == "SC") {
    type = kog::ChowLuiTree::EvalType::SC;
  } else if (string_type == "MARGINAL") {
    type = kog::ChowLuiTree::EvalType::MARGINAL;
  } else {
    printf("eval_type not valid!\n");
    return 1;
  }

  kog::Evaluator evaluator(argv[1], argv[2], type);

  for (const std::string &cn : evaluator.GetClasses()) {
    evaluator.QueueClass(cn);
  }

  evaluator.Start();

  while (evaluator.HaveWork()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    printf("\nIn Progress Results for %s %s %s\n", argv[1], argv[2], argv[3]);
    evaluator.PrintConfusionMatrix();
    printf("\n");
  }

  printf("\nFinal Results for %s %s %s\n", argv[1], argv[2], argv[3]);

  evaluator.PrintConfusionMatrix();

  return 0;
}
