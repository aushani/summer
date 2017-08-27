#include <iostream>
#include <cmath>

#include "library/timer/timer.h"

int main() {
  size_t nums = 3000;
  size_t num_trials = std::ceil(100/0.3) * std::ceil(100/0.3);
  float log_prob = 0;

  float *junk = (float*) malloc(sizeof(float) * nums * num_trials);

  library::timer::Timer t;
  for (size_t trial = 0; trial < num_trials; trial++) {
    for (size_t i = 0; i < nums; i++) {
      float v = junk[trial*nums + i];
      log_prob += log(fabs(v) + 1);
      //log_prob += junk[trail * nums + i];
    }
  }
  double ms = t.GetMs();
  printf("Took %5.3f ms to sum log's of %ld nums %ld times\n", ms, nums, num_trials);

  printf("Took %5.3f ms per trial\n", ms/num_trials);
}
