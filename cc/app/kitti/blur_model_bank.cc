#include "app/kitti/model_bank.h"

#include <iostream>
#include <fstream>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "library/timer/timer.h"

using namespace app::kitti;

inline void SaveModelBank(const ModelBank &mb, const char *fn) {
  std::ofstream ofs(fn);
  boost::archive::binary_oarchive oa(ofs);
  oa << mb;
}

inline ModelBank LoadModelBank(const char *fn) {
  ModelBank mb;

  std::ifstream ifs(fn);
  boost::archive::binary_iarchive ia(ifs);
  ia >> mb;

  return mb;
}

int main() {
  printf("Blurring model bank\n");

  library::timer::Timer t;
  ModelBank mb = LoadModelBank("model_bank_13");
  printf("Took %5.3f sec to load model bank\n", t.GetSeconds());

  mb.PrintStats();

  t.Start();
  mb.Blur();
  printf("Took %5.3f sec to blur model bank\n", t.GetSeconds());

  t.Start();
  SaveModelBank(mb, "model_bank_13_blurred");
  printf("Took %5.3f sec to save model bank\n", t.GetSeconds());
}
