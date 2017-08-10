#include <iostream>
#include <fstream>

#include <boost/archive/binary_iarchive.hpp>

#include "library/timer/timer.h"

#include "app/kitti/model_bank.h"

using namespace app::kitti;

inline ModelBank LoadModelBank(const char *fn) {
  ModelBank mb;

  std::ifstream ifs(fn);
  boost::archive::binary_iarchive ia(ifs);
  ia >> mb;

  return mb;
}

void GenerateSyntheticScans(const ModelBank &mb) {

}

int main() {
  printf("Running KITTI...\n");

  library::timer::Timer t;
  ModelBank mb = LoadModelBank("model_bank_13");
  printf("Took %5.3f sec to load model bank\n", t.GetMs()/1e3);

  mb.PrintStats();
}
