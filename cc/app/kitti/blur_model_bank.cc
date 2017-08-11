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

int main(int argc, char** argv) {
  printf("Blurring model bank\n");

  if (argc != 3) {
    printf("Usage: blur_model_bank input_file output_file\n");
    return 1;
  }

  library::timer::Timer t;
  ModelBank mb = LoadModelBank(argv[1]);
  printf("Took %5.3f sec to load model bank\n", t.GetSeconds());

  mb.PrintStats();

  t.Start();
  mb.Blur();
  printf("Took %5.3f sec to blur model bank\n", t.GetSeconds());

  t.Start();
  SaveModelBank(mb, argv[2]);
  printf("Took %5.3f sec to save model bank\n", t.GetSeconds());

  return 0;
}
