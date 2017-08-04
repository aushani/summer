#include "model_bank_builder.h"

#include <iostream>
#include <fstream>

#include <boost/archive/binary_iarchive.hpp>

ModelBank LoadModelBank(const char *fn) {
  ModelBank model_bank;
  std::ifstream ifs(fn);
  boost::archive::binary_iarchive ia(ifs);
  ia >> model_bank;

  return model_bank;
}

int main(int argc, char** argv) {
  printf("Building model bank...\n");

  if (argc < 3) {
    printf("Usage: build_model_bank min_per_save filename [prev_model_bank]\n");
    return 1;
  }

  int n_min = strtol(argv[1], NULL, 10);

  ModelBankBuilder *mbb;

  if (argc == 4) {
    printf("\tLoading %s\n", argv[3]);

    ModelBank mb = LoadModelBank(argv[3]);
    mbb = new ModelBankBuilder(mb);
  } else {
    mbb = new ModelBankBuilder();
  }

  int step = 0;
  char fn[1000];

  while (true) {
    std::this_thread::sleep_for(std::chrono::minutes(n_min));

    printf("Saving step %d\n", step);
    sprintf(fn, "%s_%06d", argv[2], step++);
    mbb->SaveModelBank(std::string(fn));
  }

  delete mbb;
}
