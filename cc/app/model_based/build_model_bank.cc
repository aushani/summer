#include "model_bank_builder.h"

int main(int argc, char** argv) {
  printf("Building model bank...\n");

  if (argc != 3) {
    printf("Usage: build_model_bank min_per_save filename\n");
    return 1;
  }

  int n_min = strtol(argv[1], NULL, 10);

  ModelBankBuilder mbb;

  int step = 0;
  char fn[1000];

  while (true) {
    std::this_thread::sleep_for(std::chrono::minutes(n_min));

    printf("Saving step %d\n", step);
    sprintf(fn, "%s_%06d", argv[2], step++);
    mbb.SaveModelBank(std::string(fn));
  }
}
