
#include "shape_completion/hello-time.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Need filename" << std::endl;
    return 1;
  }

  std::ifstream file;
  file.open(argv[1], std::ios::binary);

  uint64_t dim[3] = {0, 0, 0};
  file.read(reinterpret_cast<char*>(dim), 3*sizeof(uint64_t));

  int n = dim[0]*dim[1]*dim[2];
  float *data = new float[n];
  file.read(reinterpret_cast<char*>(data), n*sizeof(float));

  file.close();

  printf("SDF is %ld x %ld x %ld\n", dim[0], dim[1], dim[2]);

  for (int i=0; i<10; i++) {
    printf("\t[%d] = %5.3f\n", i, data[i]);
  }

  float min = 0.0;
  float max = 0.0;

  float tol = 0.5;
  int count = 0;

  float *og_data = new float[n];

  for (int i=0; i<n; i++) {
    if (std::isfinite(data[i])) {
      if (data[i] < min) min = data[i];
      if (data[i] > max) max = data[i];

      if (std::abs(data[i] < tol)) {
        count++;
        og_data[i] = 0.99;
      } else {
        og_data[i] = 0.01;
      }
    } else {
      og_data[i] = 0.01;
    }
  }

  printf("range: %5.3f - %5.3f\n", min, max);
  printf("%d within %5.3f\n", count, tol);

  std::ofstream og;
  og.open("out.og", std::ios::binary);
  og.write(reinterpret_cast<char*>(og_data), n*sizeof(float));
  og.close();

  delete[] data;
  delete[] og_data;
  return 0;
}
