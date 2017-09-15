#include <iostream>

#include <boost/assert.hpp>
#include <boost/filesystem.hpp>

#include "library/ray_tracing/occ_grid.h"
#include "library/timer/timer.h"

namespace rt = library::ray_tracing;
namespace fs = boost::filesystem;

void Write(const rt::OccGrid &og, const char *fn) {
  FILE *fp = fopen(fn, "w");

  bool first = true;

  for (int i=-15; i<15; i++) {
    for (int j=-15; j<15; j++) {
      for (int k=-10; k<10; k++) {
        rt::Location loc(i, j, k);
        bool occu = og.GetProbability(loc) > 0.6;
        bool free = og.GetProbability(loc) < 0.4;

        fprintf(fp, "%s%d\n%d",  first ? "":"\n",
            occu ? 1:0, free ? 1:0);

        first = false;
      }
    }
  }

  fclose(fp);
}

int main(int argc, char** argv) {
  printf("Make dense occ gridsl\n");

  if (argc < 3) {
    printf("Usage: %s dir_name out\n", argv[0]);
    return 1;
  }

  int count = 0;
  library::timer::Timer t;
  library::timer::Timer t_step;

  fs::path out_base(argv[2]);
  fs::create_directories(out_base);

  fs::path p(argv[1]);
  fs::directory_iterator end_it;
  for (fs::directory_iterator it(p); it != end_it; it++) {
    if (!fs::is_regular_file(it->path())) {
      continue;
    }

    if (it->path().extension().string() != ".og") {
      printf("skipping extension %s\n", it->path().extension().string().c_str());
      continue;
    }

    rt::OccGrid og = rt::OccGrid::Load(it->path().string().c_str());

    fs::path p_out = out_base / (it->path().stem().string() + ".dog");

    Write(og, p_out.string().c_str());

    count++;

    if (t_step.GetSeconds() > 60) {
      printf("Saved %d (%9.5f sec per og)\n", count, t.GetSeconds() / count);
      t_step.Start();
    }
  }

  printf("Done!\n");
}
