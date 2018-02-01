#include <iostream>
#include <fstream>
#include <osg/ArgumentParser>

#include "library/osg_nodes/occ_grid.h"
#include "library/ray_tracing/occ_grid.h"
#include "library/ray_tracing/feature_occ_grid.h"
#include "library/viewer/viewer.h"

namespace rt = library::ray_tracing;
namespace osgn = library::osg_nodes;
namespace vw = library::viewer;


// XXX Hack
rt::OccGrid LoadBin(const char* fn) {
  std::ifstream file(fn, std::ios::in | std::ios::binary);

  int kXYRangePixels_ = 16*3;                // +- 4.8 * 3 m
  int kZRangePixels_ = 16;                   // +- 4.8 m
  float res = 0.3;

  std::vector<rt::Location> locs;
  std::vector<float> log_odds;

  for (int i=-kXYRangePixels_; i<=kXYRangePixels_; i++) {
    for (int j=-kXYRangePixels_; j<=kXYRangePixels_; j++) {
      for (int k=-kZRangePixels_; k<=kZRangePixels_; k++) {
        float p = -1.0;

        // gross
        file.read((char*)(&p), sizeof(float));

        if (p > 0.51 || p < 0.49) {
          printf("p: %5.3f\n", p);
          float lo = p > 0.5 ? 1:-1;

          locs.emplace_back(i, j, k);
          log_odds.push_back(lo);
        }
      }
    }
  }

  file.close();

  return rt::OccGrid(locs, log_odds, res);
}

int main(int argc, char** argv) {
  osg::ArgumentParser args(&argc, argv);
  osg::ApplicationUsage* au = args.getApplicationUsage();

  // report any errors if they have occurred when parsing the program arguments.
  if (args.errors()) {
    args.writeErrorMessages(std::cout);
    return EXIT_FAILURE;
  }

  au->setApplicationName(args.getApplicationName());
  au->setCommandLineUsage( args.getApplicationName() + " [options]");
  au->setDescription(args.getApplicationName() + " viewer");

  au->addCommandLineOption("--og <dirname>", "Occ Grid Filename", "");
  au->addCommandLineOption("--fog <dirname>", "Occ Grid Filename", "");
  au->addCommandLineOption("--bin <dirname>", "Occ Grid Bin Filename", "");

  // handle help text
  // call AFTER init viewer so key bindings have been set
  unsigned int helpType = 0;
  if ((helpType = args.readHelpType())) {
    au->write(std::cout, helpType);
    return EXIT_SUCCESS;
  }

  // Load occ grid
  std::string fn;
  if (args.read("--og", fn)) {
    rt::OccGrid og = rt::OccGrid::Load(fn.c_str());

    vw::Viewer v(&args);
    osg::ref_ptr<osgn::OccGrid> ogn = new osgn::OccGrid(og);
    v.AddChild(ogn);
    v.Start();
  } else if (args.read("--bin", fn)) {
    rt::OccGrid og = LoadBin(fn.c_str());

    vw::Viewer v(&args);
    osg::ref_ptr<osgn::OccGrid> ogn = new osgn::OccGrid(og);
    v.AddChild(ogn);
    v.Start();
  } else if (args.read("--fog", fn)) {
    rt::FeatureOccGrid fog = rt::FeatureOccGrid::Load(fn.c_str());

    vw::Viewer v(&args);
    osg::ref_ptr<osgn::OccGrid> ogn = new osgn::OccGrid(fog);
    v.AddChild(ogn);
    v.Start();
  } else {
    printf("No --og or --fog given!\n");
  }

  return 0;
}
