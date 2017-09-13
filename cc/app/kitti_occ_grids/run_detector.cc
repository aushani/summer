#include <iostream>

#include <osg/ArgumentParser>

#include "app/kitti_occ_grids/detector_app.h"

namespace kog = app::kitti_occ_grids;

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

  au->addCommandLineOption("--kitti-challenge-dir <dirname>", "KITTI challenge data directory", "~/data/kitti_challenge/");
  au->addCommandLineOption("--models <dir>", "Models to evaluate", "");
  au->addCommandLineOption("--num <num>", "KITTI scan number", "0");

  // handle help text
  // call AFTER init viewer so key bindings have been set
  unsigned int helpType = 0;
  if ((helpType = args.readHelpType())) {
    au->write(std::cout, helpType);
    return EXIT_SUCCESS;
  }

  kog::DetectorApp app(&args);

  app.Run();

  return 0;
}
