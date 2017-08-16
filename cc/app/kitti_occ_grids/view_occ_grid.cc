#include <iostream>
#include <osg/ArgumentParser>

#include "library/osg_nodes/occ_grid.h"
#include "library/ray_tracing/occ_grid.h"
#include "library/viewer/viewer.h"

namespace rt = library::ray_tracing;
namespace osgn = library::osg_nodes;
namespace vw = library::viewer;

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

  au->addCommandLineOption("--filename <dirname>", "Occ Grid Filename", "");

  // handle help text
  // call AFTER init viewer so key bindings have been set
  unsigned int helpType = 0;
  if ((helpType = args.readHelpType())) {
    au->write(std::cout, helpType);
    return EXIT_SUCCESS;
  }

  // Load occ grid
  std::string fn;
  if (!args.read("--filename", fn)) {
    printf("Error! Need file to render!\n");
    return 1;
  }

  rt::OccGrid og = rt::OccGrid::Load(fn.c_str());

  vw::Viewer v(&args);

  osg::ref_ptr<osgn::OccGrid> ogn = new osgn::OccGrid(og);

  v.AddChild(ogn);

  v.Start();

  return 0;
}
