#include <iostream>
#include <osg/ArgumentParser>

#include "library/viewer/viewer.h"

#include "app/kitti_occ_grids/model.h"
#include "app/kitti_occ_grids/model_node.h"

namespace vw = library::viewer;

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

  au->addCommandLineOption("--filename <dirname>", "Model Filename", "");

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

  kog::Model model = kog::Model::Load(fn.c_str());

  vw::Viewer v(&args);

  osg::ref_ptr<kog::ModelNode> node = new kog::ModelNode(model);

  v.AddChild(node);

  v.Start();

  return 0;
}
