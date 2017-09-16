#include <iostream>
#include <osg/ArgumentParser>

#include "library/osg_nodes/occ_model.h"
#include "library/feature/feature_model.h"
#include "library/viewer/viewer.h"

namespace ft = library::feature;
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

  au->addCommandLineOption("--fm <dirname>", "Model Filename", "");

  // handle help text
  // call AFTER init viewer so key bindings have been set
  unsigned int helpType = 0;
  if ((helpType = args.readHelpType())) {
    au->write(std::cout, helpType);
    return EXIT_SUCCESS;
  }

  // Load joint model
  std::string fn;
  if (!args.read("--fm", fn)) {
    printf("Error! Need file to render! (--fm) \n");
    return 1;
  }

  ft::FeatureModel fm = ft::FeatureModel::Load(fn.c_str());

  printf("%d x %d x %d\n", fm.GetNXY(), fm.GetNXY(), fm.GetNZ());

  vw::Viewer v(&args);

  osg::ref_ptr<osgn::OccModel> node = new osgn::OccModel(fm);
  v.AddChild(node);

  v.Start();

  return 0;
}
