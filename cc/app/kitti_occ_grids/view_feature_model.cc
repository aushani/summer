#include <iostream>
#include <osg/ArgumentParser>

#include "library/osg_nodes/occ_model.h"
#include "library/feature/feature_model.h"
#include "library/feature/model_bank.h"
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

  au->addCommandLineOption("--mb <dirname>", "Model Bank Filename", "");
  au->addCommandLineOption("--class <classname>", "Classname", "");
  au->addCommandLineOption("--anglebin <dirname>", "Angle Bin", "");

  // handle help text
  // call AFTER init viewer so key bindings have been set
  unsigned int helpType = 0;
  if ((helpType = args.readHelpType())) {
    au->write(std::cout, helpType);
    return EXIT_SUCCESS;
  }

  // Load joint model
  std::string fn;
  if (!args.read("--mb", fn)) {
    printf("Error! Need file to render! (--mb) \n");
    return 1;
  }

  int angle_bin = 0;
  args.read("--anglebin", angle_bin);

  std::string classname;
  if (!args.read("--class", classname)) {
    printf("Error: Need class to render! (--class)\n");
    return 1;
  }

  ft::ModelBank mb = ft::ModelBank::Load(fn.c_str());
  ft::FeatureModel fm = mb.GetFeatureModel(classname, angle_bin);

  vw::Viewer v(&args);

  osg::ref_ptr<osgn::OccModel> node = new osgn::OccModel(fm);
  v.AddChild(node);

  v.Start();

  return 0;
}
