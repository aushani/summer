#include <iostream>
#include <osg/ArgumentParser>

#include "library/viewer/viewer.h"

#include "library/chow_liu_tree/dynamic_clt.h"
#include "library/chow_liu_tree/joint_model.h"

#include "library/osg_nodes/chow_liu_tree.h"

namespace vw = library::viewer;
namespace clt = library::chow_liu_tree;
namespace osgn = library::osg_nodes;

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

  au->addCommandLineOption("--jm <dirname>", "Joint Model Filename", "");

  // handle help text
  // call AFTER init viewer so key bindings have been set
  unsigned int helpType = 0;
  if ((helpType = args.readHelpType())) {
    au->write(std::cout, helpType);
    return EXIT_SUCCESS;
  }

  vw::Viewer v(&args);

  std::string fn_jm;
  if (!args.read("--jm", fn_jm)) {
    printf("No file to render!\n");
    return 1;
  }

  clt::JointModel jm = clt::JointModel::Load(fn_jm.c_str());
  clt::DynamicCLT clt = clt::DynamicCLT(jm);
  clt::Tree t = clt.GetFullTree();
  printf("Mutual Information: %f\n", t[0]->GetTreeMutualInformation());

  osg::ref_ptr<osgn::ChowLiuTree> clt_node = new osgn::ChowLiuTree(clt);
  v.AddChild(clt_node);

  v.Start();

  return 0;
}
