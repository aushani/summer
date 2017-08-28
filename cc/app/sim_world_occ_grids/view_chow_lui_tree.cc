#include <iostream>
#include <osg/ArgumentParser>

#include "library/viewer/viewer.h"

#include "app/sim_world_occ_grids/chow_lui_tree.h"
//#include "app/sim_world_occ_grids/chow_lui_tree_osg_node.h"
#include "app/sim_world_occ_grids/chow_lui_tree_node.h"

namespace vw = library::viewer;
namespace swog = app::sim_world_occ_grids;

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

  au->addCommandLineOption("--clt <dirname>", "CLT Filename", "");

  // handle help text
  // call AFTER init viewer so key bindings have been set
  unsigned int helpType = 0;
  if ((helpType = args.readHelpType())) {
    au->write(std::cout, helpType);
    return EXIT_SUCCESS;
  }

  vw::Viewer v(&args);

  std::string fn_clt;
  if (!args.read("--clt", fn_clt)) {
    printf("No CLT file to render!\n");
    return 1;
  }

  swog::ChowLuiTree clt = swog::ChowLuiTree::Load(fn_clt.c_str());
  //osg::ref_ptr<swog::ChowLuiTreeOSGNode> clt_node = new swog::ChowLuiTreeOSGNode(clt);
  osg::ref_ptr<swog::ChowLuiTreeNode> clt_node = new swog::ChowLuiTreeNode(clt);
  v.AddChild(clt_node);

  v.Start();

  return 0;
}
