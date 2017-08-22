#include <iostream>
#include <osg/ArgumentParser>

#include "library/viewer/viewer.h"

#include "app/kitti_occ_grids/chow_lui_tree.h"
#include "app/kitti_occ_grids/chow_lui_tree_node.h"
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

  kog::ChowLuiTree clt = kog::ChowLuiTree::Load(fn_clt.c_str());
  osg::ref_ptr<kog::ChowLuiTreeNode> clt_node = new kog::ChowLuiTreeNode(clt);
  v.AddChild(clt_node);

  v.Start();

  return 0;
}
