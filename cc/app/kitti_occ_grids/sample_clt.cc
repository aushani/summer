#include <algorithm>
#include <iostream>

#include <osg/ArgumentParser>

#include "library/osg_nodes/occ_grid.h"
#include "library/timer/timer.h"
#include "library/viewer/viewer.h"

#include "app/kitti_occ_grids/chow_lui_tree.h"
#include "app/kitti_occ_grids/model.h"
#include "app/kitti_occ_grids/model_node.h"

namespace osgn = library::osg_nodes;
namespace rt = library::ray_tracing;
namespace vw = library::viewer;

namespace kog = app::kitti_occ_grids;

class MyKeyboardEventHandler : public osgGA::GUIEventHandler {
 protected:
  vw::Viewer *viewer_;

  kog::ChowLuiTree clt_;

 public:
  MyKeyboardEventHandler(vw::Viewer *v, const std::string &fn_clt) :
   osgGA::GUIEventHandler(), viewer_(v), clt_(kog::ChowLuiTree::Load(fn_clt.c_str())) {
    MakeSample();
  }

  void MakeSample() {
    library::timer::Timer t;
    auto og = clt_.Sample();
    printf("Took %5.3f sec to make sample\n", t.GetSeconds());

    osg::ref_ptr<osgn::OccGrid> ogn = new osgn::OccGrid(og);

    viewer_->RemoveAllChildren();
    viewer_->AddChild(ogn);
  }

  /**
      OVERRIDE THE HANDLE METHOD:
      The handle() method should return true if the event has been dealt with
      and we do not wish it to be handled by any other handler we may also have
      defined. Whether you return true or false depends on the behaviour you
      want - here we have no other handlers defined so return true.
  **/
  virtual bool handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa, osg::Object* obj,
                      osg::NodeVisitor* nv) {
    switch (ea.getEventType()) {
      case osgGA::GUIEventAdapter::KEYDOWN: {
        switch (ea.getKey()) {
          case 'r':
            MakeSample();
            return true;
        }

        default:
          return false;
      }
    }
  }
};

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


  std::string fn_clt;
  if (!args.read("--clt", fn_clt)) {
    printf("No CLT file to render!\n");
    return 1;
  }

  vw::Viewer v(&args);
  v.AddHandler(new MyKeyboardEventHandler(&v, fn_clt));

  v.Start();

  return 0;
}
