#include <iostream>

#include <osg/ArgumentParser>
#include <boost/filesystem.hpp>

#include "library/kitti/kitti_challenge_data.h"
#include "library/osg_nodes/point_cloud.h"
#include "library/osg_nodes/object_labels.h"
#include "library/osg_nodes/tracklets.h"
#include "library/ray_tracing/occ_grid_builder.h"
#include "library/timer/timer.h"
#include "library/viewer/viewer.h"

#include "app/kitti_occ_grids/map_node.h"

namespace dt = library::detector;
namespace kt = library::kitti;
namespace rt = library::ray_tracing;
namespace osgn = library::osg_nodes;
namespace vw = library::viewer;
namespace kog = app::kitti_occ_grids;
namespace fs = boost::filesystem;

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

  // Get parameters
  int frame_num = 0;
  if (!args.read(std::string("--num"), frame_num)) {
    printf("Using default KITTI frame number: %d\n", frame_num);
  }

  std::string dirname = "/home/aushani/data/kitti_challenge/";
  if (!args.read(std::string("--kitti-challenge-dir"), dirname)) {
    printf("Using default KITTI dir: %s\n", dirname.c_str());
  }

  std::string model_dir = "/home/aushani/data/trainer/0009/";
  if (!args.read("--models", model_dir)) {
    printf("no model given!\n");
    return EXIT_FAILURE;
  }

  kt::KittiChallengeData kcd = kt::KittiChallengeData::LoadFrame(dirname, frame_num);

  // Detector
  dt::Detector detector(0.5, 50, 50);

  // Load models
  printf("loading models from %s\n", model_dir.c_str());

  fs::directory_iterator end_it;
  for (fs::directory_iterator it(model_dir); it != end_it; it++) {
    // Make sure it's not a directory
    if (!fs::is_regular_file(it->path())) {
      continue;
    }

    // Make sure it's a joint model
    if (fs::extension(it->path()) != ".jm") {
      continue;
    }

    std::string classname = it->path().stem().string();

    if (! (classname == "Car" || classname == "Cyclist" || classname == "Pedestrian" || classname == "Background")) {
      continue;
    }

    printf("Found %s\n", classname.c_str());

    clt::JointModel jm = clt::JointModel::Load(it->path().string().c_str());
    detector.AddModel(classname, jm);
  }
  printf("Loaded all models\n");

  library::timer::Timer t;
  t.Start();
  detector.Run(kcd.GetScan().GetHits());
  printf("Took %5.3f ms to run detector\n", t.GetMs());

  vw::Viewer v(&args);

  osg::ref_ptr<osgn::PointCloud> pc = new osgn::PointCloud(kcd.GetScan());
  osg::ref_ptr<osgn::ObjectLabels> ln = new osgn::ObjectLabels(kcd.GetLabels(), kcd.GetTcv());
  osg::ref_ptr<kog::MapNode> map_node = new kog::MapNode(detector, kcd);

  v.AddChild(pc);
  v.AddChild(ln);
  v.AddChild(map_node);

  v.Start();

  return 0;
}
