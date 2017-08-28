#include <iostream>
#include <osg/ArgumentParser>

#include "library/kitti/velodyne_scan.h"
#include "library/osg_nodes/point_cloud.h"
#include "library/osg_nodes/occ_grid.h"
#include "library/osg_nodes/object_labels.h"
#include "library/ray_tracing/occ_grid_builder.h"
#include "library/timer/timer.h"
#include "library/viewer/viewer.h"

namespace kt = library::kitti;
namespace rt = library::ray_tracing;
namespace osgn = library::osg_nodes;
namespace vw = library::viewer;

kt::VelodyneScan LoadVelodyneScan(const std::string &dirname, int frame_num) {
  char fn[1000];
  sprintf(fn, "%s/data_object_velodyne/training/velodyne/%06d.bin",
      dirname.c_str(), frame_num);

  printf("Loading velodyne from %s\n", fn);

  return kt::VelodyneScan(fn);
}

kt::ObjectLabels LoadLabels(const std::string &dirname, int frame_num) {
  // Load Labels
  char fn[1000];
  sprintf(fn, "%s/data_object_label_2/training/label_2/%06d.txt",
      dirname.c_str(), frame_num);

  printf("Loading labels from %s\n", fn);

  kt::ObjectLabels labels = kt::ObjectLabel::Load(fn);

  return labels;
}

Eigen::Matrix4d LoadTcv(const std::string &dirname, int frame_num) {
  // Load Labels
  char fn[1000];
  sprintf(fn, "%s/data_object_calib/training/calib/%06d.txt",
      dirname.c_str(), frame_num);

  printf("Loading calib from %s\n", fn);

  Eigen::MatrixXd T_cv = kt::ObjectLabel::LoadVelToCam(fn);

  return T_cv;
}

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

  kt::VelodyneScan scan = LoadVelodyneScan(dirname, frame_num);
  kt::ObjectLabels labels = LoadLabels(dirname, frame_num);
  Eigen::Matrix4d T_cv = LoadTcv(dirname, frame_num);

  printf("Have %ld points\n", scan.GetHits().size());

  // Build occ grid
  rt::OccGridBuilder builder(200000, 0.3, 100.0);

  library::timer::Timer t;
  rt::OccGrid og = builder.GenerateOccGrid(scan.GetHits());
  printf("Took %5.3f ms to build occ grid\n", t.GetMs());

  vw::Viewer v(&args);

  osg::ref_ptr<osgn::PointCloud> pc = new osgn::PointCloud(scan);
  osg::ref_ptr<osgn::OccGrid> ogn = new osgn::OccGrid(og);
  osg::ref_ptr<osgn::ObjectLabels> ln = new osgn::ObjectLabels(labels, T_cv);

  v.AddChild(pc);
  v.AddChild(ogn);
  v.AddChild(ln);

  v.Start();

  return 0;
}
